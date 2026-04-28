# pip install pandas numpy scikit-learn tqdm torch transformers accelerate
#
# This script continues fine-tuning the saved base RoBERTa checkpoint on hard
# rows derived from base-model validation errors and uncertain predictions.
# It intentionally does not read test data or generate a submission.

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
EPOCHS = 5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.05
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3

THRESHOLD_GRID = np.round(np.arange(0.05, 0.801, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BASE_MODEL_DIR = ROOT_DIR / "outputs" / "roberta_base" / "best_model"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_hard_specialist"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

MODEL_NAME_OR_PATH = BASE_MODEL_DIR

TRAIN_PATH = DATA_DIR / "train_hard_roberta_base.jsonl"

METRICS_PATH = OUTPUT_DIR / "roberta_hard_specialist_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_hard_specialist_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_hard_specialist_error_analysis.csv"
THRESHOLD_TABLE_PATH = OUTPUT_DIR / "roberta_hard_specialist_threshold_table.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_hard_specialist_notes.md"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def normalize_text(value: object) -> str:
    if is_missing(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_texts(values: Iterable[object], desc: str) -> List[str]:
    normalized = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        normalized.append(normalize_text(value))
    return normalized


def normalize_label(value: object) -> str:
    if is_missing(value):
        raise ValueError("Found missing label value.")

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"

    raise ValueError(f"Unexpected label value: {value}")


def encode_labels(values: Iterable[object], desc: str) -> np.ndarray:
    encoded = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        encoded.append(1 if normalize_label(value) == "TRUE" else 0)
    return np.asarray(encoded, dtype=np.int64)


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def validate_train_df(train_df: pd.DataFrame) -> None:
    required_cols = {"text", "label"}
    missing_cols = sorted(required_cols - set(train_df.columns))
    if missing_cols:
        raise ValueError(f"Hard train file is missing required columns: {missing_cols}")

    normalized_labels = train_df["label"].map(normalize_label)
    bad_labels = sorted(set(normalized_labels) - {"TRUE", "FALSE"})
    if bad_labels:
        raise ValueError(f"Unexpected labels after normalization: {bad_labels}")


def validate_base_model_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Base model checkpoint directory not found: {path}")

    required_files = ["config.json"]
    missing_files = [name for name in required_files if not (path / name).exists()]
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    if not has_weights:
        missing_files.append("model.safetensors or pytorch_model.bin")

    if missing_files:
        raise FileNotFoundError(f"Base model checkpoint is missing required files: {missing_files}")


def load_tokenizer_with_fallback() -> tuple[AutoTokenizer, str]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME_OR_PATH), use_fast=True)
        return tokenizer, str(MODEL_NAME_OR_PATH)
    except Exception as exc:
        print("Warning: could not load tokenizer from the saved base checkpoint.")
        print(f"Tokenizer load error: {exc}")
        print("Falling back to the original roberta-base tokenizer.")
        print("This is valid because the saved checkpoint is a fine-tuned roberta-base model.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        return tokenizer, MODEL_NAME


def tokenize_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    return input_ids, attention_mask


def safe_roc_auc(y_true: np.ndarray, prob_true: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob_true))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, prob_true: np.ndarray, threshold: float) -> dict:
    pred = (prob_true >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, prob_true),
        "confusion_matrix": {
            "labels": ["FALSE", "TRUE"],
            "matrix": cm.tolist(),
            "format": "[[TN, FP], [FN, TP]]",
        },
        "prediction_distribution": {
            "pred_FALSE": int((pred == 0).sum()),
            "pred_TRUE": int((pred == 1).sum()),
        },
    }


def tune_threshold_for_f1(y_true: np.ndarray, prob_true: np.ndarray, thresholds: np.ndarray) -> tuple[float, List[dict]]:
    records = []
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in tqdm(thresholds, desc="Threshold search (validation)", unit="threshold"):
        pred = (prob_true >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "pred_FALSE": int((pred == 0).sum()),
                "pred_TRUE": int((pred == 1).sum()),
            }
        )

        tie_break_current = abs(float(threshold) - 0.5)
        tie_break_best = abs(best_threshold - 0.5)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and tie_break_current < tie_break_best):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, records


def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            loss_value = float(outputs.loss.detach().cpu().item())
            if math.isfinite(loss_value):
                batch_size = input_ids.size(0)
                total_loss += loss_value * batch_size
                total_examples += batch_size

    prob_true = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    avg_loss = total_loss / max(total_examples, 1)
    return avg_loss, prob_true, y_true


def build_optimizer(model: AutoModelForSequenceClassification) -> AdamW:
    no_decay_terms = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        eps=ADAM_EPS,
    )


def make_error_analysis(val_pred_df: pd.DataFrame) -> pd.DataFrame:
    errors = val_pred_df[val_pred_df["true_label"] != val_pred_df["pred_label"]].copy()
    if errors.empty:
        return pd.DataFrame(
            columns=["text", "true_label", "pred_label", "pred_prob_TRUE", "error_type", "confidence"]
        )

    errors["error_type"] = np.where(
        (errors["true_label"] == "FALSE") & (errors["pred_label"] == "TRUE"),
        "false_positive",
        "false_negative",
    )
    errors["confidence"] = np.where(
        errors["error_type"] == "false_positive",
        errors["pred_prob_TRUE"],
        1.0 - errors["pred_prob_TRUE"],
    )

    return errors.sort_values("confidence", ascending=False)[
        ["text", "true_label", "pred_label", "pred_prob_TRUE", "error_type", "confidence"]
    ]


def write_notes(
    device: torch.device,
    tokenizer_source: str,
    train_rows: int,
    val_rows: int,
    label_distribution: dict,
    best_epoch: int,
    final_val_metrics: dict,
    best_threshold: float,
    early_stopped: bool,
) -> None:
    notes_text = f"""# RoBERTa Hard-Boundary Specialist Notes

## Purpose

This is a boundary-specialist model trained only on hard rows derived from the saved base RoBERTa validation predictions.

This is not training a fresh RoBERTa. It is continued fine-tuning of the already-trained base RoBERTa on hard boundary rows.

It is not intended to replace `outputs/roberta_base/roberta_base_submission.csv`. The intended future use is a router:

- use the base RoBERTa model for confident rows
- use this specialist only for uncertain decision-boundary rows

## Data

- Base checkpoint: `outputs/roberta_base/best_model/`
- Training file: `data/train_hard_roberta_base.jsonl`
- No test data is read by this script.
- No final submission is generated by this script.
- Hard subset rows: `{train_rows + val_rows}`
- Train split rows: `{train_rows}`
- Validation split rows: `{val_rows}`
- Label distribution: `{label_distribution}`

## Configuration

- Base architecture: `{MODEL_NAME}`
- Loaded checkpoint: `{MODEL_NAME_OR_PATH}`
- Tokenizer source: `{tokenizer_source}`
- Seed: `{SEED}`
- Device: `{device}`
- Max length: `{MAX_LENGTH}`
- Epochs configured: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Eval batch size: `{EVAL_BATCH_SIZE}`
- Gradient accumulation steps: `{GRADIENT_ACCUMULATION_STEPS}`
- Effective train batch size: `{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Weight decay: `{WEIGHT_DECAY}`
- Adam epsilon: `{ADAM_EPS}`
- Max grad norm: `{MAX_GRAD_NORM}`
- Early stopping patience: `{EARLY_STOPPING_PATIENCE}` epochs without threshold-tuned validation F1 improvement

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Best checkpoint selected by threshold-tuned validation F1, not F1 at threshold 0.5
- Threshold tuned after every epoch over 0.05-0.80 for best validation F1
- F1 at threshold 0.5 is logged only as a diagnostic because this specialist can output low probabilities early
- Early stopped: `{early_stopped}`
- Best epoch: `{best_epoch}`
- Selected threshold: `{best_threshold:.2f}`

## Intended Use

This model is not meant to be submitted alone. It should later be used in a router only for uncertain base-model rows, while the base RoBERTa model handles confident rows.

## Final Validation Metrics

- Accuracy: `{final_val_metrics['accuracy']:.6f}`
- Precision: `{final_val_metrics['precision']:.6f}`
- Recall: `{final_val_metrics['recall']:.6f}`
- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_val_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{final_val_metrics['prediction_distribution']}`

## Outputs

- `outputs/roberta_hard_specialist/best_model/`
- `outputs/roberta_hard_specialist/roberta_hard_specialist_metrics.json`
- `outputs/roberta_hard_specialist/roberta_hard_specialist_val_predictions.csv`
- `outputs/roberta_hard_specialist/roberta_hard_specialist_error_analysis.csv`
- `outputs/roberta_hard_specialist/roberta_hard_specialist_threshold_table.csv`
- `outputs/roberta_hard_specialist/roberta_hard_specialist_notes.md`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Hard-Boundary Specialist ===")
    print("This script continues fine-tuning the saved base RoBERTa on hard validation-derived rows only.")
    print("It does not read test data and does not generate a submission.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Base model checkpoint: {MODEL_NAME_OR_PATH}")
    validate_base_model_dir(BASE_MODEL_DIR)

    print("\n=== Loading hard dataset ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading hard train JSONL")
    print(f"Hard dataset shape: {train_df.shape}")
    validate_train_df(train_df)
    print("Hard dataset schema checks passed.")

    print("\n=== Preparing text and labels ===")
    texts = normalize_texts(train_df["text"], desc="Normalizing hard train text")
    y = encode_labels(train_df["label"], desc="Encoding hard train labels")

    label_distribution = {
        "FALSE": int((y == 0).sum()),
        "TRUE": int((y == 1).sum()),
    }
    print(f"Hard dataset label distribution: {label_distribution}")

    idx = np.arange(len(texts))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")
    print(
        "Validation label distribution: "
        f"FALSE={int((y_val == 0).sum())}, TRUE={int((y_val == 1).sum())}"
    )

    print("\n=== Loading tokenizer/model from base checkpoint ===")
    tokenizer, tokenizer_source = load_tokenizer_with_fallback()
    print(f"Tokenizer source: {tokenizer_source}")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_NAME_OR_PATH), num_labels=2)
    model.to(device)

    print("\n=== Tokenizing ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing hard train split",
    )
    val_input_ids, val_attention_mask = tokenize_texts(
        val_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing hard validation split",
    )

    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

    pin_memory = device.type == "cuda"
    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = build_optimizer(model)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(f"Effective train batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Total optimizer steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    print("\n=== Training ===")
    best_epoch = -1
    best_epoch_tuned_f1 = -1.0
    best_epoch_threshold = 0.5
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    best_epoch_threshold_table: List[dict] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    epoch_history: List[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                non_finite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(
                    train_loss="non_finite",
                    skipped_loss=non_finite_loss_batches,
                    skipped_grad=non_finite_grad_steps,
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                continue

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            batch_size = input_ids.size(0)
            running_loss += loss_value * batch_size
            seen_examples += batch_size

            if (step % GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norm_value = (
                    float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                )

                if not math.isfinite(grad_norm_value):
                    non_finite_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(
                        train_loss=f"{(running_loss / max(seen_examples, 1)):.4f}",
                        skipped_loss=non_finite_loss_batches,
                        skipped_grad=non_finite_grad_steps,
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = running_loss / max(seen_examples, 1)
            progress.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                skipped_loss=non_finite_loss_batches,
                skipped_grad=non_finite_grad_steps,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(seen_examples, 1)

        val_loss, val_prob_true, val_true = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{EPOCHS} - validation",
        )

        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        epoch_best_threshold, epoch_threshold_table = tune_threshold_for_f1(
            y_true=val_true,
            prob_true=val_prob_true,
            thresholds=THRESHOLD_GRID,
        )
        epoch_tuned_metrics = compute_metrics(val_true, val_prob_true, threshold=epoch_best_threshold)

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_metrics_05["roc_auc"]),
            "val_f1_at_0_5": float(val_metrics_05["f1"]),
            "val_precision_at_0_5": float(val_metrics_05["precision"]),
            "val_recall_at_0_5": float(val_metrics_05["recall"]),
            "best_tuned_f1": float(epoch_tuned_metrics["f1"]),
            "best_tuned_precision": float(epoch_tuned_metrics["precision"]),
            "best_tuned_recall": float(epoch_tuned_metrics["recall"]),
            "best_tuned_threshold": float(epoch_best_threshold),
            "prediction_distribution_at_best_threshold": epoch_tuned_metrics["prediction_distribution"],
            "confusion_matrix_at_0_5": val_metrics_05["confusion_matrix"],
            "confusion_matrix_at_best_threshold": epoch_tuned_metrics["confusion_matrix"],
            "skipped_non_finite_loss_batches": int(non_finite_loss_batches),
            "skipped_non_finite_grad_steps": int(non_finite_grad_steps),
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| val_auc={val_metrics_05['roc_auc']:.6f} | f1@0.5={val_metrics_05['f1']:.6f} "
            f"| best_tuned_f1={epoch_tuned_metrics['f1']:.6f} "
            f"| best_threshold={epoch_best_threshold:.2f} "
            f"| pred_TRUE_at_best={epoch_tuned_metrics['prediction_distribution']['pred_TRUE']} "
            f"| pred_FALSE_at_best={epoch_tuned_metrics['prediction_distribution']['pred_FALSE']} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if epoch_tuned_metrics["f1"] > best_epoch_tuned_f1:
            best_epoch_tuned_f1 = float(epoch_tuned_metrics["f1"])
            best_epoch = int(epoch)
            best_epoch_threshold = float(epoch_best_threshold)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            best_epoch_threshold_table = epoch_threshold_table
            epochs_without_improvement = 0

            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            print(
                f"Saved new best model at epoch {best_epoch} "
                f"with best_tuned_f1={best_epoch_tuned_f1:.6f} "
                f"at threshold={best_epoch_threshold:.2f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                "No threshold-tuned validation F1 improvement. "
                f"Patience counter: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(
                f"Early stopping triggered after epoch {epoch}: "
                f"threshold-tuned validation F1 did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    if best_epoch_val_probs is None or best_epoch_val_labels is None or best_epoch_threshold_table is None:
        raise RuntimeError("Best validation predictions were not captured.")

    print("\n=== Saving best-epoch threshold table ===")
    best_threshold = best_epoch_threshold
    threshold_table = best_epoch_threshold_table
    print(f"Best epoch: {best_epoch}")
    print(f"Selected specialist threshold from best epoch: {best_threshold:.2f}")

    threshold_df = pd.DataFrame(threshold_table)
    threshold_df.to_csv(THRESHOLD_TABLE_PATH, index=False)
    print(f"Saved threshold table: {THRESHOLD_TABLE_PATH}")

    final_val_metrics = compute_metrics(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        threshold=best_threshold,
    )

    val_pred = (best_epoch_val_probs >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": decode_labels(best_epoch_val_labels),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": best_epoch_val_probs,
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    print("\n=== Validation error analysis ===")
    error_df = make_error_analysis(val_pred_df)
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    metrics_payload = {
        "model": MODEL_NAME,
        "model_name_or_path": str(MODEL_NAME_OR_PATH),
        "base_model_dir": str(BASE_MODEL_DIR),
        "tokenizer_source": tokenizer_source,
        "seed": SEED,
        "device": str(device),
        "purpose": "continued fine-tuning boundary specialist trained on hard rows from base RoBERTa validation predictions",
        "continues_from_base_checkpoint": True,
        "uses_test_data": False,
        "generates_submission": False,
        "config": {
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_train_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data": {
            "train_path": str(TRAIN_PATH),
            "total_rows": int(len(texts)),
            "label_distribution": label_distribution,
        },
        "split": {
            "train_rows": int(len(train_texts)),
            "val_rows": int(len(val_texts)),
            "test_size": 0.2,
            "stratified": True,
            "random_state": SEED,
        },
        "epoch_metrics": epoch_history,
        "best_epoch": {
            "epoch": int(best_epoch),
            "selection_objective": "best_tuned_validation_f1",
            "best_tuned_f1": float(best_epoch_tuned_f1),
            "selected_specialist_threshold": float(best_threshold),
            "checkpoint_dir": str(BEST_MODEL_DIR),
        },
        "early_stopping": {
            "enabled": True,
            "patience": EARLY_STOPPING_PATIENCE,
            "triggered": bool(early_stopped),
            "epochs_completed": int(epoch_history[-1]["epoch"]) if epoch_history else 0,
        },
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_validation_f1_per_epoch_then_best_epoch_by_tuned_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
            "threshold_table_path": str(THRESHOLD_TABLE_PATH),
        },
        "validation_final": final_val_metrics,
        "paths": {
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "threshold_table": str(THRESHOLD_TABLE_PATH),
            "notes": str(NOTES_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    write_notes(
        device=device,
        tokenizer_source=tokenizer_source,
        train_rows=len(train_texts),
        val_rows=len(val_texts),
        label_distribution=label_distribution,
        best_epoch=best_epoch,
        final_val_metrics=final_val_metrics,
        best_threshold=best_threshold,
        early_stopped=early_stopped,
    )
    print(f"Saved notes: {NOTES_PATH}")

    print("\nHard-specialist training script complete.")
    print("Next intended step: build a router inference script that uses base RoBERTa for confident rows")
    print("and this specialist only for uncertain rows.")


if __name__ == "__main__":
    main()
