# pip install pandas numpy scikit-learn tqdm torch transformers accelerate
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

import json
import math
import random
import shutil
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
EPOCHS = 2
TARGET_TRAIN_BATCH_SIZE = 16
FALLBACK_TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8

THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)
PSEUDO_TRUE_THRESHOLD = 0.95
PSEUDO_FALSE_THRESHOLD = 0.05

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_pseudolabel"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_DICT_PATH = OUTPUT_DIR / "roberta_pseudolabel_best_state_dict.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
SOURCE_TEST_PROB_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_test_probabilities.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_pseudolabel_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_pseudolabel_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_pseudolabel_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_pseudolabel_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_pseudolabel_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_pseudolabel_notes.md"
PSEUDO_SUMMARY_PATH = OUTPUT_DIR / "roberta_pseudolabel_pseudo_summary.json"
PSEUDO_SAMPLES_PATH = OUTPUT_DIR / "roberta_pseudolabel_pseudo_samples.csv"



def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def detect_device() -> torch.device:
    # Requirement order: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)



def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)



def normalize_text(value: object) -> str:
    if pd.isna(value):
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
    if pd.isna(value):
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



def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    required_train_cols = {"label", "text"}
    required_test_cols = {"text"}

    missing_train_cols = sorted(required_train_cols - set(train_df.columns))
    if missing_train_cols:
        raise ValueError(f"Train file is missing required columns: {missing_train_cols}")

    missing_test_cols = sorted(required_test_cols - set(test_df.columns))
    if missing_test_cols:
        raise ValueError(f"Test file is missing required columns: {missing_test_cols}")

    if len(solution_df) != len(test_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )

    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")



def load_test_probabilities(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Pseudo-label source probabilities not found: {path}. "
            "Run the strong RoBERTa baseline first."
        )

    df = pd.read_csv(path)
    if "pred_prob_TRUE" not in df.columns:
        raise ValueError(f"{path} must contain 'pred_prob_TRUE' column.")

    probs = pd.to_numeric(df["pred_prob_TRUE"], errors="coerce")
    if probs.isna().any():
        raise ValueError(f"{path} contains non-numeric probability values.")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"{path} contains probabilities outside [0, 1].")

    df = df.copy()
    df["pred_prob_TRUE"] = probs
    return df


def is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    oom_tokens = [
        "out of memory",
        "oom",
        "not enough memory",
        "mps backend out of memory",
        "cuda out of memory",
    ]
    return any(token in msg for token in oom_tokens)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def reset_best_artifacts() -> None:
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_STATE_DICT_PATH.exists():
        BEST_STATE_DICT_PATH.unlink()


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



def compute_metrics(y_true: np.ndarray, prob_true: np.ndarray, threshold: float) -> dict:
    pred = (prob_true >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_true)),
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
    has_labels: bool,
) -> tuple[float | None, np.ndarray, np.ndarray | None]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            if has_labels:
                input_ids, attention_mask, labels = batch
                labels = labels.to(device)
            else:
                input_ids, attention_mask = batch
                labels = None

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())

            if has_labels and labels is not None:
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                total_loss += outputs.loss.item() * batch_size
                total_examples += batch_size

    prob_true = np.concatenate(all_probs)

    if has_labels:
        y_true = np.concatenate(all_labels)
        avg_loss = total_loss / max(total_examples, 1)
        return avg_loss, prob_true, y_true

    return None, prob_true, None



def build_optimizer(model: AutoModelForSequenceClassification, learning_rate: float, adam_eps: float) -> AdamW:
    # Standard transformer setup: no weight decay for bias/layernorm parameters.
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
        lr=learning_rate,
        eps=adam_eps,
    )



def train_with_batch_size(
    tokenizer: AutoTokenizer,
    train_dataset: TensorDataset,
    val_loader: DataLoader,
    device: torch.device,
    train_batch_size: int,
) -> dict:
    print("\n=== Optimizer/Scheduler setup ===")
    reset_best_artifacts()
    set_seeds(SEED)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    pin_memory = device.type == "cuda"
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )

    optimizer = build_optimizer(model=model, learning_rate=LEARNING_RATE, adam_eps=ADAM_EPS)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print("\n=== Training ===")
    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    best_epoch_train_loss = float("nan")
    best_epoch_val_loss = float("nan")
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            has_labels=True,
        )

        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_loss"] = float(train_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        val_metrics_05["skipped_non_finite_loss_batches"] = int(non_finite_loss_batches)
        val_metrics_05["skipped_non_finite_grad_steps"] = int(non_finite_grad_steps)
        epoch_history.append(val_metrics_05)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_train_loss = float(train_loss)
            best_epoch_val_loss = float(val_metrics_05["val_loss"])
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()

            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_f1": best_epoch_f1,
                    "model_state_dict": model.state_dict(),
                },
                BEST_STATE_DICT_PATH,
            )
            print(f"Saved new best model at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError("Best validation predictions were not captured.")

    return {
        "best_epoch": int(best_epoch),
        "best_epoch_f1": float(best_epoch_f1),
        "best_epoch_train_loss": float(best_epoch_train_loss),
        "best_epoch_val_loss": float(best_epoch_val_loss),
        "best_epoch_val_probs": best_epoch_val_probs,
        "best_epoch_val_labels": best_epoch_val_labels,
        "epoch_history": epoch_history,
        "warmup_steps": int(warmup_steps),
        "total_training_steps": int(total_training_steps),
    }


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    device = detect_device()
    print("\n=== Startup Configuration ===")
    print(f"device: {device}")
    print(f"model: {MODEL_NAME}")
    print("precision: fp32 only (no AMP/fp16)")

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    source_prob_df = load_test_probabilities(SOURCE_TEST_PROB_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")
    print(f"Source probability shape: {source_prob_df.shape}")

    validate_inputs(train_df, test_df, solution_df)
    if len(source_prob_df) != len(test_df):
        raise ValueError(
            f"Source probability rows ({len(source_prob_df)}) do not match test rows ({len(test_df)})."
        )
    print("Input schema checks passed.")
    print(f"Expected submission columns: {solution_df.columns.tolist()}")

    print("\n=== Preparing text and labels ===")
    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    idx = np.arange(len(texts))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_texts_original = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train_original = y[train_idx]
    y_val = y[val_idx]

    print(f"Original train split size: {len(train_texts_original)}")
    print(f"Validation split size: {len(val_texts)}")

    print("\n=== Building pseudo-labels from source probabilities ===")
    source_probs = source_prob_df["pred_prob_TRUE"].to_numpy(dtype=np.float64)
    pseudo_true_mask = source_probs >= PSEUDO_TRUE_THRESHOLD
    pseudo_false_mask = source_probs <= PSEUDO_FALSE_THRESHOLD
    pseudo_mask = pseudo_true_mask | pseudo_false_mask

    pseudo_indices = np.where(pseudo_mask)[0]
    pseudo_texts = [test_texts[i] for i in pseudo_indices]
    pseudo_labels = np.where(pseudo_true_mask[pseudo_indices], 1, 0).astype(np.int64)

    pseudo_true_count = int((pseudo_labels == 1).sum())
    pseudo_false_count = int((pseudo_labels == 0).sum())
    pseudo_total = int(len(pseudo_labels))

    print(f"Pseudo TRUE count (>= {PSEUDO_TRUE_THRESHOLD}): {pseudo_true_count}")
    print(f"Pseudo FALSE count (<= {PSEUDO_FALSE_THRESHOLD}): {pseudo_false_count}")
    print(f"Pseudo samples added to training set: {pseudo_total}")

    pseudo_summary = {
        "source_probability_path": str(SOURCE_TEST_PROB_PATH),
        "pseudo_true_threshold": float(PSEUDO_TRUE_THRESHOLD),
        "pseudo_false_threshold": float(PSEUDO_FALSE_THRESHOLD),
        "pseudo_true_count": pseudo_true_count,
        "pseudo_false_count": pseudo_false_count,
        "pseudo_total": pseudo_total,
        "ignored_test_rows": int(len(test_texts) - pseudo_total),
    }
    PSEUDO_SUMMARY_PATH.write_text(json.dumps(pseudo_summary, indent=2), encoding="utf-8")
    print(f"Saved pseudo summary: {PSEUDO_SUMMARY_PATH}")

    pseudo_samples_df = pd.DataFrame(
        {
            "test_row_index": pseudo_indices,
            "text": pseudo_texts,
            "pred_prob_TRUE": source_probs[pseudo_indices],
            "pseudo_label": decode_labels(pseudo_labels),
        }
    )
    pseudo_samples_df.to_csv(PSEUDO_SAMPLES_PATH, index=False)
    print(f"Saved pseudo samples: {PSEUDO_SAMPLES_PATH}")

    train_texts_augmented = train_texts_original + pseudo_texts
    y_train_augmented = np.concatenate([y_train_original, pseudo_labels], axis=0).astype(np.int64)

    print(f"Augmented train size: {len(train_texts_augmented)}")
    print(f"Validation size (original only): {len(val_texts)}")

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("\n=== Tokenizing ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts_augmented,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing augmented train split",
    )
    val_input_ids, val_attention_mask = tokenize_texts(
        val_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing validation split",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test split",
    )

    train_labels = torch.tensor(y_train_augmented, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Training with memory-aware batch size ===")
    batch_candidates = [TARGET_TRAIN_BATCH_SIZE]
    if FALLBACK_TRAIN_BATCH_SIZE != TARGET_TRAIN_BATCH_SIZE:
        batch_candidates.append(FALLBACK_TRAIN_BATCH_SIZE)

    training_result = None
    effective_train_batch_size = TARGET_TRAIN_BATCH_SIZE

    for candidate_batch_size in batch_candidates:
        try:
            print(f"Trying train batch size: {candidate_batch_size}")
            training_result = train_with_batch_size(
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_loader=val_loader,
                device=device,
                train_batch_size=candidate_batch_size,
            )
            effective_train_batch_size = candidate_batch_size
            break
        except RuntimeError as exc:
            if candidate_batch_size == TARGET_TRAIN_BATCH_SIZE and is_oom_error(exc):
                print(
                    f"OOM detected with train batch size {TARGET_TRAIN_BATCH_SIZE}. "
                    f"Retrying with fallback batch size {FALLBACK_TRAIN_BATCH_SIZE}."
                )
                clear_device_cache(device)
                continue
            raise

    if training_result is None:
        raise RuntimeError("Training failed for all configured batch sizes.")

    print(f"Effective train batch size: {effective_train_batch_size}")
    print(f"Effective gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective optimization batch size: {effective_train_batch_size * GRADIENT_ACCUMULATION_STEPS}")

    best_epoch = int(training_result["best_epoch"])
    best_epoch_f1 = float(training_result["best_epoch_f1"])
    best_epoch_val_probs = training_result["best_epoch_val_probs"]
    best_epoch_val_labels = training_result["best_epoch_val_labels"]
    epoch_history = training_result["epoch_history"]
    warmup_steps = int(training_result["warmup_steps"])
    total_training_steps = int(training_result["total_training_steps"])

    print("\n=== Threshold tuning on validation probabilities ===")
    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        thresholds=THRESHOLD_GRID,
    )
    print(f"Best threshold by validation F1: {best_threshold:.2f}")

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
    errors = val_pred_df[val_pred_df["true_label"] != val_pred_df["pred_label"]].copy()
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

    top_k = 200
    fp_df = errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(top_k)
    fn_df = errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(top_k)
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)
    error_df = error_df[["text", "true_label", "pred_label", "pred_prob_TRUE", "error_type", "confidence"]]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Loading best checkpoint for test inference ===")
    best_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    best_model.to(device)

    _, test_prob_true, _ = evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
        desc="Test inference",
        has_labels=False,
    )

    test_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "pred_prob_TRUE": test_prob_true,
        }
    )
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {TEST_PROB_PATH}")

    test_pred = (test_prob_true >= best_threshold).astype(int)
    test_pred_labels = decode_labels(test_pred)
    test_distribution = {
        "pred_FALSE": int((test_pred == 0).sum()),
        "pred_TRUE": int((test_pred == 1).sum()),
    }
    print(f"Test prediction distribution: {test_distribution}")

    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": test_pred_labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = test_pred_labels
    submission_df = submission_df[solution_df.columns.tolist()]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")

    metrics_payload = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
        "config": {
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "target_train_batch_size": TARGET_TRAIN_BATCH_SIZE,
            "fallback_train_batch_size": FALLBACK_TRAIN_BATCH_SIZE,
            "effective_train_batch_size": effective_train_batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
            "total_training_steps": total_training_steps,
            "fp16_amp_used": False,
        },
        "pseudo_labeling": pseudo_summary,
        "split": {
            "original_train_rows": int(len(train_texts_original)),
            "pseudo_rows_added_to_train": pseudo_total,
            "augmented_train_rows": int(len(train_texts_augmented)),
            "val_rows_original_only": int(len(val_texts)),
            "test_rows": int(len(test_texts)),
            "stratified": True,
            "random_state": SEED,
        },
        "epoch_metrics_threshold_0_5": epoch_history,
        "best_epoch": {
            "epoch": best_epoch,
            "val_f1_threshold_0_5": best_epoch_f1,
            "checkpoint_dir": str(BEST_MODEL_DIR),
            "state_dict_path": str(BEST_STATE_DICT_PATH),
            "train_loss": float(training_result["best_epoch_train_loss"]),
            "val_loss": float(training_result["best_epoch_val_loss"]),
        },
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_validation_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "validation_final": final_val_metrics,
        "test_prediction_distribution": test_distribution,
        "paths": {
            "source_test_probabilities": str(SOURCE_TEST_PROB_PATH),
            "metrics": str(METRICS_PATH),
            "pseudo_summary": str(PSEUDO_SUMMARY_PATH),
            "pseudo_samples": str(PSEUDO_SAMPLES_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_DICT_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes_text = f"""# RoBERTa Pseudo-Label Fine-Tuning Notes

## Run Configuration

- Model: `{MODEL_NAME}`
- Seed: `{SEED}`
- Device: `{device}`
- Max length: `{MAX_LENGTH}`
- Epochs: `{EPOCHS}`
- Target train batch size: `{TARGET_TRAIN_BATCH_SIZE}`
- Fallback train batch size: `{FALLBACK_TRAIN_BATCH_SIZE}`
- Effective train batch size: `{effective_train_batch_size}`
- Eval batch size: `{EVAL_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Weight decay: `{WEIGHT_DECAY}`
- Adam epsilon: `{ADAM_EPS}`

## Pseudo-Labeling

- Source probs: `outputs/roberta_base/roberta_base_test_probabilities.csv`
- TRUE rule: `pred_prob_TRUE >= {PSEUDO_TRUE_THRESHOLD}`
- FALSE rule: `pred_prob_TRUE <= {PSEUDO_FALSE_THRESHOLD}`
- Pseudo TRUE count: `{pseudo_true_count}`
- Pseudo FALSE count: `{pseudo_false_count}`
- Pseudo total added: `{pseudo_total}`
- Validation remains original-labeled data only.

## Validation Metrics

- Threshold: `{final_val_metrics['threshold']:.2f}`
- Accuracy: `{final_val_metrics['accuracy']:.6f}`
- Precision: `{final_val_metrics['precision']:.6f}`
- Recall: `{final_val_metrics['recall']:.6f}`
- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_val_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{final_val_metrics['prediction_distribution']}`

## Outputs

- `outputs/roberta_pseudolabel/roberta_pseudolabel_metrics.json`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_pseudo_summary.json`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_pseudo_samples.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_val_predictions.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_error_analysis.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_test_probabilities.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_submission.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_notes.md`
- `outputs/roberta_pseudolabel/best_model/`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_best_state_dict.pt`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
