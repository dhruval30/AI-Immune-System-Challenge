#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate
# Safety: This script is generated but not executed by Codex. User should run it manually.

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
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "roberta-base"
N_SPLITS = 3
MAX_LENGTH = 256
EPOCHS = 2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_base_cv"
CHECKPOINTS_DIR = OUTPUT_DIR / "fold_checkpoints"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_base_cv_metrics.json"
OOF_PRED_PATH = OUTPUT_DIR / "roberta_base_cv_oof_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_base_cv_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_base_cv_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_base_cv_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_base_cv_notes.md"



def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def detect_device() -> torch.device:
    # Required order: MPS -> CUDA -> CPU
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

    for threshold in tqdm(thresholds, desc="Threshold search (OOF)", unit="threshold"):
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



def get_fold_dirs(fold_idx: int) -> tuple[Path, Path, Path]:
    fold_dir = CHECKPOINTS_DIR / f"fold_{fold_idx}"
    best_model_dir = fold_dir / "best_model"
    state_dict_path = fold_dir / "roberta_base_cv_best_state_dict.pt"
    return fold_dir, best_model_dir, state_dict_path



def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    device = detect_device()
    print(f"Using device: {device}")

    train_batch_size = TRAIN_BATCH_SIZE
    grad_accum_steps = GRADIENT_ACCUMULATION_STEPS
    learning_rate = LEARNING_RATE
    adam_eps = ADAM_EPS

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    validate_inputs(train_df, test_df, solution_df)
    print("Input schema checks passed.")
    print(f"Expected submission columns: {solution_df.columns.tolist()}")

    print("\n=== Preparing text and labels ===")
    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("\n=== Tokenizing full train/test once ===")
    full_train_input_ids, full_train_attention_mask = tokenize_texts(
        texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing full train",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test",
    )

    full_train_labels = torch.tensor(y, dtype=torch.long)

    test_dataset = TensorDataset(test_input_ids, test_attention_mask)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\n=== 3-Fold Stratified CV Training ===")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_prob_true = np.zeros(len(texts), dtype=np.float64)
    oof_seen_mask = np.zeros(len(texts), dtype=bool)
    fold_test_prob_true = np.zeros((N_SPLITS, len(test_texts)), dtype=np.float64)
    fold_results: List[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(texts)), y), start=1):
        print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")

        fold_dir, best_model_dir, state_dict_path = get_fold_dirs(fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir.mkdir(parents=True, exist_ok=True)

        x_train_ids = full_train_input_ids[tr_idx]
        x_train_mask = full_train_attention_mask[tr_idx]
        y_train = full_train_labels[tr_idx]

        x_val_ids = full_train_input_ids[val_idx]
        x_val_mask = full_train_attention_mask[val_idx]
        y_val = full_train_labels[val_idx]

        train_dataset = TensorDataset(x_train_ids, x_train_mask, y_train)
        val_dataset = TensorDataset(x_val_ids, x_val_mask, y_val)

        fold_generator = torch.Generator().manual_seed(SEED + fold_idx)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=fold_generator,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)

        optimizer = build_optimizer(model=model, learning_rate=learning_rate, adam_eps=adam_eps)
        num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_training_steps = num_update_steps_per_epoch * EPOCHS
        warmup_steps = int(0.1 * total_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        best_epoch = -1
        best_epoch_f1 = -1.0
        best_epoch_train_loss = float("nan")
        best_epoch_val_loss = float("nan")
        best_epoch_val_probs: np.ndarray | None = None
        best_epoch_val_labels: np.ndarray | None = None
        fold_epoch_history: List[dict] = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            seen_examples = 0
            non_finite_loss_batches = 0
            non_finite_grad_steps = 0

            progress = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{EPOCHS} - training", unit="batch")
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

                (loss / grad_accum_steps).backward()

                batch_size = input_ids.size(0)
                running_loss += loss_value * batch_size
                seen_examples += batch_size

                if (step % grad_accum_steps == 0) or (step == len(train_loader)):
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
                desc=f"Fold {fold_idx} Epoch {epoch}/{EPOCHS} - validation",
                has_labels=True,
            )

            val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
            val_metrics_05.update(
                {
                    "fold": int(fold_idx),
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss if val_loss is not None else np.nan),
                    "skipped_non_finite_loss_batches": int(non_finite_loss_batches),
                    "skipped_non_finite_grad_steps": int(non_finite_grad_steps),
                }
            )
            fold_epoch_history.append(val_metrics_05)

            print(
                f"Fold {fold_idx} Epoch {epoch} | train_loss={train_loss:.6f} | "
                f"val_loss={val_metrics_05['val_loss']:.6f} | val_f1@0.5={val_metrics_05['f1']:.6f} | "
                f"val_auc={val_metrics_05['roc_auc']:.6f} | "
                f"skipped_non_finite_loss_batches={non_finite_loss_batches} | "
                f"skipped_non_finite_grad_steps={non_finite_grad_steps}"
            )

            if val_metrics_05["f1"] > best_epoch_f1:
                best_epoch_f1 = float(val_metrics_05["f1"])
                best_epoch = int(epoch)
                best_epoch_train_loss = float(train_loss)
                best_epoch_val_loss = float(val_metrics_05["val_loss"])
                best_epoch_val_probs = val_prob_true.copy()
                best_epoch_val_labels = val_true.copy()

                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": best_epoch,
                        "best_val_f1": best_epoch_f1,
                        "model_state_dict": model.state_dict(),
                    },
                    state_dict_path,
                )
                print(
                    f"Saved new fold-best model at fold={fold_idx}, epoch={best_epoch}, "
                    f"val_f1@0.5={best_epoch_f1:.6f}"
                )

        if best_epoch_val_probs is None or best_epoch_val_labels is None:
            raise RuntimeError(f"Fold {fold_idx}: best validation predictions were not captured.")

        oof_prob_true[val_idx] = best_epoch_val_probs
        oof_seen_mask[val_idx] = True

        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        best_model.to(device)
        _, fold_test_probs, _ = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            desc=f"Fold {fold_idx} - test inference",
            has_labels=False,
        )
        fold_test_prob_true[fold_idx - 1] = fold_test_probs

        fold_best_metrics = compute_metrics(best_epoch_val_labels, best_epoch_val_probs, threshold=0.5)
        fold_best_metrics.update(
            {
                "fold": int(fold_idx),
                "best_epoch": int(best_epoch),
                "best_train_loss": float(best_epoch_train_loss),
                "best_val_loss": float(best_epoch_val_loss),
                "num_train_rows": int(len(tr_idx)),
                "num_val_rows": int(len(val_idx)),
                "best_model_dir": str(best_model_dir),
                "best_state_dict": str(state_dict_path),
                "epoch_history": fold_epoch_history,
            }
        )
        fold_results.append(fold_best_metrics)

    if not np.all(oof_seen_mask):
        missing = int((~oof_seen_mask).sum())
        raise RuntimeError(f"OOF predictions missing for {missing} rows.")

    print("\n=== Threshold tuning on OOF probabilities ===")
    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=y,
        prob_true=oof_prob_true,
        thresholds=THRESHOLD_GRID,
    )
    print(f"Best threshold by OOF F1: {best_threshold:.2f}")

    oof_metrics = compute_metrics(y_true=y, prob_true=oof_prob_true, threshold=best_threshold)

    oof_pred = (oof_prob_true >= best_threshold).astype(int)
    oof_pred_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": decode_labels(y),
            "oof_pred_label": decode_labels(oof_pred),
            "oof_prob_TRUE": oof_prob_true,
        }
    )
    oof_pred_df.to_csv(OOF_PRED_PATH, index=False)
    print(f"Saved OOF predictions: {OOF_PRED_PATH}")

    print("\n=== OOF error analysis ===")
    errors = oof_pred_df[oof_pred_df["true_label"] != oof_pred_df["oof_pred_label"]].copy()
    errors["error_type"] = np.where(
        (errors["true_label"] == "FALSE") & (errors["oof_pred_label"] == "TRUE"),
        "false_positive",
        "false_negative",
    )
    errors["confidence"] = np.where(
        errors["error_type"] == "false_positive",
        errors["oof_prob_TRUE"],
        1.0 - errors["oof_prob_TRUE"],
    )

    top_k = 200
    fp_df = errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(top_k)
    fn_df = errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(top_k)
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)
    error_df = error_df[["text", "true_label", "oof_pred_label", "oof_prob_TRUE", "error_type", "confidence"]]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    avg_test_prob_true = fold_test_prob_true.mean(axis=0)
    test_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "pred_prob_TRUE": avg_test_prob_true,
        }
    )
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {TEST_PROB_PATH}")

    test_pred = (avg_test_prob_true >= best_threshold).astype(int)
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
            "n_splits": N_SPLITS,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "effective_train_batch_size": train_batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_gradient_accumulation_steps": grad_accum_steps,
            "learning_rate": LEARNING_RATE,
            "effective_learning_rate": learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "effective_adam_eps": adam_eps,
            "optimizer": "AdamW",
        },
        "folds": fold_results,
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_oof_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "oof_final": oof_metrics,
        "test_prediction_distribution": test_distribution,
        "paths": {
            "metrics": str(METRICS_PATH),
            "oof_predictions": str(OOF_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
            "fold_checkpoints_dir": str(CHECKPOINTS_DIR),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes_text = f"""# RoBERTa-base 3-Fold CV Fine-Tuning Notes

## Run Configuration

- Model: `{MODEL_NAME}`
- Seed: `{SEED}`
- Device: `{device}`
- Folds: `{N_SPLITS}` (StratifiedKFold)
- Max length: `{MAX_LENGTH}`
- Epochs per fold: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Eval batch size: `{EVAL_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Weight decay: `{WEIGHT_DECAY}`

## Training Strategy

- For each fold, best epoch is selected by validation F1 at threshold 0.5.
- OOF probabilities are assembled from fold-best models.
- Final threshold is tuned on full OOF probabilities (0.30 to 0.70) for best OOF F1.
- Test probabilities are averaged across fold-best models.

## Final OOF Metrics

- Threshold: `{oof_metrics['threshold']:.2f}`
- Accuracy: `{oof_metrics['accuracy']:.6f}`
- Precision: `{oof_metrics['precision']:.6f}`
- Recall: `{oof_metrics['recall']:.6f}`
- F1: `{oof_metrics['f1']:.6f}`
- ROC AUC: `{oof_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{oof_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{oof_metrics['prediction_distribution']}`

## Outputs

- `outputs/roberta_base_cv/roberta_base_cv_metrics.json`
- `outputs/roberta_base_cv/roberta_base_cv_oof_predictions.csv`
- `outputs/roberta_base_cv/roberta_base_cv_error_analysis.csv`
- `outputs/roberta_base_cv/roberta_base_cv_test_probabilities.csv`
- `outputs/roberta_base_cv/roberta_base_cv_submission.csv`
- `outputs/roberta_base_cv/roberta_base_cv_notes.md`
- `outputs/roberta_base_cv/fold_checkpoints/`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
