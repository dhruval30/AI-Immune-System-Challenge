#!/usr/bin/env python3
"""
Inference-only script for the current RoBERTa hard-weighted checkpoint.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This script does not train. It loads outputs/roberta_hard_weighted/best_model/,
rebuilds the same validation split, tunes a threshold on validation probabilities,
and writes a test submission to outputs/roberta_hard_weighted_checkpoint_inference/.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MAX_LENGTH = 256
EVAL_BATCH_SIZE = 16
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SOURCE_OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_hard_weighted"
SOURCE_MODEL_DIR = SOURCE_OUTPUT_DIR / "best_model"
SOURCE_STATE_DICT_PATH = SOURCE_OUTPUT_DIR / "roberta_hard_weighted_best_state_dict.pt"

OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_hard_weighted_checkpoint_inference"
TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_hard_weighted_checkpoint_inference_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_hard_weighted_checkpoint_inference_val_predictions.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_hard_weighted_checkpoint_inference_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_hard_weighted_checkpoint_inference_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_hard_weighted_checkpoint_inference_notes.md"


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
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if line:
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
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(f"Train file must contain text and label columns; found {list(train_df.columns)}")
    if "text" not in test_df.columns:
        raise ValueError(f"Test file must contain text column; found {list(test_df.columns)}")
    if len(solution_df) != len(test_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")
    if not SOURCE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Best model directory does not exist: {SOURCE_MODEL_DIR}")


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
    return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0)


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


def tune_threshold_for_f1(y_true: np.ndarray, prob_true: np.ndarray) -> tuple[float, list[dict]]:
    records = []
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in tqdm(THRESHOLD_GRID, desc="Threshold search (validation)", unit="threshold"):
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            if has_labels and labels is not None:
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                total_loss += float(outputs.loss.item()) * batch_size
                total_examples += batch_size
    prob_true = np.concatenate(all_probs)
    if has_labels:
        y_true = np.concatenate(all_labels)
        avg_loss = total_loss / max(total_examples, 1)
        return avg_loss, prob_true, y_true
    return None, prob_true, None


def load_checkpoint_epoch() -> int | None:
    if not SOURCE_STATE_DICT_PATH.exists():
        return None
    checkpoint = torch.load(SOURCE_STATE_DICT_PATH, map_location="cpu")
    return int(checkpoint.get("epoch")) if "epoch" in checkpoint else None


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Hard-Weighted Checkpoint Inference ===")
    print("No training is performed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Source model dir: {SOURCE_MODEL_DIR}")

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_inputs(train_df, test_df, solution_df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    print("\n=== Preparing same validation split ===")
    texts = normalize_texts(train_df["text"], "Normalizing train text")
    test_texts = normalize_texts(test_df["text"], "Normalizing test text")
    y = encode_labels(train_df["label"], "Encoding labels")
    idx = np.arange(len(texts))
    _, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    val_texts = [texts[i] for i in val_idx]
    y_val = y[val_idx]
    print(f"Validation split size: {len(val_texts)}")

    print("\n=== Loading checkpoint ===")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL_DIR)
    model.to(device)
    checkpoint_epoch = load_checkpoint_epoch()
    print(f"Checkpoint epoch: {checkpoint_epoch}")

    print("\n=== Tokenizing ===")
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

    val_labels = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)
    pin_memory = device.type == "cuda"
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    print("\n=== Validation inference ===")
    val_loss, val_prob_true, val_true = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        desc="Validation inference",
        has_labels=True,
    )
    metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
    best_threshold, threshold_table = tune_threshold_for_f1(val_true, val_prob_true)
    final_val_metrics = compute_metrics(val_true, val_prob_true, threshold=best_threshold)
    final_val_metrics["val_loss"] = float(val_loss if val_loss is not None else math.nan)
    print(f"Validation F1 @ 0.5: {metrics_05['f1']:.6f}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation F1 @ tuned threshold: {final_val_metrics['f1']:.6f}")

    val_pred = (val_prob_true >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": decode_labels(val_true),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": val_prob_true,
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    print("\n=== Test inference ===")
    _, test_prob_true, _ = evaluate_model(
        model=model,
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
        "source_model_dir": str(SOURCE_MODEL_DIR),
        "source_state_dict": str(SOURCE_STATE_DICT_PATH),
        "checkpoint_epoch": checkpoint_epoch,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "validation_metrics_threshold_0_5": metrics_05,
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
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes = f"""# Hard-Weighted Checkpoint Inference

This is inference only. No training was performed.

## Source

- Checkpoint directory: `{SOURCE_MODEL_DIR.relative_to(ROOT_DIR)}`
- State dict: `{SOURCE_STATE_DICT_PATH.relative_to(ROOT_DIR)}`
- Checkpoint epoch: `{checkpoint_epoch}`

## Validation

- F1 @ 0.5: `{metrics_05['f1']:.6f}`
- Tuned threshold: `{best_threshold:.2f}`
- F1 @ tuned threshold: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_val_metrics['confusion_matrix']['matrix']}`

## Test

- Prediction distribution: `{test_distribution}`

## Outputs

- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_submission.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_test_probabilities.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_val_predictions.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_metrics.json`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
