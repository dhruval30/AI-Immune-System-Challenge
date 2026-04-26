#!/usr/bin/env python3
"""
Train RoBERTa with jointly learned numeric label-style features.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This script fine-tunes roberta-base while feeding handcrafted EDA/style
features through a small MLP branch. The feature branch is trained jointly
with the transformer classifier head, unlike post-hoc calibration.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
ROBERTA_LR = 1e-5
HEAD_LR = 3e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
DROPOUT = 0.15
FEATURE_HIDDEN_SIZE = 32
CLASSIFIER_HIDDEN_SIZE = 256

FEATURE_COLUMNS = [
    "char_length",
    "word_count",
    "sentence_count",
    "max_word_length",
    "avg_word_length",
    "stopword_ratio",
    "unique_word_ratio",
    "repeated_token_count",
    "newline_count",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_style_features"
EDA_FEATURE_DIR = ROOT_DIR / "outputs" / "eda_label_style"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_PATH = DATA_DIR / "solution_format.csv"
TRAIN_FEATURE_PATH = EDA_FEATURE_DIR / "train_label_style_features.csv"
TEST_FEATURE_PATH = EDA_FEATURE_DIR / "test_label_style_features.csv"

BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_PATH = OUTPUT_DIR / "roberta_style_features_best_state_dict.pt"
FEATURE_STATS_PATH = OUTPUT_DIR / "roberta_style_features_feature_stats.json"
METRICS_PATH = OUTPUT_DIR / "roberta_style_features_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_style_features_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_style_features_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_style_features_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_style_features_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_style_features_notes.md"

WORD_RE = re.compile(r"[A-Za-z0-9_']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=desc, unit="lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().upper()
    if text == "TRUE":
        return 1
    if text == "FALSE":
        return 0
    raise ValueError(f"Unexpected label value: {value!r}")


def label_text(value: int | bool | np.integer) -> str:
    return "TRUE" if int(value) == 1 else "FALSE"


def find_text_column(df: pd.DataFrame) -> str:
    preferred = ["text", "conversation", "prompt", "response", "content", "message"]
    for column in preferred:
        if column in df.columns:
            return column
    non_label_columns = [column for column in df.columns if column.lower() != "label"]
    if len(non_label_columns) == 1:
        return non_label_columns[0]
    raise ValueError(
        "Could not identify text column. Expected one of "
        f"{preferred} or exactly one non-label column; found {list(df.columns)}"
    )


def validate_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    solution_df: pd.DataFrame,
    train_text_col: str,
    test_text_col: str,
) -> list[str]:
    if "label" not in train_df.columns:
        raise ValueError("Train data must contain a label column.")
    if train_text_col not in train_df.columns:
        raise ValueError(f"Train text column {train_text_col!r} not found.")
    if test_text_col not in test_df.columns:
        raise ValueError(f"Test text column {test_text_col!r} not found.")
    if len(test_df) != len(solution_df):
        raise ValueError(
            f"solution_format row count ({len(solution_df)}) does not match test row count ({len(test_df)})."
        )
    if not len(solution_df.columns):
        raise ValueError("solution_format.csv must contain at least one column.")
    if "label" not in solution_df.columns:
        raise ValueError(
            f"solution_format.csv must contain a label column; found {list(solution_df.columns)}."
        )
    return list(solution_df.columns)


def extract_style_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    word_lengths = [len(word) for word in words]
    word_count = len(words)
    unique_word_count = len(set(words))
    token_counts = Counter(words)
    repeated_token_count = sum(count - 1 for count in token_counts.values() if count > 1)

    sentence_parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    sentence_count = len(sentence_parts)
    stopword_count = sum(1 for word in words if word in STOPWORDS)

    return {
        "char_length": float(len(text)),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "max_word_length": float(max(word_lengths) if word_lengths else 0),
        "avg_word_length": float(np.mean(word_lengths) if word_lengths else 0.0),
        "stopword_ratio": float(stopword_count / word_count if word_count else 0.0),
        "unique_word_ratio": float(unique_word_count / word_count if word_count else 0.0),
        "repeated_token_count": float(repeated_token_count),
        "newline_count": float(text.count("\n")),
    }


def compute_style_feature_frame(texts: list[str], desc: str) -> pd.DataFrame:
    rows = [extract_style_features(text) for text in tqdm(texts, desc=desc, unit="rows")]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def load_or_compute_features(
    texts: list[str],
    feature_path: Path,
    expected_rows: int,
    split_name: str,
) -> pd.DataFrame:
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        missing_columns = [column for column in FEATURE_COLUMNS if column not in feature_df.columns]
        if not missing_columns and len(feature_df) == expected_rows:
            print(f"Loaded {split_name} style features from {feature_path}")
            return feature_df[FEATURE_COLUMNS].copy()
        print(
            f"{split_name} feature file exists but is incompatible. "
            f"Missing columns: {missing_columns}; rows: {len(feature_df)} expected: {expected_rows}. "
            "Computing fallback features inside script."
        )
    else:
        print(f"{split_name} feature file not found at {feature_path}. Computing fallback features.")

    return compute_style_feature_frame(texts, f"Computing {split_name} style features")


def fit_feature_normalizer(feature_matrix: np.ndarray) -> dict[str, list[float]]:
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_feature_normalizer(
    feature_matrix: np.ndarray,
    stats: dict[str, list[float]],
) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((feature_matrix.astype(np.float32) - mean) / std).astype(np.float32)


def tokenize_texts(tokenizer: AutoTokenizer, texts: list[str], desc: str) -> dict[str, torch.Tensor]:
    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    batch_size = 256
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
    }


class TextStyleDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        style_features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.style_features = torch.tensor(style_features, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "style_features": self.style_features[index],
        }
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class RobertaWithStyleFeatures(nn.Module):
    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        feature_hidden_size: int = FEATURE_HIDDEN_SIZE,
        classifier_hidden_size: int = CLASSIFIER_HIDDEN_SIZE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        roberta_hidden_size = int(self.roberta.config.hidden_size)
        self.feature_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(roberta_hidden_size + feature_hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, 2),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        feature_repr = self.feature_branch(style_features)
        fused_repr = torch.cat([cls_repr, feature_repr], dim=1)
        logits = self.classifier(fused_repr)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result


def create_optimizer(model: RobertaWithStyleFeatures) -> AdamW:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    roberta_decay_params = []
    roberta_no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay)
        is_roberta = name.startswith("roberta.")
        if is_roberta and is_no_decay:
            roberta_no_decay_params.append(parameter)
        elif is_roberta:
            roberta_decay_params.append(parameter)
        elif is_no_decay:
            head_no_decay_params.append(parameter)
        else:
            head_decay_params.append(parameter)

    return AdamW(
        [
            {
                "params": roberta_decay_params,
                "lr": ROBERTA_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": roberta_no_decay_params,
                "lr": ROBERTA_LR,
                "weight_decay": 0.0,
            },
            {
                "params": head_decay_params,
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_no_decay_params,
                "lr": HEAD_LR,
                "weight_decay": 0.0,
            },
        ],
        eps=ADAM_EPS,
    )


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def has_non_finite_gradients(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return True
    return False


def prediction_distribution(preds: np.ndarray) -> dict[str, int]:
    return {
        "pred_FALSE": int((preds == 0).sum()),
        "pred_TRUE": int((preds == 1).sum()),
    }


def compute_metrics(
    y_true: np.ndarray,
    probs_true: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    preds = (probs_true >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "prediction_distribution": prediction_distribution(preds),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs_true))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def evaluate(
    model: RobertaWithStyleFeatures,
    data_loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, unit="batch"):
            batch = batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.get("loss")
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
            batch_probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
            probs.append(batch_probs.detach().cpu().numpy())
            if "labels" in batch:
                labels.append(batch["labels"].detach().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else math.nan
    all_probs = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    all_labels = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    return mean_loss, all_probs, all_labels


def predict_test(
    model: RobertaWithStyleFeatures,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Test inference", unit="batch"):
            batch = batch_to_device(batch, device)
            outputs = model(**batch)
            batch_probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
            probs.append(batch_probs.detach().cpu().numpy())
    return np.concatenate(probs)


def tune_threshold(y_true: np.ndarray, probs_true: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for threshold in tqdm(np.round(np.arange(0.30, 0.701, 0.01), 2), desc="Threshold search", unit="threshold"):
        preds = (probs_true >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "pred_TRUE": int((preds == 1).sum()),
                "pred_FALSE": int((preds == 0).sum()),
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return {"best": best, "all": rows}


def save_checkpoint(
    model: RobertaWithStyleFeatures,
    tokenizer: AutoTokenizer,
    feature_stats: dict[str, Any],
    best_epoch: int,
    best_threshold: float,
    best_val_f1: float,
) -> None:
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    model.roberta.save_pretrained(BEST_MODEL_DIR / "roberta_encoder")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "feature_columns": FEATURE_COLUMNS,
            "feature_stats": feature_stats,
            "max_length": MAX_LENGTH,
            "best_epoch": best_epoch,
            "best_threshold": float(best_threshold),
            "best_val_f1": float(best_val_f1),
            "architecture": {
                "feature_hidden_size": FEATURE_HIDDEN_SIZE,
                "classifier_hidden_size": CLASSIFIER_HIDDEN_SIZE,
                "dropout": DROPOUT,
            },
        },
        BEST_STATE_PATH,
    )


def load_best_model(device: torch.device) -> RobertaWithStyleFeatures:
    model = RobertaWithStyleFeatures(
        model_name=MODEL_NAME,
        feature_dim=len(FEATURE_COLUMNS),
        feature_hidden_size=FEATURE_HIDDEN_SIZE,
        classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE,
        dropout=DROPOUT,
    )
    checkpoint = torch.load(BEST_STATE_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def create_submission(
    solution_df: pd.DataFrame,
    pred_labels: np.ndarray,
    submission_columns: list[str],
) -> pd.DataFrame:
    submission = solution_df.copy()
    label_values = [label_text(value) for value in pred_labels]
    if submission_columns == ["label"]:
        return pd.DataFrame({"label": label_values})
    submission["label"] = label_values
    return submission[submission_columns]


def write_error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    probs_true: np.ndarray,
    raw_features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    preds = (probs_true >= threshold).astype(int)
    error_rows = []
    for index, (text, true_label, pred_label, prob_true) in enumerate(
        zip(texts, y_true, preds, probs_true, strict=True)
    ):
        if true_label == pred_label:
            continue
        error_type = "false_positive" if pred_label == 1 else "false_negative"
        confidence = prob_true if pred_label == 1 else 1.0 - prob_true
        row = {
            "text": text,
            "true_label": label_text(true_label),
            "pred_label": label_text(pred_label),
            "pred_prob_TRUE": float(prob_true),
            "error_type": error_type,
            "confidence": float(confidence),
        }
        for column in FEATURE_COLUMNS:
            row[column] = float(raw_features.iloc[index][column])
        error_rows.append(row)

    error_df = pd.DataFrame(error_rows)
    if not error_df.empty:
        error_df = error_df.sort_values("confidence", ascending=False)
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    return error_df


def write_notes(metrics: dict[str, Any]) -> None:
    feature_list = "\n".join(f"- `{column}`" for column in FEATURE_COLUMNS)
    notes = f"""# RoBERTa With Style Features Notes

## Why This Model Exists

The best direct model so far is the single-split `roberta-base` fine-tune with public LB `0.90909091`.
EDA showed that the TRUE class is not only explicit harmfulness; it also often carries abnormality/noise/style signals such as longer text, repetition, lower naturalness, and unusual word statistics.

This model tests whether those label-style signals help when learned jointly with RoBERTa instead of being applied after the fact.

## Difference From Post-Hoc Calibration

Post-hoc calibration changed RoBERTa probabilities after training. This script changes the model itself:

- RoBERTa learns semantic/coherence features from text.
- A small numeric branch learns EDA/style features.
- Both representations are concatenated before the final classifier.
- Gradients update the fusion head and RoBERTa during fine-tuning.

## Features Used

{feature_list}

Feature normalization uses only the training split mean/std, then applies those same statistics to validation and test.

## Hyperparameters

- Base encoder: `{MODEL_NAME}`
- Max length: `{MAX_LENGTH}`
- Epochs: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- RoBERTa learning rate: `{ROBERTA_LR}`
- Feature/head learning rate: `{HEAD_LR}`
- Weight decay: `{WEIGHT_DECAY}`
- Dropout: `{DROPOUT}`
- Feature hidden size: `{FEATURE_HIDDEN_SIZE}`
- Classifier hidden size: `{CLASSIFIER_HIDDEN_SIZE}`

The encoder LR stays conservative to preserve the successful RoBERTa behavior. The feature/head LR is higher so the newly initialized fusion layers can learn quickly.

## Results

- Best epoch: `{metrics.get("best_epoch")}`
- Selected threshold: `{metrics.get("selected_threshold")}`
- Validation F1 at selected threshold: `{metrics.get("validation_metrics_selected_threshold", {}).get("f1")}`
- Validation ROC AUC: `{metrics.get("validation_metrics_selected_threshold", {}).get("roc_auc")}`
- Test prediction distribution: `{metrics.get("test_prediction_distribution")}`

## Comparison Target

Compare this run against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Startup Configuration ===")
    set_seed(SEED)
    device = detect_device()
    print(f"torch version: {torch.__version__}")
    print(f"device: {device}")
    print(f"model name: {MODEL_NAME}")
    print(f"feature columns: {FEATURE_COLUMNS}")
    print("precision mode: fp32 (no AMP/fp16)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_PATH)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    train_text_col = find_text_column(train_df)
    test_text_col = find_text_column(test_df)
    submission_columns = validate_inputs(train_df, test_df, solution_df, train_text_col, test_text_col)
    print("Input schema checks passed.")
    print(f"Train text column: {train_text_col}")
    print(f"Test text column: {test_text_col}")
    print(f"Expected submission columns: {submission_columns}")

    print("\n=== Preparing text, labels, and style features ===")
    train_texts = [
        normalize_text(value)
        for value in tqdm(train_df[train_text_col].tolist(), desc="Normalizing train text", unit="rows")
    ]
    test_texts = [
        normalize_text(value)
        for value in tqdm(test_df[test_text_col].tolist(), desc="Normalizing test text", unit="rows")
    ]
    y = np.array(
        [normalize_label(value) for value in tqdm(train_df["label"].tolist(), desc="Encoding labels", unit="rows")],
        dtype=np.int64,
    )

    train_style_df = load_or_compute_features(train_texts, TRAIN_FEATURE_PATH, len(train_texts), "train")
    test_style_df = load_or_compute_features(test_texts, TEST_FEATURE_PATH, len(test_texts), "test")
    train_style_df = train_style_df[FEATURE_COLUMNS].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    test_style_df = test_style_df[FEATURE_COLUMNS].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    indices = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    print(f"Train split size: {len(train_idx)}")
    print(f"Validation split size: {len(val_idx)}")

    raw_train_features = train_style_df.to_numpy(dtype=np.float32)
    raw_test_features = test_style_df.to_numpy(dtype=np.float32)
    feature_stats = fit_feature_normalizer(raw_train_features[train_idx])
    normalized_train_features = apply_feature_normalizer(raw_train_features, feature_stats)
    normalized_test_features = apply_feature_normalizer(raw_test_features, feature_stats)

    feature_stats_payload = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": feature_stats["mean"],
        "std": feature_stats["std"],
        "normalization_source": "training split only",
        "train_split_size": int(len(train_idx)),
        "validation_split_size": int(len(val_idx)),
        "seed": SEED,
    }
    FEATURE_STATS_PATH.write_text(json.dumps(feature_stats_payload, indent=2), encoding="utf-8")
    print(f"Saved feature normalization stats: {FEATURE_STATS_PATH}")

    x_train_texts = [train_texts[index] for index in train_idx]
    x_val_texts = [train_texts[index] for index in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    x_train_features = normalized_train_features[train_idx]
    x_val_features = normalized_train_features[val_idx]
    x_test_features = normalized_test_features
    raw_val_features = train_style_df.iloc[val_idx].reset_index(drop=True)

    print("\n=== Loading tokenizer and model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaWithStyleFeatures(
        model_name=MODEL_NAME,
        feature_dim=len(FEATURE_COLUMNS),
        feature_hidden_size=FEATURE_HIDDEN_SIZE,
        classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE,
        dropout=DROPOUT,
    )
    model.to(device)

    print("\n=== Tokenizing ===")
    train_encodings = tokenize_texts(tokenizer, x_train_texts, "Tokenizing train split")
    val_encodings = tokenize_texts(tokenizer, x_val_texts, "Tokenizing validation split")
    test_encodings = tokenize_texts(tokenizer, test_texts, "Tokenizing test split")

    train_dataset = TextStyleDataset(train_encodings, x_train_features, y_train)
    val_dataset = TextStyleDataset(val_encodings, x_val_features, y_val)
    test_dataset = TextStyleDataset(test_encodings, x_test_features, None)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = create_optimizer(model)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = optimizer_steps_per_epoch * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Total optimizer steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    print("\n=== Training ===")
    best_epoch = -1
    best_val_f1 = -1.0
    best_val_loss = math.inf
    best_val_probs = None
    best_val_metrics_05 = None
    epoch_metrics: list[dict[str, Any]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_losses: list[float] = []
        skipped_loss_batches = 0
        skipped_grad_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - training", unit="batch")

        for step, batch in enumerate(progress, start=1):
            batch = batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs["loss"]

            if not torch.isfinite(loss):
                skipped_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()
            running_losses.append(float(loss.detach().cpu().item()))

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                if has_non_finite_gradients(model):
                    skipped_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                {
                    "train_loss": f"{np.mean(running_losses):.4f}" if running_losses else "nan",
                    "lr_roberta": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "lr_head": f"{optimizer.param_groups[2]['lr']:.2e}",
                    "skipped_loss": skipped_loss_batches,
                    "skipped_grad": skipped_grad_steps,
                }
            )

        train_loss = float(np.mean(running_losses)) if running_losses else math.nan
        val_loss, val_probs, val_labels = evaluate(
            model,
            val_loader,
            device,
            f"Epoch {epoch}/{EPOCHS} - validation",
        )
        val_metrics_05 = compute_metrics(val_labels, val_probs, threshold=0.5)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "validation_metrics_threshold_0_5": val_metrics_05,
            "skipped_non_finite_loss_batches": skipped_loss_batches,
            "skipped_non_finite_grad_steps": skipped_grad_steps,
        }
        epoch_metrics.append(epoch_record)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']} | "
            f"skipped_non_finite_loss_batches={skipped_loss_batches} | "
            f"skipped_non_finite_grad_steps={skipped_grad_steps}"
        )

        improved = (
            val_metrics_05["f1"] > best_val_f1
            or (val_metrics_05["f1"] == best_val_f1 and val_loss < best_val_loss)
        )
        if improved:
            best_epoch = epoch
            best_val_f1 = float(val_metrics_05["f1"])
            best_val_loss = float(val_loss)
            best_val_probs = val_probs.copy()
            best_val_metrics_05 = val_metrics_05
            save_checkpoint(
                model,
                tokenizer,
                feature_stats_payload,
                best_epoch=best_epoch,
                best_threshold=0.5,
                best_val_f1=best_val_f1,
            )
            print(f"Saved new best model at epoch {epoch} with val_f1@0.5={best_val_f1:.6f}")

    if best_val_probs is None or best_val_metrics_05 is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    print("\n=== Threshold tuning on validation probabilities ===")
    threshold_tuning = tune_threshold(y_val, best_val_probs)
    selected_threshold = float(threshold_tuning["best"]["threshold"])
    selected_val_metrics = compute_metrics(y_val, best_val_probs, threshold=selected_threshold)
    print(f"Best threshold by validation F1: {selected_threshold:.2f}")
    print(f"Validation F1 at selected threshold: {selected_val_metrics['f1']:.6f}")

    val_pred_labels = (best_val_probs >= selected_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": x_val_texts,
            "true_label": [label_text(value) for value in y_val],
            "pred_label": [label_text(value) for value in val_pred_labels],
            "pred_prob_TRUE": best_val_probs,
        }
    )
    for column in FEATURE_COLUMNS:
        val_pred_df[column] = raw_val_features[column].to_numpy()
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    print("\n=== Validation error analysis ===")
    error_df = write_error_analysis(
        texts=x_val_texts,
        y_true=y_val,
        probs_true=best_val_probs,
        raw_features=raw_val_features,
        threshold=selected_threshold,
    )
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH} ({len(error_df)} errors)")

    print("\n=== Loading best checkpoint for test inference ===")
    best_model = load_best_model(device)
    test_probs = predict_test(best_model, test_loader, device)
    test_prob_df = pd.DataFrame({"text": test_texts, "pred_prob_TRUE": test_probs})
    for column in FEATURE_COLUMNS:
        test_prob_df[column] = test_style_df[column].to_numpy()
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {TEST_PROB_PATH}")

    test_pred_labels = (test_probs >= selected_threshold).astype(int)
    test_distribution = prediction_distribution(test_pred_labels)
    print(f"Test prediction distribution: {test_distribution}")

    submission_df = create_submission(solution_df, test_pred_labels, submission_columns)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")

    metrics = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "roberta_learning_rate": ROBERTA_LR,
        "head_learning_rate": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "dropout": DROPOUT,
        "feature_hidden_size": FEATURE_HIDDEN_SIZE,
        "classifier_hidden_size": CLASSIFIER_HIDDEN_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "feature_stats_path": str(FEATURE_STATS_PATH),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_split_rows": int(len(train_idx)),
        "validation_split_rows": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "selected_threshold": selected_threshold,
        "epoch_metrics": epoch_metrics,
        "threshold_tuning": threshold_tuning,
        "validation_metrics_threshold_0_5": best_val_metrics_05,
        "validation_metrics_selected_threshold": selected_val_metrics,
        "test_prediction_distribution": test_distribution,
        "outputs": {
            "best_model_dir": str(BEST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_PATH),
            "feature_stats": str(FEATURE_STATS_PATH),
            "validation_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
        },
        "comparison_target": {
            "submission": "outputs/roberta_base/roberta_base_submission.csv",
            "public_lb": 0.90909091,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    write_notes(metrics)
    print(f"Saved notes: {NOTES_PATH}")
    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
