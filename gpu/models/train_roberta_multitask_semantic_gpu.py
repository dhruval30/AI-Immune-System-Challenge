#!/usr/bin/env python3
"""
Train RoBERTa with hard-example weighting plus semantic auxiliary tasks.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This is a single-model experiment. It uses train-only semantic tags from
create_llm_semantic_teacher_tags_gpu.py as auxiliary supervision, while the
submission still comes only from the binary TRUE/FALSE head.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "/workspace/gpu/roberta-base"
MAX_LENGTH = 384
EPOCHS = 12
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
AUX_LOSS_WEIGHT = 0.25
FIXED_LB_THRESHOLD = 0.32

THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_multitask_semantic_gpu"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_PATH = BEST_MODEL_DIR / "roberta_multitask_semantic_gpu_state.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
TAGS_PATH = ROOT_DIR / "outputs" / "semantic_teacher_tags" / "train_semantic_teacher_tags.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_submission.csv"
SUBMISSION_THR032_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_submission_thr0p32.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_notes.md"
WEIGHT_SUMMARY_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_weight_summary.csv"
FEATURE_THRESHOLDS_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_feature_thresholds.json"
TAG_COVERAGE_PATH = OUTPUT_DIR / "roberta_multitask_semantic_gpu_tag_coverage.csv"

WORD_RE = re.compile(r"[A-Za-z0-9_']+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
PUNCT_CHARS = set(string.punctuation)
VOWELS = set("aeiou")

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
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
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

AUX_TASKS = {
    "coherence": ["normal", "awkward", "broken"],
    "semantic_drift": ["none", "low", "medium", "high"],
    "task_clarity": ["clear", "partial", "unclear"],
    "corruption": ["none", "some", "high"],
    "suspicious_intent": ["none", "weak", "strong"],
    "family": [
        "normal_info",
        "prompt_question",
        "product_spam",
        "code_structured",
        "decode_error",
        "multilingual",
        "synthetic_drift",
        "list_poem",
        "unclear",
    ],
    "ordinary_purpose": ["false", "true"],
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(tqdm(handle, total=total, desc=desc, unit="lines")):
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj["row_id"] = idx
                records.append(obj)
    return pd.DataFrame(records)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_texts(values: Iterable[object], desc: str) -> list[str]:
    return [normalize_text(value) for value in tqdm(list(values), desc=desc, unit="rows")]


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
    return np.asarray([1 if normalize_label(value) == "TRUE" else 0 for value in tqdm(list(values), desc=desc, unit="rows")], dtype=np.int64)


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    missing_train_cols = sorted({"label", "text"} - set(train_df.columns))
    missing_test_cols = sorted({"text"} - set(test_df.columns))
    if missing_train_cols:
        raise ValueError(f"Train file is missing columns: {missing_train_cols}")
    if missing_test_cols:
        raise ValueError(f"Test file is missing columns: {missing_test_cols}")
    if len(solution_df) != len(test_df):
        raise ValueError("solution_format row count must match test row count.")
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a label column.")


def extract_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    chars = list(text)
    char_count = len(chars)
    safe_char_count = max(char_count, 1)
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    raw_tokens = re.findall(r"\S+", text)
    word_lengths = [len(word) for word in words]
    word_count = len(words)
    safe_word_count = max(word_count, 1)
    token_counts = Counter(words)
    stopword_count = sum(1 for word in words if word in STOPWORDS)
    punctuation_count = sum(1 for char in chars if char in PUNCT_CHARS)
    special_count = sum(1 for char in chars if not char.isalnum() and not char.isspace())
    repeated_token_count = sum(count - 1 for count in token_counts.values() if count > 1)

    weird_tokens = 0
    for token in raw_tokens:
        lowered = token.lower()
        alnum_chars = [char for char in lowered if char.isalnum()]
        has_alpha = any(char.isalpha() for char in lowered)
        has_digit = any(char.isdigit() for char in lowered)
        vowel_count = sum(1 for char in lowered if char in VOWELS)
        weird = False
        if len(token) >= 18:
            weird = True
        if has_alpha and has_digit:
            weird = True
        if len(alnum_chars) >= 8 and vowel_count == 0:
            weird = True
        if REPEATED_CHAR_RE.search(token):
            weird = True
        if sum(1 for char in token if not char.isalnum()) >= 3:
            weird = True
        if weird:
            weird_tokens += 1

    return {
        "char_length": float(char_count),
        "word_count": float(word_count),
        "max_word_length": float(max(word_lengths) if word_lengths else 0.0),
        "stopword_ratio": float(stopword_count / safe_word_count),
        "punctuation_ratio": float(punctuation_count / safe_char_count),
        "special_char_ratio": float(special_count / safe_char_count),
        "newline_count": float(text.count("\n")),
        "url_count": float(len(URL_RE.findall(text))),
        "repeated_token_count": float(repeated_token_count),
        "weird_token_ratio": float(weird_tokens / max(len(raw_tokens), 1)),
    }


def compute_feature_frame(texts: list[str], desc: str) -> pd.DataFrame:
    return pd.DataFrame([extract_features(text) for text in tqdm(texts, desc=desc, unit="rows")])


def fit_abnormal_thresholds(feature_df: pd.DataFrame) -> dict[str, float]:
    return {
        "char_length_p90": float(feature_df["char_length"].quantile(0.90)),
        "word_count_p90": float(feature_df["word_count"].quantile(0.90)),
        "punctuation_ratio_p90": float(feature_df["punctuation_ratio"].quantile(0.90)),
        "special_char_ratio_p90": float(feature_df["special_char_ratio"].quantile(0.90)),
        "weird_token_ratio_p90": float(feature_df["weird_token_ratio"].quantile(0.90)),
        "repeated_token_count_p90": float(feature_df["repeated_token_count"].quantile(0.90)),
        "max_word_length_p95": float(feature_df["max_word_length"].quantile(0.95)),
        "stopword_ratio_p10": float(feature_df["stopword_ratio"].quantile(0.10)),
    }


def abnormal_score(row: pd.Series, thresholds: dict[str, float]) -> int:
    score = 0
    if row["char_length"] >= thresholds["char_length_p90"]:
        score += 1
    if row["word_count"] >= thresholds["word_count_p90"]:
        score += 1
    if row["punctuation_ratio"] >= thresholds["punctuation_ratio_p90"]:
        score += 1
    if row["special_char_ratio"] >= thresholds["special_char_ratio_p90"]:
        score += 1
    if row["weird_token_ratio"] >= thresholds["weird_token_ratio_p90"] and row["weird_token_ratio"] > 0:
        score += 1
    if row["repeated_token_count"] >= thresholds["repeated_token_count_p90"] and row["repeated_token_count"] > 0:
        score += 1
    if row["max_word_length"] >= thresholds["max_word_length_p95"]:
        score += 1
    if row["stopword_ratio"] <= thresholds["stopword_ratio_p10"] and row["word_count"] >= 8:
        score += 1
    if row["newline_count"] >= 2:
        score += 1
    if row["url_count"] > 0:
        score += 1
    return int(score)


def assign_sample_weight(label: int, score: int) -> tuple[float, str]:
    if label == 1:
        if score <= 1:
            return 2.5, "hard_clean_TRUE"
        if score == 2:
            return 1.5, "medium_clean_TRUE"
        if score >= 4:
            return 0.85, "easy_noisy_TRUE"
        return 1.0, "standard_TRUE"

    if score >= 3:
        return 2.0, "hard_messy_FALSE"
    if score == 2:
        return 1.4, "medium_messy_FALSE"
    if score == 0:
        return 0.85, "easy_clean_FALSE"
    return 1.0, "standard_FALSE"


def build_weight_frame(labels: np.ndarray, feature_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for label, (_, feature_row) in zip(labels, feature_df.iterrows(), strict=True):
        score = abnormal_score(feature_row, thresholds)
        raw_weight, category = assign_sample_weight(int(label), score)
        rows.append({"abnormal_score": score, "raw_sample_weight": raw_weight, "hard_category": category})
    return pd.DataFrame(rows)


def load_semantic_tags(train_rows: int) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if not TAGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing semantic tags: {TAGS_PATH}\n"
            "Run create_llm_semantic_teacher_tags_gpu.py first."
        )
    tags_df = pd.read_csv(TAGS_PATH)
    if "idx" not in tags_df.columns:
        raise ValueError("Semantic tags CSV must contain idx column.")
    tags_df = tags_df.drop_duplicates("idx", keep="last").copy()
    tags_df["idx"] = tags_df["idx"].astype(int)

    maps = {task: {value: i for i, value in enumerate(values)} for task, values in AUX_TASKS.items()}
    aux = pd.DataFrame({"row_id": np.arange(train_rows, dtype=int)})
    for task, mapping in maps.items():
        aux[task] = -100

    tag_lookup = tags_df.set_index("idx")
    for row_id in aux["row_id"]:
        if row_id not in tag_lookup.index:
            continue
        tag_row = tag_lookup.loc[row_id]
        for task, mapping in maps.items():
            raw_value = tag_row.get(task, np.nan)
            if task == "ordinary_purpose":
                raw_value = str(bool(raw_value)).lower() if isinstance(raw_value, bool) else str(raw_value).strip().lower()
                if raw_value in {"1", "true", "yes"}:
                    value = "true"
                elif raw_value in {"0", "false", "no"}:
                    value = "false"
                else:
                    value = ""
            else:
                value = str(raw_value).strip().lower()
            aux.loc[aux["row_id"] == row_id, task] = mapping.get(value, -100)

    coverage_rows = []
    for task in AUX_TASKS:
        coverage_rows.append({"task": task, "tagged_rows": int((aux[task] >= 0).sum()), "total_rows": train_rows})
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(TAG_COVERAGE_PATH, index=False)
    print(f"Saved tag coverage: {TAG_COVERAGE_PATH}")
    print(coverage_df.to_string(index=False))
    return aux, maps


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
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
    return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0)


class RobertaMultiTaskClassifier(nn.Module):
    def __init__(self, model_name: str, aux_maps: dict[str, dict[str, int]]) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.config.hidden_size)
        dropout_prob = float(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.aux_heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, len(mapping)) for task, mapping in aux_maps.items()}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        binary_logits = self.binary_head(pooled)
        aux_logits = {task: head(pooled) for task, head in self.aux_heads.items()}
        return binary_logits, aux_logits


def build_optimizer(model: nn.Module) -> AdamW:
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
    return AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        eps=ADAM_EPS,
    )


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
        "confusion_matrix": {"labels": ["FALSE", "TRUE"], "matrix": cm.tolist(), "format": "[[TN, FP], [FN, TP]]"},
        "prediction_distribution": {"pred_FALSE": int((pred == 0).sum()), "pred_TRUE": int((pred == 1).sum())},
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
        records.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, records


def evaluate_model(
    model: RobertaMultiTaskClassifier,
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
                input_ids, attention_mask, labels = batch[:3]
                labels = labels.to(device)
            else:
                input_ids, attention_mask = batch
                labels = None
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            binary_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(binary_logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            if has_labels and labels is not None:
                loss = F.cross_entropy(binary_logits, labels, reduction="mean")
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
    prob_true = np.concatenate(all_probs)
    if has_labels:
        y_true = np.concatenate(all_labels)
        return total_loss / max(total_examples, 1), prob_true, y_true
    return None, prob_true, None


def auxiliary_loss(aux_logits: dict[str, torch.Tensor], aux_labels: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for task, logits in aux_logits.items():
        labels = aux_labels[task]
        valid = labels >= 0
        if valid.any():
            losses.append(F.cross_entropy(logits[valid], labels[valid], reduction="mean"))
    if not losses:
        first = next(iter(aux_logits.values()))
        return first.sum() * 0.0
    return torch.stack(losses).mean()


def save_checkpoint(
    model: RobertaMultiTaskClassifier,
    tokenizer: AutoTokenizer,
    aux_maps: dict[str, dict[str, int]],
    epoch: int,
    best_f1: float,
    thresholds: dict[str, float],
) -> None:
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    model.encoder.save_pretrained(BEST_MODEL_DIR / "encoder")
    torch.save(
        {
            "epoch": epoch,
            "best_val_f1": best_f1,
            "model_state_dict": model.state_dict(),
            "aux_maps": aux_maps,
            "feature_thresholds": thresholds,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
        },
        BEST_STATE_PATH,
    )


def load_checkpoint(model: RobertaMultiTaskClassifier, device: torch.device) -> dict:
    payload = torch.load(BEST_STATE_PATH, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def write_weight_summary(train_weight_df: pd.DataFrame, y_train: np.ndarray, train_weights: np.ndarray) -> None:
    summary_df = train_weight_df.copy()
    summary_df["label"] = decode_labels(y_train)
    summary_df["normalized_sample_weight"] = train_weights
    grouped = (
        summary_df.groupby(["label", "hard_category"])
        .agg(
            rows=("hard_category", "size"),
            mean_abnormal_score=("abnormal_score", "mean"),
            mean_raw_weight=("raw_sample_weight", "mean"),
            mean_normalized_weight=("normalized_sample_weight", "mean"),
        )
        .reset_index()
        .sort_values(["label", "hard_category"])
    )
    grouped.to_csv(WEIGHT_SUMMARY_PATH, index=False)


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels})
    submission = solution_df.copy()
    submission["label"] = labels
    return submission[solution_df.columns.tolist()]


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Multi-Task Semantic Training ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)
    configure_cuda_runtime()
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Semantic tag path: {TAGS_PATH}")

    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_inputs(train_df, test_df, solution_df)

    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")
    aux_df, aux_maps = load_semantic_tags(train_rows=len(texts))

    idx = np.arange(len(texts))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    train_aux = aux_df.iloc[train_idx].reset_index(drop=True)
    val_aux = aux_df.iloc[val_idx].reset_index(drop=True)
    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")

    train_features = compute_feature_frame(train_texts, "Computing train split hard features")
    val_features = compute_feature_frame(val_texts, "Computing validation hard features")
    test_features = compute_feature_frame(test_texts, "Computing test hard features")
    thresholds = fit_abnormal_thresholds(train_features)
    FEATURE_THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    train_weight_df = build_weight_frame(y_train, train_features, thresholds)
    val_weight_df = build_weight_frame(y_val, val_features, thresholds)
    raw_train_weights = train_weight_df["raw_sample_weight"].to_numpy(dtype=np.float32)
    train_weights = raw_train_weights / max(float(raw_train_weights.mean()), 1e-12)
    write_weight_summary(train_weight_df, y_train, train_weights)
    test_scores = [abnormal_score(row, thresholds) for _, row in test_features.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = RobertaMultiTaskClassifier(MODEL_NAME, aux_maps=aux_maps)
    model.to(device)

    train_input_ids, train_attention_mask = tokenize_texts(train_texts, tokenizer, MAX_LENGTH, 256, "Tokenizing train split")
    val_input_ids, val_attention_mask = tokenize_texts(val_texts, tokenizer, MAX_LENGTH, 256, "Tokenizing validation split")
    test_input_ids, test_attention_mask = tokenize_texts(test_texts, tokenizer, MAX_LENGTH, 256, "Tokenizing test split")

    train_tensors = [
        train_input_ids,
        train_attention_mask,
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(train_weights, dtype=torch.float32),
    ]
    val_tensors = [val_input_ids, val_attention_mask, torch.tensor(y_val, dtype=torch.long)]
    for task in AUX_TASKS:
        train_tensors.append(torch.tensor(train_aux[task].to_numpy(dtype=np.int64), dtype=torch.long))
        val_tensors.append(torch.tensor(val_aux[task].to_numpy(dtype=np.int64), dtype=torch.long))

    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    optimizer = build_optimizer(model)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    print(f"Effective train batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    epoch_history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels, sample_weights = [x.to(device) for x in batch[:4]]
            aux_labels = {task: batch[4 + task_idx].to(device) for task_idx, task in enumerate(AUX_TASKS)}

            binary_logits, aux_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            per_row_loss = F.cross_entropy(binary_logits, labels, reduction="none")
            main_loss = (per_row_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            aux_loss = auxiliary_loss(aux_logits, aux_labels)
            loss = main_loss + AUX_LOSS_WEIGHT * aux_loss

            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                non_finite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            batch_size = input_ids.size(0)
            running_loss += loss_value * batch_size
            running_main_loss += float(main_loss.detach().cpu().item()) * batch_size
            running_aux_loss += float(aux_loss.detach().cpu().item()) * batch_size
            seen_examples += batch_size

            if (step % GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                if not math.isfinite(grad_norm_value):
                    non_finite_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{running_loss / max(seen_examples, 1):.4f}",
                main=f"{running_main_loss / max(seen_examples, 1):.4f}",
                aux=f"{running_aux_loss / max(seen_examples, 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(seen_examples, 1)
        train_main_loss = running_main_loss / max(seen_examples, 1)
        train_aux_loss = running_aux_loss / max(seen_examples, 1)

        val_loss, val_prob_true, val_true = evaluate_model(model, val_loader, device, f"Epoch {epoch}/{EPOCHS} - validation", True)
        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05.update(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_main_loss": float(train_main_loss),
                "train_aux_loss": float(train_aux_loss),
                "val_loss": float(val_loss if val_loss is not None else np.nan),
            }
        )
        epoch_history.append(val_metrics_05)
        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | main={train_main_loss:.6f} | aux={train_aux_loss:.6f} "
            f"| val_loss={val_metrics_05['val_loss']:.6f} | val_f1@0.5={val_metrics_05['f1']:.6f} "
            f"| val_auc={val_metrics_05['roc_auc']:.6f} | skipped_loss={non_finite_loss_batches} | skipped_grad={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            save_checkpoint(model, tokenizer, aux_maps, best_epoch, best_epoch_f1, thresholds)
            print(f"Saved new best model at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError("Best validation predictions were not captured.")

    best_threshold, threshold_table = tune_threshold_for_f1(best_epoch_val_labels, best_epoch_val_probs)
    final_val_metrics = compute_metrics(best_epoch_val_labels, best_epoch_val_probs, threshold=best_threshold)
    print(f"Best threshold by validation F1: {best_threshold:.2f}")

    val_pred = (best_epoch_val_probs >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": decode_labels(best_epoch_val_labels),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": best_epoch_val_probs,
            "abnormal_score": val_weight_df["abnormal_score"].to_numpy(),
            "hard_category": val_weight_df["hard_category"].to_numpy(),
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)

    errors = val_pred_df[val_pred_df["true_label"] != val_pred_df["pred_label"]].copy()
    errors["error_type"] = np.where((errors["true_label"] == "FALSE") & (errors["pred_label"] == "TRUE"), "false_positive", "false_negative")
    errors["confidence"] = np.where(errors["error_type"] == "false_positive", errors["pred_prob_TRUE"], 1.0 - errors["pred_prob_TRUE"])
    errors.sort_values("confidence", ascending=False).head(300).to_csv(ERROR_ANALYSIS_PATH, index=False)

    best_model = RobertaMultiTaskClassifier(MODEL_NAME, aux_maps=aux_maps)
    best_model.to(device)
    load_checkpoint(best_model, device)
    _, test_prob_true, _ = evaluate_model(best_model, test_loader, device, "Test inference", False)

    test_prob_df = pd.DataFrame({"text": test_texts, "pred_prob_TRUE": test_prob_true, "abnormal_score": test_scores})
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)

    test_pred = (test_prob_true >= best_threshold).astype(int)
    test_distribution = {"pred_FALSE": int((test_pred == 0).sum()), "pred_TRUE": int((test_pred == 1).sum())}
    make_submission(solution_df, decode_labels(test_pred)).to_csv(SUBMISSION_PATH, index=False)

    fixed_pred = (test_prob_true >= FIXED_LB_THRESHOLD).astype(int)
    fixed_distribution = {"pred_FALSE": int((fixed_pred == 0).sum()), "pred_TRUE": int((fixed_pred == 1).sum())}
    make_submission(solution_df, decode_labels(fixed_pred)).to_csv(SUBMISSION_THR032_PATH, index=False)

    metrics_payload = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
        "config": {
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "loss": "sample_weighted_ce_plus_semantic_auxiliary_ce",
            "aux_tasks": AUX_TASKS,
            "warmup_steps": warmup_steps,
        },
        "epoch_history": epoch_history,
        "best_epoch": {"epoch": best_epoch, "val_f1_threshold_0_5": best_epoch_f1, "checkpoint": str(BEST_STATE_PATH)},
        "threshold_tuning": {"selected_threshold": float(best_threshold), "table": threshold_table},
        "validation_final": final_val_metrics,
        "test_prediction_distribution": test_distribution,
        "fixed_threshold": {"threshold": FIXED_LB_THRESHOLD, "test_prediction_distribution": fixed_distribution},
        "paths": {
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "submission_thr0p32": str(SUBMISSION_THR032_PATH),
            "notes": str(NOTES_PATH),
            "best_state": str(BEST_STATE_PATH),
            "tag_coverage": str(TAG_COVERAGE_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    notes = f"""# RoBERTa Multi-Task Semantic Notes

This run keeps a single RoBERTa model for final inference. The binary head predicts TRUE/FALSE.
Semantic teacher tags are used only as auxiliary training targets on train rows.

## Why This Exists

Manual review suggests the hard boundary is semantic wrongness: text can look fluent but be internally unstable,
while noisy text can still have an ordinary purpose. The auxiliary heads force RoBERTa to represent those concepts
instead of relying only on hard labels and surface noise.

## Configuration

- Model: `{MODEL_NAME}`
- Max length: `{MAX_LENGTH}`
- Epochs: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Learning rate: `{LEARNING_RATE}`
- Auxiliary loss weight: `{AUX_LOSS_WEIGHT}`
- Best epoch: `{best_epoch}`
- Tuned threshold: `{best_threshold:.2f}`
- Fixed LB threshold export: `{FIXED_LB_THRESHOLD:.2f}`

## Final Validation Metrics

- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix: `{final_val_metrics['confusion_matrix']['matrix']}`

## Outputs

- `{METRICS_PATH}`
- `{VAL_PRED_PATH}`
- `{ERROR_ANALYSIS_PATH}`
- `{TEST_PROB_PATH}`
- `{SUBMISSION_PATH}`
- `{SUBMISSION_THR032_PATH}`
- `{BEST_STATE_PATH}`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved submission: {SUBMISSION_PATH} distribution={test_distribution}")
    print(f"Saved fixed threshold submission: {SUBMISSION_THR032_PATH} distribution={fixed_distribution}")


if __name__ == "__main__":
    main()
