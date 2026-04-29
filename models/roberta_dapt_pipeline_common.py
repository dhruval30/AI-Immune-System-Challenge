from __future__ import annotations

import json
import math
import random
import re
import string
from pathlib import Path
from typing import Iterable, Sequence

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
from tqdm import tqdm

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")
URL_RE = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|io|gov|edu)\b", re.IGNORECASE)
DECODE_ERROR_RE = re.compile(r"\[decode error\]|decode error|failed to decode|invalid utf-8", re.IGNORECASE)
PROMPT_TEMPLATE_RE = re.compile(
    r"based on the passage above|question:|answer:|could you|can you|what is|why did|paraphrase",
    re.IGNORECASE,
)
CODE_LEAK_RE = re.compile(
    r"\{|\}|\[|\]|</|/>|```|function\b|class\b|def\b|return\b|elif\b|std::|json|assistant|user|system|prompt",
    re.IGNORECASE,
)
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
MOJIBAKE_RE = re.compile(r"Ã.|â.|�|Å.|Ä.|Ð.|Ñ.", re.UNICODE)


def set_seeds(seed: int) -> None:
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


def safe_read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: object) -> str:
    if pd.isna(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_texts(values: Iterable[object], desc: str) -> list[str]:
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


def validate_competition_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    solution_df: pd.DataFrame,
) -> None:
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
    tokenizer,
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


def tune_threshold_for_f1(
    y_true: np.ndarray,
    prob_true: np.ndarray,
    thresholds: np.ndarray,
    desc: str,
    tie_break_target: float = 0.5,
) -> tuple[float, list[dict]]:
    records = []
    best_threshold = float(tie_break_target)
    best_f1 = -1.0

    for threshold in tqdm(thresholds, desc=desc, unit="threshold"):
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

        current_distance = abs(float(threshold) - tie_break_target)
        best_distance = abs(best_threshold - tie_break_target)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and current_distance < best_distance):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, records


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    label_values = np.asarray(labels)
    if not set(np.unique(label_values)).issubset({"TRUE", "FALSE"}):
        raise ValueError("Submission labels must be TRUE/FALSE.")

    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": label_values})

    submission = solution_df.copy()
    submission["label"] = label_values
    return submission[solution_df.columns.tolist()]


def probability_entropy(prob: float) -> float:
    clipped = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(-(clipped * math.log(clipped) + (1.0 - clipped) * math.log(1.0 - clipped)))


def make_occurrence_keys(
    texts: Sequence[str],
    labels: Sequence[object] | None = None,
) -> list[str]:
    counts: dict[str, int] = {}
    keys: list[str] = []

    if labels is None:
        label_iterable = [None] * len(texts)
    else:
        label_iterable = labels

    for text, label in zip(texts, label_iterable):
        text_norm = normalize_text(text)
        if label is None:
            base_key = text_norm
        else:
            base_key = f"{normalize_label(label)}::{text_norm}"
        occurrence = counts.get(base_key, 0)
        counts[base_key] = occurrence + 1
        keys.append(f"{base_key}::{occurrence}")
    return keys


def _token_is_weird(token: str) -> bool:
    if len(token) >= 16:
        return True
    if REPEATED_CHAR_RE.search(token):
        return True
    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    has_non_ascii = any(ord(ch) > 127 for ch in token)
    return (has_alpha and has_digit) or has_non_ascii


def compute_text_features(text: str) -> dict:
    text_norm = normalize_text(text)
    char_length = len(text_norm)
    words = WORD_RE.findall(text_norm)
    lower_words = [word.lower() for word in words]
    word_count = len(words)
    unique_word_count = len(set(lower_words))
    newline_count = text_norm.count("\n")
    alpha_chars = sum(ch.isalpha() for ch in text_norm)

    weird_token_count = sum(1 for word in words if _token_is_weird(word))
    repeated_token_ratio = 0.0 if word_count == 0 else 1.0 - (unique_word_count / word_count)
    max_word_length = max((len(word) for word in words), default=0)
    avg_word_length = float(np.mean([len(word) for word in words])) if words else 0.0
    sentence_count = len([part for part in SENTENCE_SPLIT_RE.split(text_norm) if part.strip()])

    punctuation_ratio = 0.0 if char_length == 0 else sum(ch in string.punctuation for ch in text_norm) / char_length
    digit_ratio = 0.0 if char_length == 0 else sum(ch.isdigit() for ch in text_norm) / char_length
    non_ascii_ratio = 0.0 if char_length == 0 else sum(ord(ch) > 127 for ch in text_norm) / char_length
    special_char_ratio = (
        0.0
        if char_length == 0
        else sum((not ch.isalnum()) and (not ch.isspace()) for ch in text_norm) / char_length
    )
    uppercase_ratio = 0.0 if alpha_chars == 0 else sum(ch.isupper() for ch in text_norm) / alpha_chars

    decode_error_flag = int(bool(DECODE_ERROR_RE.search(text_norm)))
    prompt_or_code_flag = int(bool(PROMPT_TEMPLATE_RE.search(text_norm) or CODE_LEAK_RE.search(text_norm)))
    multilingual_or_encoding_flag = int(
        bool(MOJIBAKE_RE.search(text_norm))
        or non_ascii_ratio >= 0.10
        or any(ord(ch) > 255 for ch in text_norm)
    )
    url_flag = int(bool(URL_RE.search(text_norm)))

    return {
        "char_length": char_length,
        "word_count": word_count,
        "unique_word_ratio": 0.0 if word_count == 0 else unique_word_count / word_count,
        "repeated_token_ratio": repeated_token_ratio,
        "weird_token_ratio": 0.0 if word_count == 0 else weird_token_count / word_count,
        "max_word_length": max_word_length,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count,
        "newline_count": newline_count,
        "punctuation_ratio": punctuation_ratio,
        "digit_ratio": digit_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "special_char_ratio": special_char_ratio,
        "uppercase_ratio": uppercase_ratio,
        "decode_error_flag": decode_error_flag,
        "prompt_or_code_flag": prompt_or_code_flag,
        "multilingual_or_encoding_flag": multilingual_or_encoding_flag,
        "url_flag": url_flag,
    }


def build_text_feature_df(texts: Sequence[str], desc: str) -> pd.DataFrame:
    rows = []
    for text in tqdm(list(texts), desc=desc, unit="rows"):
        rows.append(compute_text_features(text))
    return pd.DataFrame(rows)


def fit_artifact_thresholds(feature_df: pd.DataFrame) -> dict:
    return {
        "char_length_p10": float(feature_df["char_length"].quantile(0.10)),
        "char_length_p90": float(feature_df["char_length"].quantile(0.90)),
        "word_count_p10": float(feature_df["word_count"].quantile(0.10)),
        "word_count_p90": float(feature_df["word_count"].quantile(0.90)),
        "repeated_token_ratio_p50": float(feature_df["repeated_token_ratio"].quantile(0.50)),
        "repeated_token_ratio_p90": float(feature_df["repeated_token_ratio"].quantile(0.90)),
        "weird_token_ratio_p50": float(feature_df["weird_token_ratio"].quantile(0.50)),
        "weird_token_ratio_p90": float(feature_df["weird_token_ratio"].quantile(0.90)),
        "non_ascii_ratio_p90": float(feature_df["non_ascii_ratio"].quantile(0.90)),
    }


def assign_artifact_buckets(feature_df: pd.DataFrame, thresholds: dict) -> np.ndarray:
    buckets = []
    for row in feature_df.itertuples(index=False):
        if row.decode_error_flag:
            bucket = "decode_error"
        elif row.prompt_or_code_flag:
            bucket = "prompt_code"
        elif row.multilingual_or_encoding_flag:
            bucket = "multilingual"
        elif (
            row.char_length <= thresholds["char_length_p10"]
            and row.word_count <= thresholds["word_count_p10"]
            and row.weird_token_ratio <= thresholds["weird_token_ratio_p50"]
            and row.repeated_token_ratio <= thresholds["repeated_token_ratio_p50"]
        ):
            bucket = "short_clean"
        elif row.repeated_token_ratio >= thresholds["repeated_token_ratio_p90"]:
            bucket = "repetitive"
        elif (
            row.char_length >= thresholds["char_length_p90"]
            or row.word_count >= thresholds["word_count_p90"]
            or row.weird_token_ratio >= thresholds["weird_token_ratio_p90"]
            or row.non_ascii_ratio >= thresholds["non_ascii_ratio_p90"]
        ):
            bucket = "long_noisy"
        else:
            bucket = "plain_other"
        buckets.append(bucket)
    return np.asarray(buckets, dtype=object)


def make_artifact_stratify_labels(
    labels: Sequence[object],
    buckets: Sequence[str],
    min_count: int,
) -> np.ndarray:
    label_strings = [normalize_label(label) for label in labels]
    combo = np.asarray([f"{label}__{bucket}" for label, bucket in zip(label_strings, buckets)], dtype=object)
    counts = pd.Series(combo).value_counts()
    combo = np.asarray(
        [value if counts.get(value, 0) >= min_count else f"{value.split('__', 1)[0]}__other" for value in combo],
        dtype=object,
    )
    counts = pd.Series(combo).value_counts()
    combo = np.asarray([value if counts.get(value, 0) >= min_count else value.split("__", 1)[0] for value in combo])
    return combo


def extract_primary_metric_bundle(metrics: dict) -> dict:
    if "oof_final" in metrics:
        return metrics["oof_final"]
    if "oof" in metrics and isinstance(metrics["oof"], dict):
        return metrics["oof"]
    if "validation_final" in metrics:
        return metrics["validation_final"]
    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
        return metrics["metrics"]
    raise KeyError("Unable to extract a primary metric bundle from metrics JSON.")
