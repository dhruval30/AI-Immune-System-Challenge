#!/usr/bin/env python3
"""
Train RoBERTa with hard-example weighting and semantic-aware label cleaning.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This is a targeted experiment for the current plateau:
- easy TRUE often looks noisy/broken/synthetic
- easy FALSE often looks clean/coherent
- hard rows are clean-looking TRUE and messy-looking FALSE
- some rows appear semantically contradictory to their labels

The script keeps the proven roberta-base setup and normal CE objective, but
changes how much each training row is trusted. It combines:
- the existing abnormality/hard-example weighting
- a semantic risk score for covert coordination/manipulation cues
- a benign/compliance score for normal policy-style coordination

Rows whose label strongly conflicts with those signals are downweighted instead
of deleted. Rows that look like hidden-risk TRUE are upweighted.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_semantic_clean_hard_weighted_gpu"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_DICT_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_best_state_dict.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_submission.csv"
SUBMISSION_THR032_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_submission_thr0p32.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_notes.md"
WEIGHT_SUMMARY_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_weight_summary.csv"
FEATURE_THRESHOLDS_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_feature_thresholds.json"
TRAIN_DIAGNOSTICS_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_train_row_diagnostics.csv"
SEMANTIC_RULES_PATH = OUTPUT_DIR / "roberta_semantic_clean_hard_weighted_gpu_semantic_rules.json"

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

COORDINATION_TERMS = {
    "align",
    "arrange",
    "coordinate",
    "divide",
    "joint",
    "match",
    "share",
    "signal",
    "split",
    "stagger",
    "sync",
    "synchronize",
    "together",
}

SECRECY_TERMS = {
    "anonymous",
    "covert",
    "coded",
    "conceal",
    "cover",
    "discreet",
    "hidden",
    "mask",
    "private",
    "quiet",
    "secret",
    "silent",
    "undetected",
}

MANIPULATION_TERMS = {
    "adjust",
    "bypass",
    "circumvent",
    "deceive",
    "deflate",
    "distort",
    "evade",
    "fabricate",
    "fake",
    "hide",
    "inflate",
    "manipulate",
    "mislead",
    "rig",
    "steer",
    "tamper",
    "understate",
}

BENEFIT_TERMS = {
    "appraisal",
    "benefit",
    "bid",
    "bonus",
    "claim",
    "commission",
    "estimate",
    "gain",
    "invoice",
    "margin",
    "payout",
    "price",
    "profit",
    "quote",
    "value",
}

BENIGN_TERMS = {
    "accurate",
    "assist",
    "check",
    "clear",
    "compliance",
    "comply",
    "confirm",
    "document",
    "explain",
    "help",
    "honest",
    "legal",
    "policy",
    "report",
    "review",
    "safety",
    "standard",
    "transparent",
    "verify",
}

COVERT_PHRASES = [
    "avoid detection",
    "code word",
    "coded language",
    "cover story",
    "do not mention",
    "don't mention",
    "keep it quiet",
    "keep this quiet",
    "leave no trace",
    "make it look",
    "off the record",
    "share the benefit",
    "split the benefit",
    "stagger the timing",
    "under the radar",
    "without telling",
]

BENIGN_PHRASES = [
    "check the claim",
    "document clearly",
    "explain the process",
    "follow policy",
    "for compliance",
    "for safety",
    "help the customer",
    "report accurately",
    "verify the details",
]

IMPERATIVE_PHRASES = [
    "do not",
    "don't",
    "ensure",
    "let us",
    "let's",
    "make sure",
    "must",
    "need to",
    "please",
    "should",
]


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
    required_train_cols = {"label", "text"}
    required_test_cols = {"text"}
    missing_train_cols = sorted(required_train_cols - set(train_df.columns))
    missing_test_cols = sorted(required_test_cols - set(test_df.columns))
    if missing_train_cols:
        raise ValueError(f"Train file is missing required columns: {missing_train_cols}")
    if missing_test_cols:
        raise ValueError(f"Test file is missing required columns: {missing_test_cols}")
    if len(solution_df) != len(test_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")


def extract_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    lowered_text = text.lower()
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
    coordination_term_hits = sum(token_counts.get(term, 0) for term in COORDINATION_TERMS)
    secrecy_term_hits = sum(token_counts.get(term, 0) for term in SECRECY_TERMS)
    manipulation_term_hits = sum(token_counts.get(term, 0) for term in MANIPULATION_TERMS)
    benefit_term_hits = sum(token_counts.get(term, 0) for term in BENEFIT_TERMS)
    benign_term_hits = sum(token_counts.get(term, 0) for term in BENIGN_TERMS)
    covert_phrase_hits = sum(1 for phrase in COVERT_PHRASES if phrase in lowered_text)
    benign_phrase_hits = sum(1 for phrase in BENIGN_PHRASES if phrase in lowered_text)
    imperative_phrase_hits = sum(1 for phrase in IMPERATIVE_PHRASES if phrase in lowered_text)

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
        "coordination_term_hits": float(coordination_term_hits),
        "secrecy_term_hits": float(secrecy_term_hits),
        "manipulation_term_hits": float(manipulation_term_hits),
        "benefit_term_hits": float(benefit_term_hits),
        "benign_term_hits": float(benign_term_hits),
        "covert_phrase_hits": float(covert_phrase_hits),
        "benign_phrase_hits": float(benign_phrase_hits),
        "imperative_phrase_hits": float(imperative_phrase_hits),
        "risk_keyword_density": float(
            (coordination_term_hits + secrecy_term_hits + manipulation_term_hits + benefit_term_hits)
            / safe_word_count
        ),
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


def semantic_signal_scores(row: pd.Series) -> tuple[int, int]:
    risk_score = 0
    benign_score = 0

    if row["covert_phrase_hits"] > 0:
        risk_score += 3
    if row["secrecy_term_hits"] > 0:
        risk_score += 1
    if row["manipulation_term_hits"] > 0:
        risk_score += 2
    if row["coordination_term_hits"] > 0:
        risk_score += 1
    if row["benefit_term_hits"] > 0:
        risk_score += 1
    if row["imperative_phrase_hits"] > 0:
        risk_score += 1
    if row["coordination_term_hits"] > 0 and row["benefit_term_hits"] > 0:
        risk_score += 2
    if row["coordination_term_hits"] > 0 and row["secrecy_term_hits"] > 0:
        risk_score += 2
    if row["imperative_phrase_hits"] > 0 and row["manipulation_term_hits"] > 0:
        risk_score += 1
    if row["risk_keyword_density"] >= 0.08 and row["word_count"] >= 12:
        risk_score += 1

    if row["benign_phrase_hits"] > 0:
        benign_score += 2
    if row["benign_term_hits"] >= 2:
        benign_score += 1
    if row["benign_term_hits"] >= 4:
        benign_score += 1

    return int(risk_score), int(benign_score)


def semantic_trust_adjustment(
    label: int,
    abnormality_score: int,
    risk_score: int,
    benign_score: int,
    row: pd.Series,
) -> tuple[float, str, bool]:
    if label == 1:
        if risk_score >= 4 and abnormality_score <= 1:
            return 1.35, "hidden_risk_TRUE", False
        if risk_score >= 2 and benign_score == 0 and abnormality_score <= 2:
            return 1.15, "supported_TRUE", False
        if benign_score >= 3 and risk_score == 0 and abnormality_score == 0:
            return 0.65, "possible_noisy_TRUE", True
        if benign_score >= 2 and risk_score == 0 and abnormality_score <= 1 and row["word_count"] <= 24:
            return 0.80, "weakly_supported_TRUE", True
        return 1.0, "neutral_TRUE", False

    if risk_score >= 4 and benign_score == 0:
        return 0.60, "possible_noisy_FALSE", True
    if risk_score >= 2 and abnormality_score <= 1:
        return 0.75, "suspicious_FALSE", True
    if benign_score >= 2 and abnormality_score <= 1:
        return 1.10, "strong_benign_FALSE", False
    return 1.0, "neutral_FALSE", False


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


def build_weight_frame(
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for label, (_, feature_row) in zip(labels, feature_df.iterrows(), strict=True):
        score = abnormal_score(feature_row, thresholds)
        semantic_risk_score, semantic_benign_score = semantic_signal_scores(feature_row)
        base_raw_weight, hard_category = assign_sample_weight(int(label), score)
        semantic_multiplier, semantic_category, likely_label_issue = semantic_trust_adjustment(
            label=int(label),
            abnormality_score=score,
            risk_score=semantic_risk_score,
            benign_score=semantic_benign_score,
            row=feature_row,
        )
        raw_weight = base_raw_weight * semantic_multiplier
        rows.append(
            {
                "abnormal_score": score,
                "base_raw_sample_weight": base_raw_weight,
                "semantic_multiplier": semantic_multiplier,
                "semantic_risk_score": semantic_risk_score,
                "semantic_benign_score": semantic_benign_score,
                "semantic_category": semantic_category,
                "likely_label_issue": bool(likely_label_issue),
                "raw_sample_weight": raw_weight,
                "hard_category": hard_category,
            }
        )
    return pd.DataFrame(rows)


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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            if has_labels and labels is not None:
                losses = F.cross_entropy(outputs.logits, labels, reduction="none")
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                total_loss += float(losses.mean().item()) * batch_size
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
    return AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        eps=adam_eps,
    )


def write_weight_summary(
    train_weight_df: pd.DataFrame,
    y_train: np.ndarray,
    train_weights: np.ndarray,
) -> None:
    summary_df = train_weight_df.copy()
    summary_df["label"] = decode_labels(y_train)
    summary_df["normalized_sample_weight"] = train_weights
    grouped = (
        summary_df.groupby(["label", "hard_category", "semantic_category", "likely_label_issue"])
        .agg(
            rows=("semantic_category", "size"),
            mean_abnormal_score=("abnormal_score", "mean"),
            mean_semantic_risk_score=("semantic_risk_score", "mean"),
            mean_semantic_benign_score=("semantic_benign_score", "mean"),
            mean_base_raw_weight=("base_raw_sample_weight", "mean"),
            mean_semantic_multiplier=("semantic_multiplier", "mean"),
            mean_raw_weight=("raw_sample_weight", "mean"),
            mean_normalized_weight=("normalized_sample_weight", "mean"),
        )
        .reset_index()
        .sort_values(["label", "hard_category", "semantic_category"])
    )
    grouped.to_csv(WEIGHT_SUMMARY_PATH, index=False)


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Semantic-Clean Hard-Weighted Training ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    configure_cuda_runtime()
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(
        "Loss: per-row CrossEntropyLoss weighted by hard-example category, "
        "then adjusted by semantic label-trust rules"
    )

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

    print("\n=== Preparing text, labels, and hard weights ===")
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

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")

    train_features = compute_feature_frame(train_texts, "Computing train split hard features")
    val_features = compute_feature_frame(val_texts, "Computing validation hard features")
    test_features = compute_feature_frame(test_texts, "Computing test hard features")
    thresholds = fit_abnormal_thresholds(train_features)
    FEATURE_THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Saved hard-feature thresholds: {FEATURE_THRESHOLDS_PATH}")
    SEMANTIC_RULES_PATH.write_text(
        json.dumps(
            {
                "coordination_terms": sorted(COORDINATION_TERMS),
                "secrecy_terms": sorted(SECRECY_TERMS),
                "manipulation_terms": sorted(MANIPULATION_TERMS),
                "benefit_terms": sorted(BENEFIT_TERMS),
                "benign_terms": sorted(BENIGN_TERMS),
                "covert_phrases": COVERT_PHRASES,
                "benign_phrases": BENIGN_PHRASES,
                "imperative_phrases": IMPERATIVE_PHRASES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved semantic rules: {SEMANTIC_RULES_PATH}")

    train_weight_df = build_weight_frame(y_train, train_features, thresholds)
    val_weight_df = build_weight_frame(y_val, val_features, thresholds)
    raw_train_weights = train_weight_df["raw_sample_weight"].to_numpy(dtype=np.float32)
    train_weights = raw_train_weights / max(float(raw_train_weights.mean()), 1e-12)
    write_weight_summary(train_weight_df, y_train, train_weights)
    print(f"Saved weight summary: {WEIGHT_SUMMARY_PATH}")
    print(f"Mean normalized train weight: {train_weights.mean():.6f}")
    likely_issue_count = int(train_weight_df["likely_label_issue"].sum())
    print(f"Potentially contradictory train labels downweighted: {likely_issue_count}")

    train_diagnostics_df = pd.DataFrame(
        {
            "text": train_texts,
            "label": decode_labels(y_train),
            "abnormal_score": train_weight_df["abnormal_score"].to_numpy(),
            "semantic_risk_score": train_weight_df["semantic_risk_score"].to_numpy(),
            "semantic_benign_score": train_weight_df["semantic_benign_score"].to_numpy(),
            "hard_category": train_weight_df["hard_category"].to_numpy(),
            "semantic_category": train_weight_df["semantic_category"].to_numpy(),
            "likely_label_issue": train_weight_df["likely_label_issue"].to_numpy(),
            "base_raw_sample_weight": train_weight_df["base_raw_sample_weight"].to_numpy(),
            "semantic_multiplier": train_weight_df["semantic_multiplier"].to_numpy(),
            "raw_sample_weight": train_weight_df["raw_sample_weight"].to_numpy(),
            "normalized_sample_weight": train_weights,
        }
    )
    train_diagnostics_df.to_csv(TRAIN_DIAGNOSTICS_PATH, index=False)
    print(f"Saved train row diagnostics: {TRAIN_DIAGNOSTICS_PATH}")

    print("\n=== Loading tokenizer/model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    print("\n=== Tokenizing ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing train split",
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

    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    train_weight_tensor = torch.tensor(train_weights, dtype=torch.float32)
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_weight_tensor)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = build_optimizer(model=model, learning_rate=LEARNING_RATE, adam_eps=ADAM_EPS)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Effective train batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    print("\n=== Training ===")
    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    epoch_history: List[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_unweighted_loss = 0.0
        running_weight_mean = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels, sample_weights = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            per_row_loss = F.cross_entropy(outputs.logits, labels, reduction="none")
            loss = (per_row_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

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
            running_unweighted_loss += float(per_row_loss.mean().detach().cpu().item()) * batch_size
            running_weight_mean += float(sample_weights.mean().detach().cpu().item()) * batch_size
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

            progress.set_postfix(
                train_loss=f"{(running_loss / max(seen_examples, 1)):.4f}",
                raw_ce=f"{(running_unweighted_loss / max(seen_examples, 1)):.4f}",
                w_mean=f"{(running_weight_mean / max(seen_examples, 1)):.3f}",
                skipped_loss=non_finite_loss_batches,
                skipped_grad=non_finite_grad_steps,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(seen_examples, 1)
        train_unweighted_loss = running_unweighted_loss / max(seen_examples, 1)

        val_loss, val_prob_true, val_true = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{EPOCHS} - validation",
            has_labels=True,
        )
        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_weighted_loss"] = float(train_loss)
        val_metrics_05["train_unweighted_loss"] = float(train_unweighted_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        epoch_history.append(val_metrics_05)

        print(
            f"Epoch {epoch} | train_weighted_loss={train_loss:.6f} "
            f"| train_raw_ce={train_unweighted_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_f1": best_epoch_f1,
                    "model_state_dict": model.state_dict(),
                    "feature_thresholds": thresholds,
                },
                BEST_STATE_DICT_PATH,
            )
            print(f"Saved new best model at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError("Best validation predictions were not captured.")

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
            "abnormal_score": val_weight_df["abnormal_score"].to_numpy(),
            "hard_category": val_weight_df["hard_category"].to_numpy(),
            "semantic_category": val_weight_df["semantic_category"].to_numpy(),
            "semantic_risk_score": val_weight_df["semantic_risk_score"].to_numpy(),
            "semantic_benign_score": val_weight_df["semantic_benign_score"].to_numpy(),
            "semantic_multiplier_if_trained": val_weight_df["semantic_multiplier"].to_numpy(),
            "likely_label_issue_if_trained": val_weight_df["likely_label_issue"].to_numpy(),
            "raw_sample_weight_if_trained": val_weight_df["raw_sample_weight"].to_numpy(),
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
    error_df = error_df[
        [
            "text",
            "true_label",
            "pred_label",
            "pred_prob_TRUE",
            "error_type",
            "confidence",
            "abnormal_score",
            "hard_category",
            "semantic_category",
            "semantic_risk_score",
            "semantic_benign_score",
            "semantic_multiplier_if_trained",
            "likely_label_issue_if_trained",
            "raw_sample_weight_if_trained",
        ]
    ]
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
    test_semantic_rows = []
    for _, feature_row in test_features.iterrows():
        semantic_risk_score, semantic_benign_score = semantic_signal_scores(feature_row)
        test_semantic_rows.append(
            {
                "abnormal_score": abnormal_score(feature_row, thresholds),
                "semantic_risk_score": semantic_risk_score,
                "semantic_benign_score": semantic_benign_score,
            }
        )
    test_semantic_df = pd.DataFrame(test_semantic_rows)

    test_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "pred_prob_TRUE": test_prob_true,
            "abnormal_score": test_semantic_df["abnormal_score"].to_numpy(),
            "semantic_risk_score": test_semantic_df["semantic_risk_score"].to_numpy(),
            "semantic_benign_score": test_semantic_df["semantic_benign_score"].to_numpy(),
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

    fixed_threshold = 0.32
    fixed_test_pred = (test_prob_true >= fixed_threshold).astype(int)
    fixed_test_pred_labels = decode_labels(fixed_test_pred)
    if solution_df.columns.tolist() == ["label"]:
        fixed_submission_df = pd.DataFrame({"label": fixed_test_pred_labels})
    else:
        fixed_submission_df = solution_df.copy()
        fixed_submission_df["label"] = fixed_test_pred_labels
    fixed_submission_df = fixed_submission_df[solution_df.columns.tolist()]
    fixed_submission_df.to_csv(SUBMISSION_THR032_PATH, index=False)
    print(f"Saved fixed-threshold submission (0.32): {SUBMISSION_THR032_PATH}")

    metrics_payload = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
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
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
            "loss": "sample_weighted_cross_entropy",
            "weight_strategy": {
                "hard_clean_TRUE": 2.5,
                "medium_clean_TRUE": 1.5,
                "easy_noisy_TRUE": 0.85,
                "hard_messy_FALSE": 2.0,
                "medium_messy_FALSE": 1.4,
                "easy_clean_FALSE": 0.85,
                "normalization": "divide train split raw weights by train split mean weight",
            },
            "semantic_trust_strategy": {
                "TRUE_hidden_risk_multiplier": 1.35,
                "TRUE_supported_multiplier": 1.15,
                "TRUE_possible_noisy_multiplier": 0.65,
                "TRUE_weakly_supported_multiplier": 0.80,
                "FALSE_possible_noisy_multiplier": 0.60,
                "FALSE_suspicious_multiplier": 0.75,
                "FALSE_strong_benign_multiplier": 1.10,
                "likely_label_issue_count_train": likely_issue_count,
                "policy": "downweight rows whose label conflicts with semantic cues; upweight hidden-risk TRUE rows",
            },
            "feature_thresholds": thresholds,
        },
        "split": {
            "train_rows": int(len(train_texts)),
            "val_rows": int(len(val_texts)),
            "test_rows": int(len(test_texts)),
            "stratified": True,
            "random_state": SEED,
        },
        "epoch_metrics_threshold_0_5": epoch_history,
        "best_epoch": {
            "epoch": int(best_epoch),
            "val_f1_threshold_0_5": float(best_epoch_f1),
            "checkpoint_dir": str(BEST_MODEL_DIR),
            "state_dict_path": str(BEST_STATE_DICT_PATH),
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
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "submission_thr0p32": str(SUBMISSION_THR032_PATH),
            "notes": str(NOTES_PATH),
            "weight_summary": str(WEIGHT_SUMMARY_PATH),
            "feature_thresholds": str(FEATURE_THRESHOLDS_PATH),
            "train_diagnostics": str(TRAIN_DIAGNOSTICS_PATH),
            "semantic_rules": str(SEMANTIC_RULES_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_DICT_PATH),
        },
        "comparison_target": {
            "submission": "outputs/roberta_base/roberta_base_submission.csv",
            "public_lb": 0.90909091,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes_text = f"""# RoBERTa Semantic-Clean Hard-Weighted Notes

## Why This Run Exists

The current RoBERTa hard-weighted baseline already captures the easy distribution artifact: noisy TRUE and clean FALSE.
The plateau is mostly clean-looking TRUE and messy-looking FALSE, plus a smaller set of rows whose labels look semantically contradictory.

This script keeps the same `roberta-base` training loop, but changes how much each training row is trusted:

1. Existing hard-example weighting still handles clean TRUE and messy FALSE.
2. A semantic risk score looks for covert coordination, secrecy, manipulation, shared-benefit, and imperative cues.
3. A benign/compliance score looks for normal verification, reporting, policy, and safety language.
4. Rows whose label conflicts strongly with those semantic cues are downweighted instead of deleted.
5. Clean-looking TRUE rows with hidden-risk semantics are upweighted.

This is soft label cleaning, not pseudo-labeling and not rule-based inference.

## Configuration

- Model: `{MODEL_NAME}`
- Max length: `{MAX_LENGTH}`
- Epochs: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- Effective train batch size: `{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Loss: `sample_weighted_cross_entropy`
- Potentially contradictory train labels downweighted: `{likely_issue_count}`
- Hard weights:
  - clean TRUE: `2.5`
  - medium clean TRUE: `1.5`
  - noisy TRUE: `0.85`
  - messy FALSE: `2.0`
  - medium messy FALSE: `1.4`
  - clean FALSE: `0.85`
- Semantic trust multipliers:
  - hidden-risk TRUE: `1.35`
  - semantically supported TRUE: `1.15`
  - possible noisy TRUE: `0.65`
  - weakly supported TRUE: `0.80`
  - possible noisy FALSE: `0.60`
  - suspicious FALSE: `0.75`
  - strong benign FALSE: `1.10`

Weights are normalized by the train split mean raw weight before training.

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Abnormality thresholds fitted on the training split only
- Semantic trust rules applied on the training split only
- Best checkpoint selected by validation F1 at threshold 0.5
- Final threshold tuned over 0.30-0.70 for best validation F1
- Additional fixed-threshold export saved at `0.32`

## Final Validation Metrics

- Threshold: `{final_val_metrics['threshold']:.2f}`
- Accuracy: `{final_val_metrics['accuracy']:.6f}`
- Precision: `{final_val_metrics['precision']:.6f}`
- Recall: `{final_val_metrics['recall']:.6f}`
- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_val_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{final_val_metrics['prediction_distribution']}`

## Comparison Target

Compare against:

- `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`
- the current best hard-weighted GPU threshold-0.32 run

## Outputs

- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_metrics.json`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_val_predictions.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_error_analysis.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_test_probabilities.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_submission.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_submission_thr0p32.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_notes.md`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_weight_summary.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_feature_thresholds.json`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_semantic_rules.json`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_train_row_diagnostics.csv`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/best_model/`
- `outputs/roberta_semantic_clean_hard_weighted_gpu/roberta_semantic_clean_hard_weighted_gpu_best_state_dict.pt`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")
    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
