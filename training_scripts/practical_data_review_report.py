#!/usr/bin/env python3
"""
Create a practical dataset review report in one Markdown file.

Dependency install command:
pip install pandas numpy scikit-learn tqdm

This script does not train a transformer. For the "basic model" section it uses
the existing plain RoBERTa validation predictions from outputs/roberta_base when
available. That keeps this as a fast review script instead of another expensive
training run.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"

SEED = 42
SAMPLE_PER_CLASS = 100
MAX_SNIPPET_CHARS = 360

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_PATH = DATA_DIR / "solution_format.csv"

ROBERTA_VAL_PRED_PATH = OUTPUT_DIR / "roberta_base" / "roberta_base_val_predictions.csv"
ROBERTA_METRICS_PATH = OUTPUT_DIR / "roberta_base" / "roberta_base_metrics.json"

REPORT_PATH = OUTPUT_DIR / "practical_data_review.md"

WORD_RE = re.compile(r"[A-Za-z0-9_']+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[.!?]+|\n+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
BRACKET_CHARS = set("()[]{}<>")
QUOTE_CHARS = set("\"'`“”‘’")
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


FEATURE_COLUMNS = [
    "char_length",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "max_word_length",
    "unique_word_ratio",
    "stopword_ratio",
    "punctuation_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "special_char_ratio",
    "non_ascii_ratio",
    "newline_count",
    "comma_count",
    "quote_count",
    "bracket_count",
    "url_count",
    "repeated_token_count",
    "long_token_count",
    "weird_token_ratio",
    "repeated_char_count",
]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = count_lines(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    clean = normalize_text(text).replace("\n", " / ")
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


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
    repeated_token_count = sum(count - 1 for count in token_counts.values() if count > 1)
    stopword_count = sum(1 for word in words if word in STOPWORDS)
    sentence_parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]

    punctuation_count = sum(1 for char in chars if char in PUNCT_CHARS)
    digit_count = sum(1 for char in chars if char.isdigit())
    alpha_count = sum(1 for char in chars if char.isalpha())
    uppercase_count = sum(1 for char in chars if char.isupper())
    special_count = sum(1 for char in chars if not char.isalnum() and not char.isspace())
    non_ascii_count = sum(1 for char in chars if ord(char) > 127)

    comma_count = text.count(",")
    quote_count = sum(1 for char in chars if char in QUOTE_CHARS)
    bracket_count = sum(1 for char in chars if char in BRACKET_CHARS)
    repeated_char_count = len(REPEATED_CHAR_RE.findall(text))
    long_token_count = sum(1 for token in raw_tokens if len(token) >= 18)

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
        "sentence_count": float(len(sentence_parts)),
        "avg_word_length": float(np.mean(word_lengths) if word_lengths else 0.0),
        "max_word_length": float(max(word_lengths) if word_lengths else 0.0),
        "unique_word_ratio": float(len(set(words)) / safe_word_count),
        "stopword_ratio": float(stopword_count / safe_word_count),
        "punctuation_ratio": float(punctuation_count / safe_char_count),
        "digit_ratio": float(digit_count / safe_char_count),
        "uppercase_ratio": float(uppercase_count / max(alpha_count, 1)),
        "special_char_ratio": float(special_count / safe_char_count),
        "non_ascii_ratio": float(non_ascii_count / safe_char_count),
        "newline_count": float(text.count("\n")),
        "comma_count": float(comma_count),
        "quote_count": float(quote_count),
        "bracket_count": float(bracket_count),
        "url_count": float(len(URL_RE.findall(text))),
        "repeated_token_count": float(repeated_token_count),
        "long_token_count": float(long_token_count),
        "weird_token_ratio": float(weird_tokens / max(len(raw_tokens), 1)),
        "repeated_char_count": float(repeated_char_count),
    }


def compute_feature_frame(texts: list[str], desc: str) -> pd.DataFrame:
    return pd.DataFrame([extract_features(text) for text in tqdm(texts, desc=desc, unit="rows")])


def feature_summary_by_label(df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df[["label"]].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    rows = []
    for label in ["FALSE", "TRUE"]:
        subset = combined[combined["label"] == label]
        for feature in FEATURE_COLUMNS:
            values = subset[feature].to_numpy(dtype=float)
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def feature_effect_sizes(df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df[["label"]].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    false_df = combined[combined["label"] == "FALSE"]
    true_df = combined[combined["label"] == "TRUE"]
    rows = []
    for feature in FEATURE_COLUMNS:
        false_values = false_df[feature].to_numpy(dtype=float)
        true_values = true_df[feature].to_numpy(dtype=float)
        false_mean = float(np.mean(false_values))
        true_mean = float(np.mean(true_values))
        pooled_std = math.sqrt((float(np.var(false_values)) + float(np.var(true_values))) / 2.0)
        effect = (true_mean - false_mean) / pooled_std if pooled_std > 1e-12 else 0.0
        rows.append(
            {
                "feature": feature,
                "false_mean": false_mean,
                "true_mean": true_mean,
                "difference_true_minus_false": true_mean - false_mean,
                "effect_size": effect,
                "direction": "higher_TRUE" if effect > 0 else "higher_FALSE",
            }
        )
    return pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False)


def make_thresholds(feature_df: pd.DataFrame) -> dict[str, float]:
    return {
        "char_length_p90": float(feature_df["char_length"].quantile(0.90)),
        "word_count_p90": float(feature_df["word_count"].quantile(0.90)),
        "punctuation_ratio_p90": float(feature_df["punctuation_ratio"].quantile(0.90)),
        "weird_token_ratio_p90": float(feature_df["weird_token_ratio"].quantile(0.90)),
        "repeated_token_count_p90": float(feature_df["repeated_token_count"].quantile(0.90)),
        "max_word_length_p95": float(feature_df["max_word_length"].quantile(0.95)),
        "stopword_ratio_p10": float(feature_df["stopword_ratio"].quantile(0.10)),
        "special_char_ratio_p90": float(feature_df["special_char_ratio"].quantile(0.90)),
    }


def tag_text(features: pd.Series, thresholds: dict[str, float]) -> list[str]:
    tags = []
    if features["char_length"] >= thresholds["char_length_p90"] or features["word_count"] >= thresholds["word_count_p90"]:
        tags.append("long")
    if features["punctuation_ratio"] >= thresholds["punctuation_ratio_p90"]:
        tags.append("punctuation-heavy")
    if features["weird_token_ratio"] >= thresholds["weird_token_ratio_p90"] and features["weird_token_ratio"] > 0:
        tags.append("weird-token")
    if features["repeated_token_count"] >= thresholds["repeated_token_count_p90"] and features["repeated_token_count"] > 0:
        tags.append("repetitive")
    if features["max_word_length"] >= thresholds["max_word_length_p95"]:
        tags.append("long-token")
    if features["stopword_ratio"] <= thresholds["stopword_ratio_p10"] and features["word_count"] >= 8:
        tags.append("low-stopword")
    if features["special_char_ratio"] >= thresholds["special_char_ratio_p90"]:
        tags.append("formatting-noise")
    if features["newline_count"] >= 2:
        tags.append("multi-line")
    if features["url_count"] > 0:
        tags.append("url")
    if not tags:
        tags.append("plain/coherent-looking")
    return tags


def abnormal_score(features: pd.Series, thresholds: dict[str, float]) -> int:
    tags = tag_text(features, thresholds)
    return sum(tag != "plain/coherent-looking" for tag in tags)


def sample_review_rows(train_df: pd.DataFrame, feature_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    combined = pd.concat([train_df[["text", "label"]].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    samples = []
    for label in ["TRUE", "FALSE"]:
        subset = combined[combined["label"] == label]
        n = min(SAMPLE_PER_CLASS, len(subset))
        samples.append(subset.sample(n=n, random_state=SEED))
    sample_df = pd.concat(samples, ignore_index=True)
    sample_df["tags"] = [", ".join(tag_text(row, thresholds)) for _, row in sample_df.iterrows()]
    sample_df["abnormal_score"] = [abnormal_score(row, thresholds) for _, row in sample_df.iterrows()]
    sample_df["snippet"] = [snippet(text) for text in sample_df["text"]]
    return sample_df


def normalize_boolish_label_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_label)


def load_roberta_metrics_threshold() -> float:
    if not ROBERTA_METRICS_PATH.exists():
        return 0.5
    metrics = json.loads(ROBERTA_METRICS_PATH.read_text(encoding="utf-8"))
    return float(metrics.get("threshold_tuning", {}).get("selected_threshold", 0.5))


def analyze_roberta_mistakes(thresholds: dict[str, float]) -> tuple[str, pd.DataFrame | None]:
    if not ROBERTA_VAL_PRED_PATH.exists():
        return (
            "Existing RoBERTa validation predictions were not found, so this section could not analyze model mistakes.",
            None,
        )

    val_df = pd.read_csv(ROBERTA_VAL_PRED_PATH)
    required_cols = {"text", "true_label", "pred_label", "pred_prob_TRUE"}
    missing = sorted(required_cols - set(val_df.columns))
    if missing:
        return f"RoBERTa validation predictions are missing required columns: `{missing}`.", None

    val_df = val_df.copy()
    val_df["text"] = [normalize_text(value) for value in val_df["text"]]
    val_df["true_label_norm"] = normalize_boolish_label_series(val_df["true_label"])
    val_df["pred_label_norm"] = normalize_boolish_label_series(val_df["pred_label"])
    y_true = (val_df["true_label_norm"] == "TRUE").astype(int).to_numpy()
    y_pred = (val_df["pred_label_norm"] == "TRUE").astype(int).to_numpy()
    prob_true = pd.to_numeric(val_df["pred_prob_TRUE"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    val_features = compute_feature_frame(val_df["text"].tolist(), "Computing RoBERTa validation mistake features")
    val_df = pd.concat([val_df, val_features], axis=1)
    val_df["error_type"] = "correct"
    val_df.loc[(val_df["true_label_norm"] == "FALSE") & (val_df["pred_label_norm"] == "TRUE"), "error_type"] = "false_positive"
    val_df.loc[(val_df["true_label_norm"] == "TRUE") & (val_df["pred_label_norm"] == "FALSE"), "error_type"] = "false_negative"
    selected_threshold = load_roberta_metrics_threshold()
    val_df["threshold_margin"] = np.abs(val_df["pred_prob_TRUE"].astype(float) - selected_threshold)
    val_df["abnormal_score"] = [abnormal_score(row, thresholds) for _, row in val_df.iterrows()]
    val_df["tags"] = [", ".join(tag_text(row, thresholds)) for _, row in val_df.iterrows()]

    errors = val_df[val_df["error_type"] != "correct"].copy()
    if errors.empty:
        summary = "RoBERTa validation predictions have no errors in the saved file."
        return summary, errors

    def classify_error(row: pd.Series) -> str:
        if row["threshold_margin"] <= 0.10:
            return "borderline/probability"
        if row["abnormal_score"] >= 3:
            return "stylistic/noise-driven"
        if row["error_type"] == "false_negative" and row["abnormal_score"] <= 1:
            return "semantic/subtle"
        if row["error_type"] == "false_positive" and row["abnormal_score"] <= 1:
            return "semantic/label-noise-or-overread"
        return "mixed"

    errors["failure_bucket"] = [classify_error(row) for _, row in errors.iterrows()]
    bucket_counts = errors["failure_bucket"].value_counts().rename_axis("failure_bucket").reset_index(name="count")
    error_type_counts = errors["error_type"].value_counts().rename_axis("error_type").reset_index(name="count")

    lines = [
        "### Existing Basic RoBERTa Model Review",
        "",
        "This uses the saved `outputs/roberta_base/roberta_base_val_predictions.csv` artifact. It is the plain RoBERTa classifier baseline, not a new training run.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Precision | {metrics['precision']:.6f} |",
        f"| Recall | {metrics['recall']:.6f} |",
        f"| F1 | {metrics['f1']:.6f} |",
        f"| ROC AUC | {metrics['roc_auc']:.6f} |",
        f"| Confusion matrix [[TN, FP], [FN, TP]] | `{metrics['confusion_matrix']}` |",
        f"| Selected threshold from metrics | {selected_threshold:.2f} |",
        "",
        "#### Error Counts",
        "",
        error_type_counts.to_markdown(index=False),
        "",
        "#### Heuristic Failure Buckets",
        "",
        bucket_counts.to_markdown(index=False),
        "",
        "Interpretation: `semantic/subtle` means the text does not look obviously noisy by simple features, so the miss is probably about meaning, label rule, or subtle abnormality. `stylistic/noise-driven` means the text has strong surface abnormality signals.",
    ]

    return "\n".join(lines), errors


def top_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if df.empty:
        return "_None._"
    return df[columns].head(n).to_markdown(index=False)


def format_sample_appendix(sample_df: pd.DataFrame) -> str:
    lines = []
    for label in ["TRUE", "FALSE"]:
        subset = sample_df[sample_df["label"] == label].reset_index(drop=True)
        lines.append(f"### {label} Sample Snippets")
        lines.append("")
        for i, row in subset.iterrows():
            lines.append(f"{i + 1}. `{row['tags']}` score={int(row['abnormal_score'])}: {row['snippet']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Practical Data Review Report ===")
    print("This script writes one Markdown report and does not train a model.")

    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_PATH)
    train_df["text"] = [normalize_text(value) for value in tqdm(train_df["text"], desc="Normalizing train text")]
    test_df["text"] = [normalize_text(value) for value in tqdm(test_df["text"], desc="Normalizing test text")]
    train_df["label"] = [normalize_label(value) for value in tqdm(train_df["label"], desc="Normalizing labels")]

    train_features = compute_feature_frame(train_df["text"].tolist(), "Computing train dumb-signal features")
    thresholds = make_thresholds(train_features)
    summary_df = feature_summary_by_label(train_df, train_features)
    effects_df = feature_effect_sizes(train_df, train_features)
    sample_df = sample_review_rows(train_df, train_features, thresholds)

    sample_tag_counts = (
        sample_df.assign(tag=sample_df["tags"].str.split(", "))
        .explode("tag")
        .groupby(["label", "tag"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[True, False])
    )

    true_higher = effects_df[effects_df["effect_size"] > 0].head(8)
    false_higher = effects_df[effects_df["effect_size"] < 0].head(8)

    roberta_section, errors_df = analyze_roberta_mistakes(thresholds)

    report_lines = [
        "# Practical Data Review",
        "",
        "This report follows the simple workflow: read samples, inspect dumb signals, use the existing plain RoBERTa baseline, and examine mistakes.",
        "",
        "## Dataset Summary",
        "",
        f"- Train rows: `{len(train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Solution rows: `{len(solution_df)}`",
        f"- Sample reviewed per class: `{SAMPLE_PER_CLASS}`",
        "",
        train_df["label"].value_counts().rename_axis("label").reset_index(name="count").to_markdown(index=False),
        "",
        "## What Differentiates TRUE vs FALSE?",
        "",
        "Based on simple feature summaries and the deterministic 100/100 sample review:",
        "",
        "- TRUE tends to be longer and more likely to contain abnormal wording or generation artifacts.",
        "- TRUE often has stronger weird-token, long-token, repetition, or formatting signals.",
        "- FALSE tends to be more coherent/task-like, but some FALSE rows are also weird, so style alone is not enough.",
        "- The hard cases are not just toxicity. Many look like subtle low-fluency, corrupted, or semantically off text.",
        "",
        "## Dumb Signal Effect Sizes",
        "",
        "Positive effect size means higher in TRUE. Negative means higher in FALSE.",
        "",
        effects_df.head(18).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Feature Means By Label",
        "",
        summary_df.pivot(index="feature", columns="label", values="mean")
        .reset_index()
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 100 TRUE / 100 FALSE Sample Tag Counts",
        "",
        sample_tag_counts.to_markdown(index=False),
        "",
        "## Stronger TRUE-Leaning Dumb Signals",
        "",
        true_higher[["feature", "false_mean", "true_mean", "effect_size"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Stronger FALSE-Leaning Dumb Signals",
        "",
        false_higher[["feature", "false_mean", "true_mean", "effect_size"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        roberta_section,
        "",
    ]

    if errors_df is not None and not errors_df.empty:
        errors_for_report = errors_df.copy()
        errors_for_report["snippet"] = [snippet(text) for text in errors_for_report["text"]]
        false_pos = errors_for_report[errors_for_report["error_type"] == "false_positive"].sort_values(
            "pred_prob_TRUE",
            ascending=False,
        )
        false_neg = errors_for_report[errors_for_report["error_type"] == "false_negative"].sort_values(
            "pred_prob_TRUE",
            ascending=True,
        )
        report_lines.extend(
            [
                "## RoBERTa False Positive Examples",
                "",
                top_table(
                    false_pos,
                    ["snippet", "pred_prob_TRUE", "failure_bucket", "tags", "abnormal_score"],
                    n=12,
                ),
                "",
                "## RoBERTa False Negative Examples",
                "",
                top_table(
                    false_neg,
                    ["snippet", "pred_prob_TRUE", "failure_bucket", "tags", "abnormal_score"],
                    n=12,
                ),
                "",
            ]
        )

    representative_true = sample_df[sample_df["label"] == "TRUE"].sort_values("abnormal_score", ascending=False)
    representative_false = sample_df[sample_df["label"] == "FALSE"].sort_values("abnormal_score", ascending=True)
    report_lines.extend(
        [
            "## Representative Sample Snippets",
            "",
            "### TRUE examples with strongest surface abnormality in the 100-row sample",
            "",
            representative_true[["snippet", "tags", "abnormal_score"]].head(12).to_markdown(index=False),
            "",
            "### FALSE examples with most plain/coherent-looking surface in the 100-row sample",
            "",
            representative_false[["snippet", "tags", "abnormal_score"]].head(12).to_markdown(index=False),
            "",
            "## Practical Conclusion",
            "",
            "- Dumb signals are real, but they overlap heavily. They explain part of TRUE/FALSE, not the whole task.",
            "- Existing RoBERTa errors should be read manually, especially false negatives with low abnormality scores.",
            "- If failures are mostly `semantic/subtle`, model choice or representation learning matters more than punctuation/length features.",
            "- If failures are mostly `stylistic/noise-driven`, add targeted preprocessing/features or use contrastive learning carefully.",
            "- Do not assume a larger model or weighted loss fixes this. Prior experiments already showed those can hurt.",
            "",
            "## Appendix: 100 TRUE / 100 FALSE Review Snippets",
            "",
            "Snippets are truncated to keep this local report readable.",
            "",
            format_sample_appendix(sample_df),
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
