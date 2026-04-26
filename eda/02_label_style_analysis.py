#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm

from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "eda_label_style"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
ROBERTA_VAL_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv"
ROBERTA_ERROR_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_error_analysis.csv"

FEATURE_SUMMARY_PATH = OUTPUT_DIR / "feature_summary_by_label.csv"
FEATURE_EFFECTS_PATH = OUTPUT_DIR / "feature_effect_sizes.csv"
VAL_OUTCOME_SUMMARY_PATH = OUTPUT_DIR / "roberta_val_outcome_feature_summary.csv"
ROBERTA_ERRORS_PATH = OUTPUT_DIR / "roberta_errors_with_features.csv"
WORD_TRUE_PATH = OUTPUT_DIR / "top_true_word_indicators.csv"
WORD_FALSE_PATH = OUTPUT_DIR / "top_false_word_indicators.csv"
CHAR_TRUE_PATH = OUTPUT_DIR / "top_true_char_indicators.csv"
CHAR_FALSE_PATH = OUTPUT_DIR / "top_false_char_indicators.csv"
EXTREME_EXAMPLES_PATH = OUTPUT_DIR / "feature_extreme_examples.csv"
LB_PATTERN_PATH = OUTPUT_DIR / "lb_submission_true_count_pattern.csv"
REPORT_PATH = OUTPUT_DIR / "label_style_report.md"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

SPAM_TERMS = {
    "affiliate",
    "buy",
    "click",
    "coupon",
    "deal",
    "free",
    "offer",
    "order",
    "product",
    "products",
    "promo",
    "sale",
    "shopping",
    "subscribe",
}

RISK_TERMS = {
    "attack",
    "combat",
    "danger",
    "exploit",
    "fight",
    "harm",
    "hate",
    "kill",
    "threat",
    "weapon",
}

PROMPT_PATTERN = re.compile(
    r"based on the passage above|answer:|question:|could you|can you|what is|how did|why did|paraphrase",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|io|gov|edu)\b", re.IGNORECASE)
CODE_PATTERN = re.compile(
    r"\{|\}|\[|\]|=>|==|</|/>|function\b|var\b|const\b|let\b|def\b|class\b|model\(|user_story|setvar",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+|\n+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label: {value}")


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def count_adjacent_repeats(tokens: list[str]) -> int:
    return sum(1 for prev, cur in zip(tokens, tokens[1:]) if prev == cur)


def text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    chars = len(text)
    tokens = WORD_PATTERN.findall(text)
    lower_tokens = [token.lower() for token in tokens]
    token_count = len(tokens)
    alpha_count = sum(ch.isalpha() for ch in text)
    uppercase_count = sum(ch.isupper() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    punctuation_count = sum(ch in string.punctuation for ch in text)
    whitespace_count = sum(ch.isspace() for ch in text)
    non_ascii_count = sum(ord(ch) > 127 for ch in text)
    alnum_count = sum(ch.isalnum() for ch in text)
    sentence_parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    sentence_count = len(sentence_parts)
    token_lengths = [len(token) for token in tokens]
    unique_word_ratio = safe_ratio(len(set(lower_tokens)), token_count)
    stopword_ratio = safe_ratio(sum(token in STOPWORDS for token in lower_tokens), token_count)
    spam_keyword_count = sum(token in SPAM_TERMS for token in lower_tokens)
    risk_keyword_count = sum(token in RISK_TERMS for token in lower_tokens)

    weird_tokens = 0
    for token in tokens:
        has_digit = any(ch.isdigit() for ch in token)
        has_alpha = any(ch.isalpha() for ch in token)
        has_non_ascii = any(ord(ch) > 127 for ch in token)
        has_repeated_chars = bool(REPEATED_CHAR_PATTERN.search(token))
        has_internal_punct = bool(re.search(r"[_/\\-]", token))
        if (has_digit and has_alpha) or has_non_ascii or has_repeated_chars or has_internal_punct:
            weird_tokens += 1

    return {
        "char_length": float(chars),
        "word_count": float(token_count),
        "avg_word_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "max_word_length": float(max(token_lengths)) if token_lengths else 0.0,
        "unique_word_ratio": unique_word_ratio,
        "stopword_ratio": stopword_ratio,
        "digit_ratio": safe_ratio(digit_count, chars),
        "uppercase_ratio": safe_ratio(uppercase_count, alpha_count),
        "punctuation_ratio": safe_ratio(punctuation_count, chars),
        "whitespace_ratio": safe_ratio(whitespace_count, chars),
        "non_ascii_ratio": safe_ratio(non_ascii_count, chars),
        "non_alnum_ratio": safe_ratio(chars - alnum_count, chars),
        "newline_count": float(text.count("\n")),
        "url_domain_count": float(len(URL_PATTERN.findall(text))),
        "code_symbol_count": float(len(CODE_PATTERN.findall(text))),
        "decode_error_flag": float("[DECODE ERROR]" in text),
        "prompt_template_flag": float(bool(PROMPT_PATTERN.search(text))),
        "spam_keyword_count": float(spam_keyword_count),
        "risk_keyword_count": float(risk_keyword_count),
        "repeated_token_count": float(count_adjacent_repeats(lower_tokens)),
        "very_long_token_count": float(sum(length >= 20 for length in token_lengths)),
        "weird_token_ratio": safe_ratio(weird_tokens, token_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_word_count": safe_ratio(token_count, sentence_count),
    }


def make_feature_frame(texts: Iterable[str], desc: str) -> pd.DataFrame:
    rows = []
    for text in tqdm(list(texts), desc=desc, unit="rows"):
        rows.append(text_features(text))
    return pd.DataFrame(rows)


def summarize_by_label(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = df.groupby("label_norm")[feature_cols].agg(["mean", "median", "std", "min", "max"]).round(6)
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    records = []
    true_df = df[df["label_norm"] == "TRUE"]
    false_df = df[df["label_norm"] == "FALSE"]
    for feature in feature_cols:
        true_mean = true_df[feature].mean()
        false_mean = false_df[feature].mean()
        true_std = true_df[feature].std(ddof=1)
        false_std = false_df[feature].std(ddof=1)
        pooled = math.sqrt((true_std**2 + false_std**2) / 2) if not pd.isna(true_std + false_std) else 0.0
        records.append(
            {
                "feature": feature,
                "true_mean": true_mean,
                "false_mean": false_mean,
                "diff_true_minus_false": true_mean - false_mean,
                "ratio_true_over_false": safe_ratio(true_mean, false_mean),
                "standardized_diff": safe_ratio(true_mean - false_mean, pooled),
            }
        )
    effects = pd.DataFrame(records).sort_values("standardized_diff", key=lambda s: s.abs(), ascending=False)
    return summary, effects


def log_odds_indicators(
    texts: pd.Series,
    labels: pd.Series,
    vectorizer: CountVectorizer,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = vectorizer.fit_transform(texts)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    y = labels.to_numpy()
    true_counts = np.asarray(x[y == "TRUE"].sum(axis=0)).ravel()
    false_counts = np.asarray(x[y == "FALSE"].sum(axis=0)).ravel()
    alpha = 1.0
    true_total = true_counts.sum() + alpha * len(feature_names)
    false_total = false_counts.sum() + alpha * len(feature_names)
    log_odds = np.log((true_counts + alpha) / true_total) - np.log((false_counts + alpha) / false_total)
    out = pd.DataFrame(
        {
            "ngram": feature_names,
            "true_count": true_counts.astype(int),
            "false_count": false_counts.astype(int),
            "log_odds_true_minus_false": log_odds,
        }
    )
    return (
        out.sort_values("log_odds_true_minus_false", ascending=False).head(top_n),
        out.sort_values("log_odds_true_minus_false", ascending=True).head(top_n),
    )


def add_roberta_outcomes(val_df: pd.DataFrame) -> pd.DataFrame:
    val_df = val_df.copy()
    val_df["true_label_norm"] = val_df["true_label"].map(normalize_label)
    val_df["pred_label_norm"] = val_df["pred_label"].map(normalize_label)
    val_df["outcome"] = np.select(
        [
            (val_df["true_label_norm"] == "TRUE") & (val_df["pred_label_norm"] == "TRUE"),
            (val_df["true_label_norm"] == "FALSE") & (val_df["pred_label_norm"] == "FALSE"),
            (val_df["true_label_norm"] == "FALSE") & (val_df["pred_label_norm"] == "TRUE"),
            (val_df["true_label_norm"] == "TRUE") & (val_df["pred_label_norm"] == "FALSE"),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )
    return val_df


def build_extreme_examples(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    selected = [
        "char_length",
        "non_ascii_ratio",
        "code_symbol_count",
        "weird_token_ratio",
        "repeated_token_count",
        "spam_keyword_count",
        "prompt_template_flag",
        "decode_error_flag",
    ]
    rows = []
    for label in ["TRUE", "FALSE"]:
        label_df = df[df["label_norm"] == label]
        for feature in selected:
            if feature not in feature_cols:
                continue
            subset = label_df.sort_values(feature, ascending=False).head(20)
            for rank, row in enumerate(subset.itertuples(), start=1):
                rows.append(
                    {
                        "label": label,
                        "feature": feature,
                        "rank": rank,
                        "feature_value": getattr(row, feature),
                        "text": row.text,
                    }
                )
    return pd.DataFrame(rows)


def build_lb_pattern() -> pd.DataFrame:
    known = [
        ("roberta_base", "outputs/roberta_base/roberta_base_submission.csv", 0.90909091),
        ("verification_regenerated", "outputs/roberta_base/verification_regenerated_submission.csv", 0.90909091),
        ("roberta_base_cv", "outputs/roberta_base_cv/roberta_base_cv_submission.csv", 0.87591241),
        ("roberta_base_cv_v2", "outputs/roberta_base_cv_v2/roberta_base_cv_v2_submission.csv", 0.83054004),
        ("modernbert_base", "outputs/modernbert_base/modernbert_base_submission.csv", 0.83012821),
        ("modernbert_lora", "outputs/modernbert_lora/modernbert_lora_submission.csv", 0.87145242),
        ("modernbert_lora_v2", "outputs/modernbert_lora_v2/modernbert_lora_v2_submission.csv", 0.88628763),
        ("roberta_pseudolabel", "outputs/roberta_pseudolabel/roberta_pseudolabel_submission.csv", 0.79125249),
        ("blend_r095_m005_thr032", "outputs/blends/roberta_modernbert_blend_r0p95_m0p05_thr0p32.csv", 0.91919192),
        ("blend_r090_m010_thr034", "outputs/blends/roberta_modernbert_blend_r0p90_m0p10_thr0p34.csv", 0.90940767),
        ("blend_r090_m010_thr036", "outputs/blends/roberta_modernbert_blend_r0p90_m0p10_thr0p36.csv", 0.91068301),
        ("ensemble_or", "outputs/blends/roberta_modernbert_ensemble_or.csv", 0.85670732),
        ("ensemble_and", "outputs/blends/roberta_modernbert_ensemble_and.csv", 0.88148148),
        ("tfidf_features_cv", "outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_submission.csv", 0.64610866),
        ("tfidf_features", "outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_submission.csv", 0.62416107),
        ("minilm_cv", "outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_submission.csv", 0.51649928),
    ]
    rows = []
    for name, rel_path, score in known:
        path = ROOT_DIR / rel_path
        if not path.exists():
            continue
        labels = pd.read_csv(path)["label"].map(normalize_label)
        rows.append(
            {
                "submission": name,
                "path": rel_path,
                "lb_score": score,
                "true_count": int((labels == "TRUE").sum()),
                "false_count": int((labels == "FALSE").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("lb_score", ascending=False)


def write_report(
    train_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    lb_df: pd.DataFrame,
) -> None:
    label_counts = train_df["label_norm"].value_counts().to_dict()
    top_true_features = effects_df.sort_values("standardized_diff", ascending=False).head(8)
    top_false_features = effects_df.sort_values("standardized_diff", ascending=True).head(8)

    lines = [
        "# Label Style EDA Report",
        "",
        "## Dataset",
        "",
        f"- Train rows: `{len(train_df)}`",
        f"- Label counts: `{label_counts}`",
        "",
        "## Strongest Numeric Differences",
        "",
        "Features higher in `TRUE`:",
        "",
    ]
    for row in top_true_features.itertuples():
        lines.append(
            f"- `{row.feature}`: TRUE mean `{row.true_mean:.4f}`, FALSE mean `{row.false_mean:.4f}`, standardized diff `{row.standardized_diff:.4f}`"
        )

    lines += ["", "Features higher in `FALSE`:", ""]
    for row in top_false_features.itertuples():
        lines.append(
            f"- `{row.feature}`: TRUE mean `{row.true_mean:.4f}`, FALSE mean `{row.false_mean:.4f}`, standardized diff `{row.standardized_diff:.4f}`"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The labels are not explained by explicit harm terms alone. `TRUE` is better understood as harmful, unsafe, or abnormal generated text. The useful signals are weak and overlapping: length, malformed tokens, non-ASCII/noisy text, code-like fragments, repetition, prompt-template residue, and spam/product wording all matter, but none is a rule by itself.",
        "",
        "The practical modeling implication is that RoBERTa should remain the semantic anchor, while deterministic abnormality features should be used as a correction/calibration layer around borderline examples.",
        "",
    ]

    if val_df is not None:
        outcome_counts = val_df["outcome"].value_counts().to_dict()
        lines += [
            "## RoBERTa Validation Error Pattern",
            "",
            f"- Outcome counts: `{outcome_counts}`",
            "",
            "False positives are usually text that looks abnormal but is labeled `FALSE`: mixed-language noise, code fragments, malformed prompts, and weird blog/product prose.",
            "",
            "False negatives are usually less visually extreme but still semantically broken: plausible-looking product, travel, science, or business sentences with drift, odd substitutions, or stitched-together fragments.",
            "",
        ]

    if not lb_df.empty:
        best = lb_df.iloc[0]
        lines += [
            "## Leaderboard Positive Count Pattern",
            "",
            f"- Best known submission: `{best.submission}` with score `{best.lb_score:.8f}` and `{int(best.true_count)}` TRUE predictions.",
            "- Good public submissions cluster around roughly `574-580` TRUE predictions.",
            "- Bad submissions are often far from that region, but TRUE count alone is not enough; the selected rows still matter.",
            "",
        ]

    lines += [
        "## Recommended Next Step",
        "",
        "Build a lightweight calibrator using RoBERTa probability plus the deterministic label-style features from this EDA. Keep it simple and regularized first. Use the known RoBERTa validation split as a quick experiment, then move to OOF only if the idea shows promise.",
        "",
        "Generated files are saved under `outputs/eda_label_style/`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train/test data")
    train_df = load_jsonl(TRAIN_PATH)
    test_df = load_jsonl(TEST_PATH)
    train_df["text"] = train_df["text"].map(normalize_text)
    test_df["text"] = test_df["text"].map(normalize_text)
    train_df["label_norm"] = train_df["label"].map(normalize_label)

    print("Computing train/test label-style features")
    train_features = make_feature_frame(train_df["text"], desc="Train text features")
    test_features = make_feature_frame(test_df["text"], desc="Test text features")
    feature_cols = list(train_features.columns)
    train_full = pd.concat([train_df.reset_index(drop=True), train_features], axis=1)
    test_full = pd.concat([test_df.reset_index(drop=True), test_features], axis=1)

    train_full.to_csv(OUTPUT_DIR / "train_label_style_features.csv", index=False)
    test_full.to_csv(OUTPUT_DIR / "test_label_style_features.csv", index=False)

    print("Summarizing feature differences by label")
    summary_df, effects_df = summarize_by_label(train_full, feature_cols)
    summary_df.to_csv(FEATURE_SUMMARY_PATH, index=False)
    effects_df.to_csv(FEATURE_EFFECTS_PATH, index=False)

    print("Extracting discriminative word and char n-grams")
    word_true, word_false = log_odds_indicators(
        texts=train_full["text"],
        labels=train_full["label_norm"],
        vectorizer=CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=30000),
        top_n=150,
    )
    char_true, char_false = log_odds_indicators(
        texts=train_full["text"],
        labels=train_full["label_norm"],
        vectorizer=CountVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000),
        top_n=150,
    )
    word_true.to_csv(WORD_TRUE_PATH, index=False)
    word_false.to_csv(WORD_FALSE_PATH, index=False)
    char_true.to_csv(CHAR_TRUE_PATH, index=False)
    char_false.to_csv(CHAR_FALSE_PATH, index=False)

    print("Saving extreme feature examples")
    build_extreme_examples(train_full, feature_cols).to_csv(EXTREME_EXAMPLES_PATH, index=False)

    val_with_features = None
    if ROBERTA_VAL_PATH.exists():
        print("Analyzing RoBERTa validation outcomes")
        val_df = pd.read_csv(ROBERTA_VAL_PATH)
        val_df["text"] = val_df["text"].map(normalize_text)
        val_df = add_roberta_outcomes(val_df)
        val_features = make_feature_frame(val_df["text"], desc="RoBERTa val text features")
        val_with_features = pd.concat([val_df.reset_index(drop=True), val_features], axis=1)
        outcome_summary = (
            val_with_features.groupby("outcome")[feature_cols]
            .agg(["mean", "median", "std"])
            .round(6)
        )
        outcome_summary.columns = ["_".join(col).strip() for col in outcome_summary.columns.to_flat_index()]
        outcome_summary.reset_index().to_csv(VAL_OUTCOME_SUMMARY_PATH, index=False)

        if ROBERTA_ERROR_PATH.exists():
            errors = pd.read_csv(ROBERTA_ERROR_PATH)
            errors["text"] = errors["text"].map(normalize_text)
            error_features = make_feature_frame(errors["text"], desc="RoBERTa error text features")
            pd.concat([errors.reset_index(drop=True), error_features], axis=1).to_csv(ROBERTA_ERRORS_PATH, index=False)

    print("Summarizing leaderboard submission count pattern")
    lb_df = build_lb_pattern()
    lb_df.to_csv(LB_PATTERN_PATH, index=False)

    print("Writing report")
    write_report(train_full, effects_df, val_with_features, lb_df)

    print(f"Saved EDA outputs to: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
