#!/usr/bin/env python3
"""
Categorize binary text-classification model errors into fixed failure patterns.

Dependency install command:
pip install pandas numpy

Default input:
outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_val_predictions.csv

The script does not relabel data. It only assigns one failure_category per row
based on observable patterns in model mistakes.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    ROOT_DIR
    / "outputs"
    / "roberta_hard_weighted_checkpoint_inference"
    / "roberta_hard_weighted_checkpoint_inference_val_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "error_failure_categories"

ALLOWED_CATEGORIES = [
    "CLEAN_SEMANTIC_TRUE",
    "NOISY_FALSE",
    "PROMPT_OR_CODE_LEAKAGE",
    "REPETITIVE_OR_SPAM",
    "MULTILINGUAL_OR_ENCODING",
    "BORDERLINE_AMBIGUOUS",
    "LIKELY_LABEL_NOISE",
]

PROMPT_CODE_PATTERNS = [
    r"\buser\s*:",
    r"\bassistant\s*:",
    r"\bsystem\s*:",
    r"```",
    r"</s>",
    r"<s>",
    r"\[/inst\]",
    r"\[inst\]",
    r"\bjson\b",
    r"\bimport\s+[a-zA-Z_]",
    r"\bdef\s+[a-zA-Z_]",
    r"\bclass\s+[a-zA-Z_]",
    r"\bprint\s*\(",
    r"\breturn\s+",
    r"\bfunction\s*\(",
    r"\bconsole\.log\s*\(",
    r"\{[\"']?[a-zA-Z_][a-zA-Z0-9_ -]*[\"']?\s*:",
    r"^\s*[-*]\s+",
    r"^\s*#{1,6}\s+",
]

WORD_RE = re.compile(r"[A-Za-z0-9_']+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
PUNCT_CHARS = set(string.punctuation)


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def safe_prob(value: Any) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(prob):
        return 0.0
    return min(max(prob, 0.0), 1.0)


def text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    lower = text.lower()
    chars = list(text)
    char_count = max(len(chars), 1)
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    raw_tokens = re.findall(r"\S+", text)
    word_count = max(len(words), 1)
    token_count = max(len(raw_tokens), 1)
    token_counts = pd.Series(words).value_counts() if words else pd.Series(dtype=int)

    repeated_token_count = int((token_counts[token_counts > 1] - 1).sum()) if len(token_counts) else 0
    repeated_token_ratio = repeated_token_count / word_count
    unique_word_ratio = len(set(words)) / word_count if words else 0.0
    non_ascii_ratio = sum(ord(char) > 127 for char in chars) / char_count
    punctuation_ratio = sum(char in PUNCT_CHARS for char in chars) / char_count
    digit_ratio = sum(char.isdigit() for char in chars) / char_count
    newline_count = text.count("\n")
    repeated_char_count = len(REPEATED_CHAR_RE.findall(text))

    weird_tokens = 0
    long_tokens = 0
    mixed_alnum_tokens = 0
    symbol_heavy_tokens = 0
    for token in raw_tokens:
        has_alpha = any(char.isalpha() for char in token)
        has_digit = any(char.isdigit() for char in token)
        symbol_count = sum(not char.isalnum() for char in token)
        if len(token) >= 18:
            long_tokens += 1
        if has_alpha and has_digit:
            mixed_alnum_tokens += 1
        if symbol_count >= 3:
            symbol_heavy_tokens += 1
        if len(token) >= 18 or (has_alpha and has_digit) or symbol_count >= 3 or REPEATED_CHAR_RE.search(token):
            weird_tokens += 1

    repeated_phrase_count = 0
    if len(words) >= 6:
        ngrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        ngram_counts = CounterSafe(ngrams)
        repeated_phrase_count = sum(count - 1 for count in ngram_counts.values() if count > 1)

    prompt_code_hits = sum(1 for pattern in PROMPT_CODE_PATTERNS if re.search(pattern, lower, flags=re.MULTILINE))

    return {
        "char_length": float(len(text)),
        "word_count": float(len(words)),
        "token_count": float(len(raw_tokens)),
        "unique_word_ratio": float(unique_word_ratio),
        "repeated_token_count": float(repeated_token_count),
        "repeated_token_ratio": float(repeated_token_ratio),
        "repeated_phrase_count": float(repeated_phrase_count),
        "weird_token_count": float(weird_tokens),
        "weird_token_ratio": float(weird_tokens / token_count),
        "long_token_count": float(long_tokens),
        "mixed_alnum_token_count": float(mixed_alnum_tokens),
        "symbol_heavy_token_count": float(symbol_heavy_tokens),
        "non_ascii_ratio": float(non_ascii_ratio),
        "punctuation_ratio": float(punctuation_ratio),
        "digit_ratio": float(digit_ratio),
        "newline_count": float(newline_count),
        "repeated_char_count": float(repeated_char_count),
        "prompt_code_hits": float(prompt_code_hits),
    }


def CounterSafe(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def assign_failure_category(
    text: str,
    true_label: str,
    pred_label: str,
    pred_prob_true: float,
) -> tuple[str, dict[str, float]]:
    text = normalize_text(text)
    true_label = normalize_label(true_label)
    pred_label = normalize_label(pred_label)
    pred_prob_true = safe_prob(pred_prob_true)
    features = text_features(text)

    is_error = true_label != pred_label
    if not is_error:
        return "BORDERLINE_AMBIGUOUS", features

    noisy = bool(
        features["weird_token_ratio"] >= 0.08
        or features["punctuation_ratio"] >= 0.08
        or features["long_token_count"] >= 1
        or features["mixed_alnum_token_count"] >= 1
        or features["symbol_heavy_token_count"] >= 1
        or features["newline_count"] >= 2
        or features["repeated_char_count"] >= 1
    )
    repetitive = bool(
        features["repeated_token_ratio"] >= 0.20
        or features["repeated_phrase_count"] >= 1
        or (features["word_count"] >= 25 and features["unique_word_ratio"] <= 0.72)
    )
    multilingual = bool(features["non_ascii_ratio"] >= 0.08)
    prompt_or_code = bool(features["prompt_code_hits"] >= 1)

    confident_wrong = (
        (pred_label == "TRUE" and pred_prob_true >= 0.85)
        or (pred_label == "FALSE" and pred_prob_true <= 0.15)
    )

    clean = not noisy and not repetitive and not multilingual and not prompt_or_code

    if prompt_or_code:
        return "PROMPT_OR_CODE_LEAKAGE", features
    if multilingual:
        return "MULTILINGUAL_OR_ENCODING", features
    if repetitive:
        return "REPETITIVE_OR_SPAM", features

    if true_label == "FALSE" and pred_label == "TRUE":
        if noisy:
            return "NOISY_FALSE", features
        if clean and confident_wrong:
            return "LIKELY_LABEL_NOISE", features
        return "BORDERLINE_AMBIGUOUS", features

    if true_label == "TRUE" and pred_label == "FALSE":
        if clean:
            if pred_prob_true <= 0.05 and features["word_count"] >= 8:
                return "LIKELY_LABEL_NOISE", features
            return "CLEAN_SEMANTIC_TRUE", features
        if noisy and confident_wrong:
            return "LIKELY_LABEL_NOISE", features
        return "BORDERLINE_AMBIGUOUS", features

    return "BORDERLINE_AMBIGUOUS", features


def validate_columns(df: pd.DataFrame) -> None:
    required = {"text", "true_label", "pred_label", "pred_prob_TRUE"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def snippet(text: str, max_chars: int = 220) -> str:
    clean = normalize_text(text).replace("\n", " / ")
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def write_markdown_report(
    output_path: Path,
    input_path: Path,
    categorized_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    error_only: bool,
) -> None:
    display_df = categorized_df.copy()
    display_df["snippet"] = [snippet(text) for text in display_df["text"]]
    example_rows = []
    for category in ALLOWED_CATEGORIES:
        subset = display_df[display_df["failure_category"] == category]
        if subset.empty:
            continue
        subset = subset.sort_values("confidence", ascending=False).head(5)
        example_rows.append(f"### {category}")
        example_rows.append("")
        example_rows.append(
            subset[
                [
                    "snippet",
                    "true_label",
                    "pred_label",
                    "pred_prob_TRUE",
                    "confidence",
                ]
            ].to_markdown(index=False)
        )
        example_rows.append("")

    text = f"""# Model Error Failure Category Report

Input file: `{input_path}`

Rows categorized: `{len(categorized_df)}`

Error-only mode: `{error_only}`

## Category Summary

{summary_df.to_markdown(index=False)}

## Category Definitions

- `CLEAN_SEMANTIC_TRUE`: fluent/normal-looking text labeled TRUE but predicted FALSE.
- `NOISY_FALSE`: messy/corrupted/weird text labeled FALSE but predicted TRUE.
- `PROMPT_OR_CODE_LEAKAGE`: prompt, assistant/user, code, JSON, markdown, or templated instruction leakage.
- `REPETITIVE_OR_SPAM`: heavy repetition, duplicated segments, SEO-like phrasing.
- `MULTILINGUAL_OR_ENCODING`: mixed language or broken encoding.
- `BORDERLINE_AMBIGUOUS`: no dominant observable failure signal.
- `LIKELY_LABEL_NOISE`: label appears visibly inconsistent with text and model confidence.

## Examples

{''.join(example_rows)}
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign fixed failure categories to model error rows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV with text,true_label,pred_label,pred_prob_TRUE columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for categorized CSV and summary outputs.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Categorize all rows. By default only model mistakes are categorized.",
    )
    return parser.parse_args()


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)
    df = df.copy()
    df["text"] = [normalize_text(value) for value in df["text"]]
    df["true_label"] = [normalize_label(value) for value in df["true_label"]]
    df["pred_label"] = [normalize_label(value) for value in df["pred_label"]]
    df["pred_prob_TRUE"] = [safe_prob(value) for value in df["pred_prob_TRUE"]]
    df["is_error"] = df["true_label"] != df["pred_label"]

    if args.include_correct:
        work_df = df.copy()
    else:
        work_df = df[df["is_error"]].copy()

    categories = []
    feature_rows = []
    for _, row in work_df.iterrows():
        category, features = assign_failure_category(
            text=row["text"],
            true_label=row["true_label"],
            pred_label=row["pred_label"],
            pred_prob_true=row["pred_prob_TRUE"],
        )
        categories.append(category)
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    categorized_df = pd.concat([work_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    categorized_df["failure_category"] = categories
    categorized_df["confidence"] = np.where(
        categorized_df["pred_label"] == "TRUE",
        categorized_df["pred_prob_TRUE"],
        1.0 - categorized_df["pred_prob_TRUE"],
    )

    invalid = sorted(set(categorized_df["failure_category"]) - set(ALLOWED_CATEGORIES))
    if invalid:
        raise ValueError(f"Invalid categories produced: {invalid}")

    input_stem = input_path.stem
    categorized_path = output_dir / f"{input_stem}_failure_categories.csv"
    summary_path = output_dir / f"{input_stem}_failure_category_summary.csv"
    report_path = output_dir / f"{input_stem}_failure_category_report.md"
    meta_path = output_dir / f"{input_stem}_failure_category_meta.json"

    categorized_df.to_csv(categorized_path, index=False)

    summary_df = (
        categorized_df.groupby(["failure_category", "true_label", "pred_label"])
        .agg(
            rows=("failure_category", "size"),
            mean_pred_prob_TRUE=("pred_prob_TRUE", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_word_count=("word_count", "mean"),
            mean_weird_token_ratio=("weird_token_ratio", "mean"),
            mean_repeated_token_ratio=("repeated_token_ratio", "mean"),
            mean_non_ascii_ratio=("non_ascii_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["failure_category", "rows"], ascending=[True, False])
    )
    summary_df.to_csv(summary_path, index=False)

    write_markdown_report(
        output_path=report_path,
        input_path=input_path,
        categorized_df=categorized_df,
        summary_df=summary_df,
        error_only=not args.include_correct,
    )

    meta = {
        "input": str(input_path),
        "rows_in_input": int(len(df)),
        "rows_categorized": int(len(categorized_df)),
        "include_correct": bool(args.include_correct),
        "allowed_categories": ALLOWED_CATEGORIES,
        "outputs": {
            "categorized": str(categorized_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "meta": str(meta_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved categorized rows: {categorized_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
