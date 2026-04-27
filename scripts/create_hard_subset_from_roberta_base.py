#!/usr/bin/env python3
"""
Create a hard validation subset from saved RoBERTa-base predictions.

Dependency install command:
pip install pandas numpy

This script does not train a model and does not touch test data. It builds a
boundary/error subset from the existing base RoBERTa validation artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data" / "train_labeled_comp.jsonl"
VAL_PRED_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv"
ERROR_ANALYSIS_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_error_analysis.csv"
METRICS_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_metrics.json"

OUTPUT_JSONL_PATH = ROOT_DIR / "data" / "train_hard_roberta_base.jsonl"
OUTPUT_DIR = ROOT_DIR / "analysis" / "hard_subset"
OUTPUT_CSV_PATH = OUTPUT_DIR / "roberta_base_hard_subset.csv"
OUTPUT_REPORT_PATH = OUTPUT_DIR / "roberta_base_hard_subset_report.md"

REQUIRED_VAL_COLUMNS = {"text", "true_label", "pred_label", "pred_prob_TRUE"}
REQUIRED_ERROR_COLUMNS = {"text", "true_label", "pred_label", "pred_prob_TRUE", "error_type", "confidence"}
REQUIRED_TRAIN_COLUMNS = {"text", "label"}
DEFAULT_UNCERTAIN_LOW = 0.05
DEFAULT_UNCERTAIN_HIGH = 0.95
TARGET_MIN_ROWS = 300
MAX_FINAL_ROWS = 350
MIN_CLASS_FRACTION = 0.35


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


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
    raise ValueError(f"Unexpected label value: {value!r}")


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_labels(labels: Iterable[object], name: str) -> None:
    normalized = {normalize_label(value) for value in labels}
    invalid = normalized - {"TRUE", "FALSE"}
    if invalid:
        raise ValueError(f"{name} contains invalid labels after normalization: {sorted(invalid)}")


def load_selected_threshold(metrics_path: Path) -> float:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    try:
        return float(metrics["threshold_tuning"]["selected_threshold"])
    except KeyError as exc:
        raise KeyError("metrics JSON must contain threshold_tuning.selected_threshold") from exc


def reason_string(row: pd.Series) -> str:
    reasons = []
    if bool(row["is_uncertain_band"]):
        reasons.append("uncertain_band")
    if bool(row["is_wrong_prediction"]):
        reasons.append("wrong_prediction")
    if bool(row["is_error_analysis"]):
        reasons.append("error_analysis")
    return "+".join(reasons)


def short_text(text: str, limit: int = 420) -> str:
    text = normalize_text(text).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def markdown_examples(df: pd.DataFrame, title: str, max_rows: int = 8) -> list[str]:
    lines = [f"## {title}", ""]
    if df.empty:
        lines.append("No rows found.")
        lines.append("")
        return lines

    cols = ["label", "pred_label", "pred_prob_TRUE", "source_reason", "text"]
    example_df = df.head(max_rows).copy()
    example_df["text"] = example_df["text"].map(short_text)
    lines.append(example_df[cols].to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a hard validation subset from saved RoBERTa-base outputs.",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=DEFAULT_UNCERTAIN_LOW,
        help="Lower bound for uncertainty band, inclusive. Default: 0.05",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=DEFAULT_UNCERTAIN_HIGH,
        help="Upper bound for uncertainty band, inclusive. Default: 0.95",
    )
    args = parser.parse_args()
    if not (0.0 <= args.low <= 1.0):
        raise ValueError("--low must be between 0 and 1.")
    if not (0.0 <= args.high <= 1.0):
        raise ValueError("--high must be between 0 and 1.")
    if args.low > args.high:
        raise ValueError("--low must be <= --high.")
    return args


def label_counts_for(df: pd.DataFrame) -> tuple[int, int, int]:
    false_count = int((df["label"] == "FALSE").sum())
    true_count = int((df["label"] == "TRUE").sum())
    total = int(len(df))
    return false_count, true_count, total


def add_near_boundary_support_rows(
    core_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selected_threshold: float,
) -> tuple[pd.DataFrame, int, int]:
    """Add near-boundary validation rows without using test data.

    Core hard rows are never removed. Support rows are correct, outside-band rows
    not already selected by core hard rules. FALSE support is added first because
    the core subset is currently TRUE-heavy.
    """
    support_pool = val_df[~val_df["text_key"].isin(set(core_df["text_key"]))].copy()
    support_pool["label"] = support_pool["true_label"]
    support_pool["decision_boundary_distance"] = (
        support_pool["pred_prob_TRUE"] - selected_threshold
    ).abs()

    false_pool = support_pool[support_pool["label"] == "FALSE"].sort_values(
        ["decision_boundary_distance", "pred_prob_TRUE"],
        ascending=[True, False],
    )
    true_pool = support_pool[support_pool["label"] == "TRUE"].sort_values(
        ["decision_boundary_distance", "pred_prob_TRUE"],
        ascending=[True, True],
    )

    selected_frames = [core_df]
    added_false = 0
    added_true = 0
    current_df = core_df.copy()

    for _, row in false_pool.iterrows():
        false_count, true_count, total = label_counts_for(current_df)
        false_fraction = false_count / max(total, 1)
        if total >= MAX_FINAL_ROWS:
            break
        if total >= TARGET_MIN_ROWS and false_fraction >= MIN_CLASS_FRACTION:
            break

        support_row = row.copy()
        support_row["source_reason"] = "support_FALSE_near_boundary"
        support_df = pd.DataFrame([support_row])
        selected_frames.append(support_df)
        current_df = pd.concat([current_df, support_df], ignore_index=True)
        added_false += 1

    for _, row in true_pool.iterrows():
        false_count, true_count, total = label_counts_for(current_df)
        true_fraction = true_count / max(total, 1)
        projected_true_fraction = (true_count + 1) / max(total + 1, 1)
        if total >= MAX_FINAL_ROWS:
            break
        if true_fraction >= MIN_CLASS_FRACTION:
            break
        if projected_true_fraction > (1.0 - MIN_CLASS_FRACTION):
            break

        support_row = row.copy()
        support_row["source_reason"] = "support_TRUE_near_boundary"
        support_df = pd.DataFrame([support_row])
        selected_frames.append(support_df)
        current_df = pd.concat([current_df, support_df], ignore_index=True)
        added_true += 1

    final_df = pd.concat(selected_frames, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["text_key"], keep="first").copy()
    return final_df, added_false, added_true


def main() -> None:
    args = parse_args()
    uncertain_low = float(args.low)
    uncertain_high = float(args.high)

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Create hard subset from RoBERTa-base validation outputs ===")
    print("No model training is performed. Test data is not used.")
    print(f"Uncertainty band: [{uncertain_low:.2f}, {uncertain_high:.2f}]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train data: {TRAIN_PATH}")
    for path in [VAL_PRED_PATH, ERROR_ANALYSIS_PATH, METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required RoBERTa artifact: {path}")

    train_df = load_jsonl(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PRED_PATH)
    err_df = pd.read_csv(ERROR_ANALYSIS_PATH)
    selected_threshold = load_selected_threshold(METRICS_PATH)

    validate_columns(train_df, REQUIRED_TRAIN_COLUMNS, "train_labeled_comp.jsonl")
    validate_columns(val_df, REQUIRED_VAL_COLUMNS, "roberta_base_val_predictions.csv")
    validate_columns(err_df, REQUIRED_ERROR_COLUMNS, "roberta_base_error_analysis.csv")

    train_df = train_df.copy()
    val_df = val_df.copy()
    err_df = err_df.copy()
    train_df["text"] = train_df["text"].map(normalize_text)
    val_df["text"] = val_df["text"].map(normalize_text)
    err_df["text"] = err_df["text"].map(normalize_text)
    train_df["text_key"] = train_df["text"]
    val_df["text_key"] = val_df["text"]
    err_df["text_key"] = err_df["text"]

    train_df["label"] = train_df["label"].map(normalize_label)
    val_df["true_label"] = val_df["true_label"].map(normalize_label)
    val_df["pred_label"] = val_df["pred_label"].map(normalize_label)
    err_df["true_label"] = err_df["true_label"].map(normalize_label)
    err_df["pred_label"] = err_df["pred_label"].map(normalize_label)

    validate_labels(train_df["label"], "train label")
    validate_labels(val_df["true_label"], "validation true_label")
    validate_labels(val_df["pred_label"], "validation pred_label")
    validate_labels(err_df["true_label"], "error true_label")
    validate_labels(err_df["pred_label"], "error pred_label")

    val_df["pred_prob_TRUE"] = pd.to_numeric(val_df["pred_prob_TRUE"], errors="raise").astype(float)
    if not val_df["pred_prob_TRUE"].between(0.0, 1.0).all():
        raise ValueError("pred_prob_TRUE values must be between 0 and 1.")

    error_keys = set(err_df["text_key"])
    val_df["is_uncertain_band"] = val_df["pred_prob_TRUE"].between(
        uncertain_low,
        uncertain_high,
        inclusive="both",
    )
    val_df["is_wrong_prediction"] = val_df["true_label"] != val_df["pred_label"]
    val_df["is_error_analysis"] = val_df["text_key"].isin(error_keys)

    hard_df = val_df[
        val_df["is_uncertain_band"] | val_df["is_wrong_prediction"] | val_df["is_error_analysis"]
    ].copy()

    if hard_df.empty:
        raise RuntimeError("No hard rows selected. Check thresholds and input files.")

    hard_df["source_reason"] = hard_df.apply(reason_string, axis=1)
    before_dedup = len(hard_df)
    hard_df = hard_df.drop_duplicates(subset=["text_key"], keep="first").copy()
    core_after_dedup = len(hard_df)
    hard_df["label"] = hard_df["true_label"]

    hard_df, added_false_support_count, added_true_support_count = add_near_boundary_support_rows(
        core_df=hard_df,
        val_df=val_df,
        selected_threshold=selected_threshold,
    )
    after_dedup = len(hard_df)

    train_lookup = train_df.drop_duplicates(subset=["text_key"], keep="first").set_index("text_key")["label"]
    missing_from_train = sorted(set(hard_df["text_key"]) - set(train_lookup.index))
    if missing_from_train:
        raise ValueError(f"{len(missing_from_train)} hard rows were not found in train_labeled_comp.jsonl.")

    hard_df["train_label"] = hard_df["text_key"].map(train_lookup)
    label_mismatch = hard_df[hard_df["train_label"] != hard_df["label"]]
    if not label_mismatch.empty:
        raise ValueError(
            "Found label mismatch between validation true_label and train_labeled_comp.jsonl "
            f"for {len(label_mismatch)} hard rows."
        )

    output_cols = ["text", "label", "pred_prob_TRUE", "pred_label", "true_label", "source_reason"]
    diagnostic_df = hard_df[output_cols].copy()
    diagnostic_df.to_csv(OUTPUT_CSV_PATH, index=False)

    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in diagnostic_df[["text", "label"]].itertuples(index=False):
            handle.write(json.dumps({"text": row.text, "label": row.label}, ensure_ascii=False) + "\n")

    label_counts = diagnostic_df["label"].value_counts().reindex(["FALSE", "TRUE"], fill_value=0)
    label_pct = (diagnostic_df["label"].value_counts(normalize=True).reindex(["FALSE", "TRUE"], fill_value=0) * 100)
    majority_pct = float(label_pct.max()) if not label_pct.empty else 0.0
    mean_prob_by_label = diagnostic_df.groupby("label")["pred_prob_TRUE"].mean().reindex(["FALSE", "TRUE"])
    reason_counts = Counter()
    for value in diagnostic_df["source_reason"]:
        for reason in str(value).split("+"):
            reason_counts[reason] += 1
    exact_reason_counts = diagnostic_df["source_reason"].value_counts()

    hard_true_examples = diagnostic_df[diagnostic_df["label"] == "TRUE"].sort_values(
        ["pred_prob_TRUE"], ascending=True
    )
    hard_false_examples = diagnostic_df[diagnostic_df["label"] == "FALSE"].sort_values(
        ["pred_prob_TRUE"], ascending=False
    )
    false_negatives = diagnostic_df[
        (diagnostic_df["true_label"] == "TRUE") & (diagnostic_df["pred_label"] == "FALSE")
    ].sort_values("pred_prob_TRUE", ascending=True)
    false_positives = diagnostic_df[
        (diagnostic_df["true_label"] == "FALSE") & (diagnostic_df["pred_label"] == "TRUE")
    ].sort_values("pred_prob_TRUE", ascending=False)

    lines = [
        "# RoBERTa Base Hard Subset Report",
        "",
        "This report builds a hard validation subset from saved `roberta_base` outputs. It does not train a model and does not use test labels or test rows.",
        "",
        "## Summary",
        "",
        f"- Full train rows loaded: `{len(train_df)}`",
        f"- Base validation rows: `{len(val_df)}`",
        f"- Selected threshold from metrics: `{selected_threshold:.4f}`",
        f"- Uncertainty band: `[{uncertain_low:.2f}, {uncertain_high:.2f}]`",
        f"- Core hard rows before deduplication: `{before_dedup}`",
        f"- Core hard rows after deduplication: `{core_after_dedup}`",
        f"- Duplicate core hard rows removed by normalized text: `{before_dedup - core_after_dedup}`",
        f"- Added FALSE support rows: `{added_false_support_count}`",
        f"- Added TRUE support rows: `{added_true_support_count}`",
        f"- Final hard rows after balancing: `{after_dedup}`",
        f"- Hard rows matched back to train: `{after_dedup}`",
        f"- Final TRUE percentage: `{label_pct['TRUE']:.2f}%`",
        f"- Final FALSE percentage: `{label_pct['FALSE']:.2f}%`",
        f"- Final row cap: `{MAX_FINAL_ROWS}`",
        f"- Output JSONL: `data/train_hard_roberta_base.jsonl`",
        f"- Diagnostic CSV: `analysis/hard_subset/roberta_base_hard_subset.csv`",
        "",
    ]

    warnings = []
    if after_dedup < TARGET_MIN_ROWS:
        warnings.append(
            f"Hard subset size is below target minimum `{TARGET_MIN_ROWS}`; consider increasing `MAX_FINAL_ROWS` or reviewing support-row availability."
        )
    if majority_pct > (1.0 - MIN_CLASS_FRACTION) * 100:
        warnings.append(
            f"Hard subset label split is worse than 65/35 (`{label_pct.idxmax()}` = `{majority_pct:.2f}%`); specialist training may become biased."
        )
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
        lines.append("")

    lines.extend([
        "## Label Distribution In Hard Subset",
        "",
        "| label | count | pct |",
        "|---|---:|---:|",
    ])
    for label in ["FALSE", "TRUE"]:
        lines.append(f"| {label} | {int(label_counts[label])} | {label_pct[label]:.2f}% |")

    lines.extend([
        "",
        "## Mean `pred_prob_TRUE` By Label",
        "",
        "| label | mean_pred_prob_TRUE |",
        "|---|---:|",
    ])
    for label in ["FALSE", "TRUE"]:
        value = mean_prob_by_label[label]
        lines.append(f"| {label} | {value:.6f} |" if pd.notna(value) else f"| {label} | n/a |")

    lines.extend([
        "",
        "## Count By Source Reason",
        "",
        "Reason counts are non-exclusive because one row can have multiple reasons.",
        "",
        "| source_reason_component | count |",
        "|---|---:|",
    ])
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend([
        "",
        "## Exact Source Reason Combinations",
        "",
        "| source_reason | count |",
        "|---|---:|",
    ])
    for reason, count in exact_reason_counts.items():
        lines.append(f"| {reason} | {int(count)} |")
    lines.append("")

    lines.extend(markdown_examples(hard_true_examples, "Examples Of Hard TRUE"))
    lines.extend(markdown_examples(hard_false_examples, "Examples Of Hard FALSE"))
    lines.extend(markdown_examples(false_negatives, "Examples Of False Negatives"))
    lines.extend(markdown_examples(false_positives, "Examples Of False Positives"))

    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Base validation rows: {len(val_df)}")
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(f"Core hard rows after deduplication: {core_after_dedup}")
    print(f"Added FALSE support rows: {added_false_support_count}")
    print(f"Added TRUE support rows: {added_true_support_count}")
    print(f"Final hard rows after balancing: {after_dedup}")
    print("Label distribution:")
    print(label_counts.to_string())
    print(f"Saved JSONL: {OUTPUT_JSONL_PATH}")
    print(f"Saved diagnostics CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved report: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
