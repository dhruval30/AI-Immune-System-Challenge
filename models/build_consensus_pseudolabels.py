#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roberta_dapt_pipeline_common import load_jsonl, make_occurrence_keys, normalize_texts, probability_entropy

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_consensus_pseudolabels"

TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
CONSENSUS_PATH = ROOT_DIR / "outputs" / "roberta_teacher_targets" / "roberta_teacher_soft_targets_test.csv"

PSEUDO_LABELS_CSV_PATH = OUTPUT_DIR / "roberta_consensus_pseudolabels.csv"
PSEUDO_LABELS_JSONL_PATH = OUTPUT_DIR / "roberta_consensus_pseudolabels.jsonl"
METRICS_PATH = OUTPUT_DIR / "roberta_consensus_pseudolabels_metrics.json"
NOTES_PATH = OUTPUT_DIR / "roberta_consensus_pseudolabels_notes.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced pseudo labels from teacher consensus.")
    parser.add_argument("--consensus-path", type=Path, default=CONSENSUS_PATH)
    parser.add_argument(
        "--extra-test-prob-path",
        type=Path,
        action="append",
        default=[],
        help="Optional extra test probability csv files to average into the consensus.",
    )
    parser.add_argument("--positive-threshold", type=float, default=0.83)
    parser.add_argument("--negative-threshold", type=float, default=0.03)
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--min-teacher-count", type=int, default=2)
    parser.add_argument("--max-prob-std", type=float, default=0.15)
    return parser.parse_args()


def infer_prob_column(df: pd.DataFrame) -> str:
    for column in ("teacher_prob_TRUE", "pred_prob_TRUE", "pseudo_prob_TRUE", "oof_prob_TRUE", "oof_prob_TRUE_mean"):
        if column in df.columns:
            return column
    raise ValueError(f"Unable to infer probability column from columns: {df.columns.tolist()}")


def align_extra_probabilities(
    probability_path: Path,
    target_keys: list[str],
) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(probability_path)
    if "text" not in df.columns:
        raise ValueError(f"{probability_path} is missing the required 'text' column.")
    prob_column = infer_prob_column(df)
    keys = make_occurrence_keys(df["text"].tolist())
    target_lookup = {key: idx for idx, key in enumerate(target_keys)}
    aligned = np.full(len(target_keys), np.nan, dtype=np.float64)

    matched = 0
    duplicates = 0
    unmatched = 0
    for key, prob in zip(keys, df[prob_column].tolist()):
        if pd.isna(prob):
            continue
        target_idx = target_lookup.get(key)
        if target_idx is None:
            unmatched += 1
            continue
        if np.isfinite(aligned[target_idx]):
            duplicates += 1
            continue
        aligned[target_idx] = float(prob)
        matched += 1

    report = {
        "path": str(probability_path),
        "rows": int(len(df)),
        "matched_rows": int(matched),
        "coverage": float(matched / max(len(target_keys), 1)),
        "duplicate_rows_ignored": int(duplicates),
        "unmatched_rows": int(unmatched),
        "probability_column": prob_column,
    }
    return aligned, report


def build_jsonl_records(df: pd.DataFrame) -> list[str]:
    records = []
    for row in df.itertuples(index=False):
        payload = {
            "test_row_index": int(row.test_row_index),
            "text": row.text,
            "label": row.pseudo_label,
            "pseudo_prob_TRUE": float(row.pseudo_prob_TRUE),
            "consensus_entropy": float(row.consensus_entropy),
            "teacher_count": int(row.teacher_count),
        }
        records.append(json.dumps(payload, ensure_ascii=True))
    return records


def write_notes(metrics: dict) -> None:
    notes = f"""# Consensus Pseudolabel Notes

## Purpose

This script creates a balanced pseudo-label slice from the highest-confidence test rows.
It is designed for optional second-stage training only after the DAPT and distillation stages.

## Configuration

- Positive threshold: `{metrics['config']['positive_threshold']}`
- Negative threshold: `{metrics['config']['negative_threshold']}`
- Max per class: `{metrics['config']['max_per_class']}`
- Min teacher count: `{metrics['config']['min_teacher_count']}`
- Max probability std: `{metrics['config']['max_prob_std']}`

## Result

- Selected TRUE pseudo labels: `{metrics['selection']['selected_true']}`
- Selected FALSE pseudo labels: `{metrics['selection']['selected_false']}`
- Total pseudo labels: `{metrics['selection']['selected_total']}`

## Outputs

- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels.csv`
- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels.jsonl`
- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels_metrics.json`
- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels_notes.md`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Building balanced consensus pseudo labels ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.consensus_path.exists():
        raise FileNotFoundError(f"Consensus teacher file not found: {args.consensus_path}")

    print("\n=== Loading canonical test rows ===")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    test_keys = make_occurrence_keys(test_texts)

    print("\n=== Loading teacher consensus ===")
    consensus_df = pd.read_csv(args.consensus_path)
    if "text" not in consensus_df.columns:
        raise ValueError(f"{args.consensus_path} is missing the required 'text' column.")
    if "teacher_prob_TRUE" not in consensus_df.columns:
        raise ValueError(f"{args.consensus_path} must contain 'teacher_prob_TRUE'.")

    consensus_keys = make_occurrence_keys(consensus_df["text"].tolist())
    consensus_lookup = {key: idx for idx, key in enumerate(consensus_keys)}
    aligned_teacher_prob = np.full(len(test_texts), np.nan, dtype=np.float64)
    aligned_teacher_count = np.zeros(len(test_texts), dtype=np.int64)
    aligned_teacher_std = np.full(len(test_texts), np.nan, dtype=np.float64)

    for target_idx, target_key in enumerate(test_keys):
        consensus_idx = consensus_lookup.get(target_key)
        if consensus_idx is None:
            continue
        aligned_teacher_prob[target_idx] = float(consensus_df.iloc[consensus_idx]["teacher_prob_TRUE"])
        if "teacher_count" in consensus_df.columns:
            aligned_teacher_count[target_idx] = int(consensus_df.iloc[consensus_idx]["teacher_count"])
        if "teacher_prob_std" in consensus_df.columns:
            aligned_teacher_std[target_idx] = float(consensus_df.iloc[consensus_idx]["teacher_prob_std"])

    source_columns = {"teacher_consensus_prob_TRUE": aligned_teacher_prob}
    source_reports = [
        {
            "path": str(args.consensus_path),
            "matched_rows": int(np.isfinite(aligned_teacher_prob).sum()),
            "coverage": float(np.isfinite(aligned_teacher_prob).sum() / max(len(test_texts), 1)),
            "source_name": "teacher_consensus",
        }
    ]

    for extra_path in args.extra_test_prob_path:
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra probability file not found: {extra_path}")
        aligned_probs, report = align_extra_probabilities(extra_path, test_keys)
        column_name = f"{extra_path.stem}_prob_TRUE"
        source_columns[column_name] = aligned_probs
        source_reports.append({"source_name": extra_path.stem, **report})

    source_matrix_columns = list(source_columns)
    source_matrix = np.column_stack([source_columns[column] for column in source_matrix_columns])

    consensus_prob = np.nanmean(source_matrix, axis=1)
    consensus_entropy = np.asarray(
        [probability_entropy(prob) / np.log(2.0) if np.isfinite(prob) else np.nan for prob in consensus_prob],
        dtype=np.float64,
    )

    test_output = pd.DataFrame(
        {
            "test_row_index": np.arange(len(test_texts), dtype=np.int64),
            "text": test_texts,
            "consensus_prob_TRUE": consensus_prob,
            "consensus_entropy": consensus_entropy,
            "teacher_count": aligned_teacher_count,
            "teacher_prob_std": aligned_teacher_std,
        }
    )
    for column in source_matrix_columns:
        test_output[column] = source_columns[column]

    base_candidate_mask = np.isfinite(test_output["consensus_prob_TRUE"].to_numpy())
    if "teacher_count" in test_output.columns:
        base_candidate_mask &= test_output["teacher_count"].to_numpy() >= args.min_teacher_count
    if "teacher_prob_std" in test_output.columns:
        std_values = test_output["teacher_prob_std"].to_numpy()
        std_mask = np.isnan(std_values) | (std_values <= args.max_prob_std)
        base_candidate_mask &= std_mask

    positive_candidates = test_output[
        base_candidate_mask & (test_output["consensus_prob_TRUE"].to_numpy() >= args.positive_threshold)
    ].copy()
    negative_candidates = test_output[
        base_candidate_mask & (test_output["consensus_prob_TRUE"].to_numpy() <= args.negative_threshold)
    ].copy()

    positive_candidates = positive_candidates.sort_values(
        by=["consensus_prob_TRUE", "teacher_count"],
        ascending=[False, False],
    )
    negative_candidates = negative_candidates.sort_values(
        by=["consensus_prob_TRUE", "teacher_count"],
        ascending=[True, False],
    )

    per_class_quota = min(args.max_per_class, len(positive_candidates), len(negative_candidates))
    selected_positive = positive_candidates.head(per_class_quota).copy()
    selected_negative = negative_candidates.head(per_class_quota).copy()

    selected_positive["pseudo_label"] = "TRUE"
    selected_positive["pseudo_prob_TRUE"] = selected_positive["consensus_prob_TRUE"]
    selected_negative["pseudo_label"] = "FALSE"
    selected_negative["pseudo_prob_TRUE"] = selected_negative["consensus_prob_TRUE"]

    selected = pd.concat([selected_positive, selected_negative], ignore_index=True)
    selected["pseudo_label_int"] = np.where(selected["pseudo_label"] == "TRUE", 1, 0)
    selected = selected.sort_values(by=["pseudo_label", "pseudo_prob_TRUE"], ascending=[True, False]).reset_index(drop=True)

    selected.to_csv(PSEUDO_LABELS_CSV_PATH, index=False)
    PSEUDO_LABELS_JSONL_PATH.write_text("\n".join(build_jsonl_records(selected)) + "\n", encoding="utf-8")

    metrics = {
        "config": {
            "consensus_path": str(args.consensus_path),
            "extra_test_prob_path": [str(path) for path in args.extra_test_prob_path],
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "max_per_class": args.max_per_class,
            "min_teacher_count": args.min_teacher_count,
            "max_prob_std": args.max_prob_std,
        },
        "data": {
            "test_rows": len(test_texts),
            "rows_with_consensus": int(np.isfinite(consensus_prob).sum()),
        },
        "sources": source_reports,
        "selection": {
            "positive_candidates": int(len(positive_candidates)),
            "negative_candidates": int(len(negative_candidates)),
            "selected_true": int(len(selected_positive)),
            "selected_false": int(len(selected_negative)),
            "selected_total": int(len(selected)),
        },
        "paths": {
            "pseudo_labels_csv": str(PSEUDO_LABELS_CSV_PATH),
            "pseudo_labels_jsonl": str(PSEUDO_LABELS_JSONL_PATH),
            "metrics": str(METRICS_PATH),
            "notes": str(NOTES_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_notes(metrics)

    print(f"Saved pseudo labels CSV: {PSEUDO_LABELS_CSV_PATH}")
    print(f"Saved pseudo labels JSONL: {PSEUDO_LABELS_JSONL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
