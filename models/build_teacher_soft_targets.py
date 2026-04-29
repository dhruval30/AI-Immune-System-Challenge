#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from roberta_dapt_pipeline_common import (
    extract_primary_metric_bundle,
    load_jsonl,
    make_occurrence_keys,
    normalize_label,
    normalize_texts,
    probability_entropy,
    safe_read_json,
    validate_competition_inputs,
)

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_teacher_targets"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

TRAIN_SOFT_TARGETS_PATH = OUTPUT_DIR / "roberta_teacher_soft_targets_train.csv"
TEST_SOFT_TARGETS_PATH = OUTPUT_DIR / "roberta_teacher_soft_targets_test.csv"
METRICS_PATH = OUTPUT_DIR / "roberta_teacher_soft_targets_metrics.json"
NOTES_PATH = OUTPUT_DIR / "roberta_teacher_soft_targets_notes.md"


@dataclass(frozen=True)
class TeacherSource:
    name: str
    manual_weight: float
    train_path: Path | None = None
    test_path: Path | None = None
    metrics_path: Path | None = None


DEFAULT_SOURCES = [
    TeacherSource(
        name="roberta_base_cv_v2",
        manual_weight=1.00,
        train_path=ROOT_DIR / "outputs" / "roberta_base_cv_v2" / "roberta_base_cv_v2_oof_predictions.csv",
        test_path=ROOT_DIR / "outputs" / "roberta_base_cv_v2" / "roberta_base_cv_v2_test_probabilities.csv",
        metrics_path=ROOT_DIR / "outputs" / "roberta_base_cv_v2" / "roberta_base_cv_v2_metrics.json",
    ),
    TeacherSource(
        name="roberta_base_cv",
        manual_weight=0.65,
        train_path=ROOT_DIR / "outputs" / "roberta_base_cv" / "roberta_base_cv_oof_predictions.csv",
        test_path=ROOT_DIR / "outputs" / "roberta_base_cv" / "roberta_base_cv_test_probabilities.csv",
        metrics_path=ROOT_DIR / "outputs" / "roberta_base_cv" / "roberta_base_cv_metrics.json",
    ),
    TeacherSource(
        name="electra_base",
        manual_weight=0.90,
        train_path=ROOT_DIR / "outputs" / "electra_base" / "electra_base_val_predictions.csv",
        test_path=ROOT_DIR / "outputs" / "electra_base" / "electra_base_test_probabilities.csv",
        metrics_path=ROOT_DIR / "outputs" / "electra_base" / "electra_base_metrics.json",
    ),
    TeacherSource(
        name="modernbert_base",
        manual_weight=0.85,
        train_path=ROOT_DIR / "outputs" / "modernbert_base" / "modernbert_base_val_predictions.csv",
        test_path=ROOT_DIR / "outputs" / "modernbert_base" / "modernbert_base_test_probabilities.csv",
        metrics_path=ROOT_DIR / "outputs" / "modernbert_base" / "modernbert_base_metrics.json",
    ),
    TeacherSource(
        name="roberta_base_single_split",
        manual_weight=0.45,
        train_path=ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv",
        test_path=ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_test_probabilities.csv",
        metrics_path=ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_metrics.json",
    ),
    TeacherSource(
        name="baseline_nli_roberta_logreg_cv",
        manual_weight=0.35,
        train_path=ROOT_DIR
        / "outputs"
        / "baseline_nli_roberta_logreg_cv"
        / "baseline_nli_roberta_logreg_cv_oof_predictions.csv",
        test_path=None,
        metrics_path=ROOT_DIR
        / "outputs"
        / "baseline_nli_roberta_logreg_cv"
        / "baseline_nli_roberta_logreg_cv_metrics.json",
    ),
    TeacherSource(
        name="baseline_minilm_logreg_cv",
        manual_weight=0.20,
        train_path=ROOT_DIR
        / "outputs"
        / "baseline_minilm_logreg_cv"
        / "baseline_minilm_logreg_cv_oof_predictions.csv",
        test_path=None,
        metrics_path=ROOT_DIR
        / "outputs"
        / "baseline_minilm_logreg_cv"
        / "baseline_minilm_logreg_cv_metrics.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher soft targets for honest RoBERTa distillation.")
    parser.add_argument("--entropy-weight", type=float, default=0.60)
    parser.add_argument("--disagreement-weight", type=float, default=0.60)
    parser.add_argument(
        "--exclude-partial-train-sources",
        action="store_true",
        help="Drop train sources that do not cover the full labeled train set.",
    )
    parser.add_argument(
        "--min-partial-coverage",
        type=float,
        default=0.05,
        help="Ignore train sources whose train-row coverage falls below this fraction.",
    )
    return parser.parse_args()


def infer_prob_column(df: pd.DataFrame) -> str:
    for column in ("oof_prob_TRUE_mean", "oof_prob_TRUE", "pred_prob_TRUE", "teacher_prob_TRUE", "pseudo_prob_TRUE"):
        if column in df.columns:
            return column
    raise ValueError(f"Unable to infer probability column from columns: {df.columns.tolist()}")


def infer_label_column(df: pd.DataFrame) -> str | None:
    for column in ("true_label", "label"):
        if column in df.columns:
            return column
    return None


def load_quality_weight(source: TeacherSource) -> tuple[float, dict]:
    if source.metrics_path is None or not source.metrics_path.exists():
        return source.manual_weight, {"manual_weight": source.manual_weight, "quality_weight": 1.0}

    metrics = safe_read_json(source.metrics_path)
    primary = extract_primary_metric_bundle(metrics)

    metric_values = []
    for key in ("f1", "roc_auc", "precision", "recall"):
        value = primary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metric_values.append(float(value))

    quality_weight = float(np.mean(metric_values[:2])) if metric_values else 1.0
    effective_weight = float(source.manual_weight * quality_weight)
    return effective_weight, {
        "manual_weight": source.manual_weight,
        "quality_weight": quality_weight,
        "primary_metrics": {key: primary.get(key) for key in ("f1", "roc_auc", "precision", "recall", "threshold")},
    }


def align_source_probabilities(
    source: TeacherSource,
    prediction_path: Path,
    target_keys: list[str],
    is_train: bool,
) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(prediction_path)
    if "text" not in df.columns:
        raise ValueError(f"{prediction_path} is missing the required 'text' column.")

    label_column = infer_label_column(df) if is_train else None
    prob_column = infer_prob_column(df)
    prediction_keys = make_occurrence_keys(
        texts=df["text"].tolist(),
        labels=df[label_column].tolist() if label_column else None,
    )

    target_lookup = {key: idx for idx, key in enumerate(target_keys)}
    aligned = np.full(len(target_keys), np.nan, dtype=np.float64)

    matched = 0
    duplicate_rows = 0
    unmatched_rows = 0
    for key, prob in zip(prediction_keys, df[prob_column].tolist()):
        if pd.isna(prob):
            continue
        target_idx = target_lookup.get(key)
        if target_idx is None:
            unmatched_rows += 1
            continue
        if np.isfinite(aligned[target_idx]):
            duplicate_rows += 1
            continue
        aligned[target_idx] = float(prob)
        matched += 1

    report = {
        "path": str(prediction_path),
        "rows": int(len(df)),
        "matched_rows": int(matched),
        "coverage": float(matched / max(len(target_keys), 1)),
        "duplicate_rows_ignored": int(duplicate_rows),
        "unmatched_rows": int(unmatched_rows),
        "probability_column": prob_column,
        "label_column": label_column,
    }
    return aligned, report


def summarize_probabilities(
    probability_matrix: np.ndarray,
    weight_vector: np.ndarray,
    entropy_weight: float,
    disagreement_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_count = probability_matrix.shape[0]
    agg_prob = np.full(row_count, np.nan, dtype=np.float64)
    teacher_count = np.zeros(row_count, dtype=np.int64)
    teacher_std = np.full(row_count, np.nan, dtype=np.float64)
    teacher_entropy = np.full(row_count, np.nan, dtype=np.float64)
    agreement_score = np.full(row_count, np.nan, dtype=np.float64)
    sample_weight = np.ones(row_count, dtype=np.float64)

    for row_idx in range(row_count):
        values = probability_matrix[row_idx]
        available_mask = np.isfinite(values)
        if not np.any(available_mask):
            continue

        row_values = values[available_mask]
        row_weights = weight_vector[available_mask]
        row_weights = row_weights / row_weights.sum()

        mean_prob = float(np.average(row_values, weights=row_weights))
        variance = float(np.average((row_values - mean_prob) ** 2, weights=row_weights))
        std_prob = math.sqrt(max(variance, 0.0))
        normalized_entropy = float(probability_entropy(mean_prob) / math.log(2.0))
        disagreement_score = min(std_prob / 0.25, 1.0)
        agreement = 1.0 - disagreement_score

        agg_prob[row_idx] = mean_prob
        teacher_count[row_idx] = int(row_values.size)
        teacher_std[row_idx] = std_prob
        teacher_entropy[row_idx] = normalized_entropy
        agreement_score[row_idx] = agreement
        sample_weight[row_idx] = float(1.0 + (entropy_weight * normalized_entropy) + (disagreement_weight * disagreement_score))

    return agg_prob, teacher_count, teacher_std, teacher_entropy, agreement_score, sample_weight


def write_notes(metrics: dict) -> None:
    source_lines = []
    for source in metrics["sources"]:
        source_lines.append(
            f"- `{source['name']}`: effective weight `{source['effective_weight']:.4f}`, "
            f"train coverage `{source['train'].get('coverage', 0.0):.4f}`, "
            f"test coverage `{source['test'].get('coverage', 0.0):.4f}`"
        )

    notes = f"""# Teacher Soft Target Notes

## Purpose

This script builds honest teacher soft targets for train-time distillation.
It aligns existing OOF and held-out validation artifacts back to the canonical train rows
and builds an aggregated teacher probability plus disagreement-aware sample weights.

## Configuration

- Entropy weight: `{metrics['config']['entropy_weight']}`
- Disagreement weight: `{metrics['config']['disagreement_weight']}`
- Exclude partial train sources: `{metrics['config']['exclude_partial_train_sources']}`
- Minimum partial coverage: `{metrics['config']['min_partial_coverage']}`

## Sources

{chr(10).join(source_lines)}

## Output Shapes

- Train rows: `{metrics['data']['train_rows']}`
- Test rows: `{metrics['data']['test_rows']}`
- Train rows with at least one teacher: `{metrics['aggregates']['train_rows_with_teacher']}`
- Test rows with at least one teacher: `{metrics['aggregates']['test_rows_with_teacher']}`

## Outputs

- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_train.csv`
- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_test.csv`
- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_metrics.json`
- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_notes.md`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Building teacher soft targets ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading canonical competition rows ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_competition_inputs(train_df, test_df, solution_df)

    train_texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    train_labels = [normalize_label(value) for value in train_df["label"].tolist()]

    train_keys = make_occurrence_keys(train_texts, train_labels)
    test_keys = make_occurrence_keys(test_texts)

    source_reports = []
    train_columns: dict[str, np.ndarray] = {}
    test_columns: dict[str, np.ndarray] = {}
    train_weight_by_column: dict[str, float] = {}
    test_weight_by_column: dict[str, float] = {}

    print("\n=== Aligning teacher sources ===")
    for source in DEFAULT_SOURCES:
        effective_weight, weight_meta = load_quality_weight(source)
        source_report = {
            "name": source.name,
            "effective_weight": effective_weight,
            "weight_meta": weight_meta,
            "train": {},
            "test": {},
        }

        if source.train_path is not None and source.train_path.exists():
            train_probs, train_report = align_source_probabilities(source, source.train_path, train_keys, is_train=True)
            source_report["train"] = train_report
            coverage = train_report["coverage"]
            should_include_train = coverage > 0.0
            if args.exclude_partial_train_sources and coverage < 0.999:
                should_include_train = False
            if coverage < args.min_partial_coverage:
                should_include_train = False
            if should_include_train:
                column_name = f"{source.name}_train_prob_TRUE"
                train_columns[column_name] = train_probs
                train_weight_by_column[column_name] = effective_weight
        else:
            source_report["train"] = {"path": str(source.train_path) if source.train_path else None, "missing": True}

        if source.test_path is not None and source.test_path.exists():
            test_probs, test_report = align_source_probabilities(source, source.test_path, test_keys, is_train=False)
            source_report["test"] = test_report
            if test_report["coverage"] > 0.0:
                column_name = f"{source.name}_test_prob_TRUE"
                test_columns[column_name] = test_probs
                test_weight_by_column[column_name] = effective_weight
        else:
            source_report["test"] = {"path": str(source.test_path) if source.test_path else None, "missing": True}

        source_reports.append(source_report)

    if not train_columns:
        raise RuntimeError("No usable teacher train sources were found. Build OOF/validation artifacts first.")
    if not test_columns:
        raise RuntimeError("No usable teacher test sources were found. Build teacher test probabilities first.")

    train_matrix_columns = list(train_columns)
    test_matrix_columns = list(test_columns)
    train_matrix = np.column_stack([train_columns[column] for column in train_matrix_columns])
    test_matrix = np.column_stack([test_columns[column] for column in test_matrix_columns])
    train_weight_vector = np.asarray([train_weight_by_column[column] for column in train_matrix_columns], dtype=np.float64)
    test_weight_vector = np.asarray([test_weight_by_column[column] for column in test_matrix_columns], dtype=np.float64)

    print("\n=== Aggregating train soft targets ===")
    (
        train_teacher_prob,
        train_teacher_count,
        train_teacher_std,
        train_teacher_entropy,
        train_agreement_score,
        train_sample_weight,
    ) = summarize_probabilities(
        probability_matrix=train_matrix,
        weight_vector=train_weight_vector,
        entropy_weight=args.entropy_weight,
        disagreement_weight=args.disagreement_weight,
    )

    print("\n=== Aggregating test soft targets ===")
    (
        test_teacher_prob,
        test_teacher_count,
        test_teacher_std,
        test_teacher_entropy,
        test_agreement_score,
        test_sample_weight,
    ) = summarize_probabilities(
        probability_matrix=test_matrix,
        weight_vector=test_weight_vector,
        entropy_weight=args.entropy_weight,
        disagreement_weight=args.disagreement_weight,
    )

    train_output = pd.DataFrame(
        {
            "row_index": np.arange(len(train_texts), dtype=np.int64),
            "text": train_texts,
            "label": train_labels,
            "label_int": np.asarray([1 if label == "TRUE" else 0 for label in train_labels], dtype=np.int64),
            "teacher_prob_TRUE": train_teacher_prob,
            "teacher_pred_label": np.where(train_teacher_prob >= 0.5, "TRUE", "FALSE"),
            "teacher_count": train_teacher_count,
            "teacher_prob_std": train_teacher_std,
            "teacher_entropy": train_teacher_entropy,
            "agreement_score": train_agreement_score,
            "disagreement_score": 1.0 - np.nan_to_num(train_agreement_score, nan=0.0),
            "sample_weight": train_sample_weight,
        }
    )
    for column in train_matrix_columns:
        train_output[column] = train_columns[column]

    test_output = pd.DataFrame(
        {
            "test_row_index": np.arange(len(test_texts), dtype=np.int64),
            "text": test_texts,
            "teacher_prob_TRUE": test_teacher_prob,
            "teacher_pred_label": np.where(test_teacher_prob >= 0.5, "TRUE", "FALSE"),
            "teacher_count": test_teacher_count,
            "teacher_prob_std": test_teacher_std,
            "teacher_entropy": test_teacher_entropy,
            "agreement_score": test_agreement_score,
            "disagreement_score": 1.0 - np.nan_to_num(test_agreement_score, nan=0.0),
            "sample_weight": test_sample_weight,
        }
    )
    for column in test_matrix_columns:
        test_output[column] = test_columns[column]

    train_output.to_csv(TRAIN_SOFT_TARGETS_PATH, index=False)
    test_output.to_csv(TEST_SOFT_TARGETS_PATH, index=False)

    metrics = {
        "config": {
            "entropy_weight": args.entropy_weight,
            "disagreement_weight": args.disagreement_weight,
            "exclude_partial_train_sources": args.exclude_partial_train_sources,
            "min_partial_coverage": args.min_partial_coverage,
        },
        "data": {
            "train_rows": len(train_texts),
            "test_rows": len(test_texts),
        },
        "sources": source_reports,
        "aggregates": {
            "train_source_columns": train_matrix_columns,
            "test_source_columns": test_matrix_columns,
            "train_rows_with_teacher": int(np.isfinite(train_teacher_prob).sum()),
            "test_rows_with_teacher": int(np.isfinite(test_teacher_prob).sum()),
            "train_teacher_count_distribution": pd.Series(train_teacher_count).value_counts().sort_index().to_dict(),
            "test_teacher_count_distribution": pd.Series(test_teacher_count).value_counts().sort_index().to_dict(),
            "train_teacher_prob_summary": {
                "mean": float(np.nanmean(train_teacher_prob)),
                "std": float(np.nanstd(train_teacher_prob)),
            },
            "test_teacher_prob_summary": {
                "mean": float(np.nanmean(test_teacher_prob)),
                "std": float(np.nanstd(test_teacher_prob)),
            },
        },
        "paths": {
            "train_soft_targets": str(TRAIN_SOFT_TARGETS_PATH),
            "test_soft_targets": str(TEST_SOFT_TARGETS_PATH),
            "metrics": str(METRICS_PATH),
            "notes": str(NOTES_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_notes(metrics)

    print(f"Saved train teacher soft targets: {TRAIN_SOFT_TARGETS_PATH}")
    print(f"Saved test teacher soft targets: {TEST_SOFT_TARGETS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
