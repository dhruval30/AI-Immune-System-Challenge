#!/usr/bin/env python3
"""
Create RoBERTa-anchored ensemble candidate submissions from existing output artifacts.

Dependency install command:
pip install pandas numpy

No model training is performed. This script combines existing test probabilities/scores,
keeps roberta_base as the anchor, and generates a small set of fixed-TRUE-count submissions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "mega_ensemble_ranker"
SOLUTION_PATH = ROOT_DIR / "data" / "solution_format.csv"

ANCHOR_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_test_probabilities.csv"
ANCHOR_SUBMISSION_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_submission.csv"

SOURCE_SPECS = [
    {
        "name": "roberta_base",
        "path": ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.60,
        "required": True,
    },
    {
        "name": "roberta_label_style_calibrated",
        "path": ROOT_DIR
        / "outputs"
        / "roberta_label_style_calibrator"
        / "roberta_label_style_calibrator_test_scores.csv",
        "column": "calibrated_prob_TRUE",
        "kind": "probability",
        "weight": 0.10,
        "required": False,
    },
    {
        "name": "roberta_label_style_rankblend_0p90",
        "path": ROOT_DIR
        / "outputs"
        / "roberta_label_style_calibrator"
        / "roberta_label_style_calibrator_test_scores.csv",
        "column": "rank_blend_roberta_0p90",
        "kind": "rank_score",
        "weight": 0.08,
        "required": False,
    },
    {
        "name": "modernbert_lora_v2",
        "path": ROOT_DIR / "outputs" / "modernbert_lora_v2" / "modernbert_lora_v2_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.07,
        "required": False,
    },
    {
        "name": "modernbert_base",
        "path": ROOT_DIR / "outputs" / "modernbert_base" / "modernbert_base_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.05,
        "required": False,
    },
    {
        "name": "roberta_base_cv",
        "path": ROOT_DIR / "outputs" / "roberta_base_cv" / "roberta_base_cv_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.04,
        "required": False,
    },
    {
        "name": "roberta_base_cv_v2",
        "path": ROOT_DIR / "outputs" / "roberta_base_cv_v2" / "roberta_base_cv_v2_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.04,
        "required": False,
    },
    {
        "name": "roberta_style_features",
        "path": ROOT_DIR / "outputs" / "roberta_style_features" / "roberta_style_features_test_probabilities.csv",
        "column": "pred_prob_TRUE",
        "kind": "probability",
        "weight": 0.02,
        "required": False,
    },
]

# Keep the search small. The best observed candidates lived around 577-580 TRUE.
TRUE_COUNTS = [570, 574, 577, 578, 580, 585, 590]

# Multiple controlled recipes, all anchored strongly to roberta_base.
RECIPES = [
    {
        "name": "anchor_rank_light",
        "source_weight_multiplier": {
            "roberta_base": 0.80,
            "roberta_label_style_calibrated": 0.08,
            "roberta_label_style_rankblend_0p90": 0.04,
            "modernbert_lora_v2": 0.04,
            "modernbert_base": 0.02,
            "roberta_base_cv": 0.01,
            "roberta_base_cv_v2": 0.01,
            "roberta_style_features": 0.00,
        },
    },
    {
        "name": "anchor_rank_balanced",
        "source_weight_multiplier": {
            "roberta_base": 0.70,
            "roberta_label_style_calibrated": 0.10,
            "roberta_label_style_rankblend_0p90": 0.07,
            "modernbert_lora_v2": 0.06,
            "modernbert_base": 0.03,
            "roberta_base_cv": 0.02,
            "roberta_base_cv_v2": 0.02,
            "roberta_style_features": 0.00,
        },
    },
    {
        "name": "anchor_rank_calibrator",
        "source_weight_multiplier": {
            "roberta_base": 0.78,
            "roberta_label_style_calibrated": 0.14,
            "roberta_label_style_rankblend_0p90": 0.08,
            "modernbert_lora_v2": 0.00,
            "modernbert_base": 0.00,
            "roberta_base_cv": 0.00,
            "roberta_base_cv_v2": 0.00,
            "roberta_style_features": 0.00,
        },
    },
    {
        "name": "anchor_rank_modernbert",
        "source_weight_multiplier": {
            "roberta_base": 0.82,
            "roberta_label_style_calibrated": 0.00,
            "roberta_label_style_rankblend_0p90": 0.00,
            "modernbert_lora_v2": 0.10,
            "modernbert_base": 0.05,
            "roberta_base_cv": 0.02,
            "roberta_base_cv_v2": 0.01,
            "roberta_style_features": 0.00,
        },
    },
]

# Known bad branches intentionally excluded:
# - roberta_large: LB 0.76803119
# - roberta_pseudolabel: LB 0.79125249
# - baseline_minilm_logreg_cv: LB 0.51649928
# - hard OR/AND ensembles: underperformed roberta_base


def load_solution() -> pd.DataFrame:
    solution_df = pd.read_csv(SOLUTION_PATH)
    if "label" not in solution_df.columns:
        raise ValueError(f"solution_format.csv must contain label column; found {list(solution_df.columns)}")
    return solution_df


def load_source(spec: dict[str, Any], expected_rows: int) -> pd.DataFrame | None:
    path = Path(spec["path"])
    if not path.exists():
        if spec.get("required", False):
            raise FileNotFoundError(f"Required source not found: {path}")
        print(f"Skipping missing optional source: {spec['name']} -> {path}")
        return None

    df = pd.read_csv(path)
    if len(df) != expected_rows:
        raise ValueError(f"{spec['name']} row count {len(df)} does not match expected {expected_rows}: {path}")
    if spec["column"] not in df.columns:
        raise ValueError(f"{spec['name']} missing column {spec['column']!r}; found {list(df.columns)}")

    if "text" not in df.columns:
        df["text"] = ""

    source_df = pd.DataFrame(
        {
            "row_id": np.arange(expected_rows),
            "text": df["text"].astype(str),
            spec["name"]: pd.to_numeric(df[spec["column"]], errors="coerce").fillna(0.0).astype(float),
        }
    )
    return source_df


def percentile_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def normalize_weights(weights: dict[str, float], available_sources: list[str]) -> dict[str, float]:
    filtered = {name: float(weight) for name, weight in weights.items() if name in available_sources and weight > 0}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("Recipe has no positive weights after filtering available sources.")
    return {name: weight / total for name, weight in filtered.items()}


def make_submission(solution_df: pd.DataFrame, true_mask: np.ndarray) -> pd.DataFrame:
    labels = np.where(true_mask, "TRUE", "FALSE")
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels})
    submission = solution_df.copy()
    submission["label"] = labels
    return submission[solution_df.columns.tolist()]


def load_anchor_submission(expected_rows: int) -> pd.Series | None:
    if not ANCHOR_SUBMISSION_PATH.exists():
        return None
    df = pd.read_csv(ANCHOR_SUBMISSION_PATH)
    if len(df) != expected_rows or "label" not in df.columns:
        return None
    return df["label"].astype(str).str.upper()


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Mega Ensemble Ranker ===")
    print("No training is performed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solution_df = load_solution()
    expected_rows = len(solution_df)
    print(f"Solution rows: {expected_rows}")

    loaded_sources: list[pd.DataFrame] = []
    loaded_specs: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        source_df = load_source(spec, expected_rows)
        if source_df is not None:
            loaded_sources.append(source_df)
            loaded_specs.append(spec)
            print(f"Loaded source: {spec['name']} from {spec['path']}")

    if not loaded_sources:
        raise RuntimeError("No sources loaded.")

    merged = loaded_sources[0]
    for source_df in loaded_sources[1:]:
        # Align test artifacts by row order. Text strings can differ slightly because
        # CSV quoting/newlines are not always preserved identically across scripts.
        source_name = [col for col in source_df.columns if col not in {"row_id", "text"}][0]
        merged = merged.merge(source_df[["row_id", source_name]], on="row_id", how="inner")

    if len(merged) != expected_rows:
        raise ValueError(f"Merged source row count {len(merged)} does not match expected {expected_rows}.")

    available_sources = [spec["name"] for spec in loaded_specs]
    if "roberta_base" not in available_sources:
        raise RuntimeError("roberta_base is required as anchor.")

    rank_columns = []
    for name in available_sources:
        rank_col = f"{name}_rank"
        merged[rank_col] = percentile_rank(merged[name].to_numpy(dtype=float))
        rank_columns.append(rank_col)

    anchor_probs = merged["roberta_base"].to_numpy(dtype=float)
    anchor_ranks = merged["roberta_base_rank"].to_numpy(dtype=float)
    anchor_submission = load_anchor_submission(expected_rows)

    manifest_rows: list[dict[str, Any]] = []

    source_summary = []
    for spec in loaded_specs:
        name = spec["name"]
        values = merged[name].to_numpy(dtype=float)
        source_summary.append(
            {
                "source": name,
                "path": str(spec["path"]),
                "column": spec["column"],
                "kind": spec["kind"],
                "min": float(np.min(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)),
                "rows": int(len(values)),
            }
        )

    for recipe in RECIPES:
        weights = normalize_weights(recipe["source_weight_multiplier"], available_sources)
        score = np.zeros(expected_rows, dtype=float)
        for name, weight in weights.items():
            score += weight * merged[f"{name}_rank"].to_numpy(dtype=float)

        score_col = f"score_{recipe['name']}"
        merged[score_col] = score
        score_order = np.argsort(-score)

        for true_count in TRUE_COUNTS:
            true_mask = np.zeros(expected_rows, dtype=bool)
            true_mask[score_order[:true_count]] = True
            submission = make_submission(solution_df, true_mask)

            filename = f"mega_{recipe['name']}_top{true_count}.csv"
            output_path = OUTPUT_DIR / filename
            submission.to_csv(output_path, index=False)

            selected_anchor_probs = anchor_probs[true_mask]
            selected_anchor_ranks = anchor_ranks[true_mask]
            anchor_true_count = None
            changed_from_anchor = None
            if anchor_submission is not None:
                anchor_true_mask = anchor_submission.eq("TRUE").to_numpy()
                anchor_true_count = int(anchor_true_mask.sum())
                changed_from_anchor = int(np.not_equal(anchor_true_mask, true_mask).sum())

            manifest_rows.append(
                {
                    "filename": filename,
                    "path": str(output_path),
                    "recipe": recipe["name"],
                    "true_count": int(true_count),
                    "false_count": int(expected_rows - true_count),
                    "anchor_true_count": anchor_true_count,
                    "changed_rows_vs_roberta_base_submission": changed_from_anchor,
                    "mean_anchor_prob_selected_true": float(np.mean(selected_anchor_probs)),
                    "min_anchor_prob_selected_true": float(np.min(selected_anchor_probs)),
                    "mean_anchor_rank_selected_true": float(np.mean(selected_anchor_ranks)),
                    "weights": json.dumps(weights, sort_keys=True),
                }
            )
            print(f"Saved {filename}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "mega_ensemble_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    score_path = OUTPUT_DIR / "mega_ensemble_scores.csv"
    score_columns = ["row_id", "text"] + available_sources + rank_columns + [
        f"score_{recipe['name']}" for recipe in RECIPES
    ]
    merged[score_columns].to_csv(score_path, index=False)

    report = {
        "script": "training_scripts/mega_ensemble_ranker.py",
        "training_performed": False,
        "anchor": "outputs/roberta_base/roberta_base_test_probabilities.csv",
        "excluded_by_default": [
            "roberta_large",
            "roberta_pseudolabel",
            "baseline_minilm_logreg_cv",
            "hard_label_or_and_ensembles",
        ],
        "true_counts": TRUE_COUNTS,
        "recipes": RECIPES,
        "source_summary": source_summary,
        "manifest": str(manifest_path),
        "scores": str(score_path),
        "recommendation": (
            "Start with candidates around top577/top578 from anchor_rank_light or anchor_rank_calibrator. "
            "These keep the known-good RoBERTa TRUE-count region while swapping only borderline rows."
        ),
    }
    report_path = OUTPUT_DIR / "mega_ensemble_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    notes = f"""# Mega Ensemble Ranker Notes

No training was performed. This script creates fixed-TRUE-count submissions by combining existing prediction artifacts.

## Strategy

- Keep `roberta_base` as the anchor.
- Combine model rankings rather than raw probabilities, because different models are calibrated differently.
- Exclude known bad branches by default: `roberta_large`, `roberta_pseudolabel`, `baseline_minilm_logreg_cv`, and hard OR/AND ensembles.
- Generate candidates around the known good TRUE-count region.

## First Candidates To Try

If using submissions from this folder, start with:

- `mega_anchor_rank_light_top577.csv`
- `mega_anchor_rank_light_top578.csv`
- `mega_anchor_rank_calibrator_top577.csv`
- `mega_anchor_rank_calibrator_top578.csv`

Do not submit all candidates blindly. Check the manifest first.

## Outputs

- `{manifest_path.relative_to(ROOT_DIR)}`
- `{score_path.relative_to(ROOT_DIR)}`
- `{report_path.relative_to(ROOT_DIR)}`
"""
    notes_path = OUTPUT_DIR / "mega_ensemble_notes.md"
    notes_path.write_text(notes, encoding="utf-8")

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved scores: {score_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved notes: {notes_path}")


if __name__ == "__main__":
    main()
