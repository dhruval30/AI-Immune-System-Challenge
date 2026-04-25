#!/usr/bin/env python3
# pip install pandas numpy
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
BLENDS_DIR = OUTPUTS_DIR / "blends"

SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
ROBERTA_SUBMISSION_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_submission.csv"
MODERNBERT_SUBMISSION_PATH = OUTPUTS_DIR / "modernbert_base" / "modernbert_base_submission.csv"
ENSEMBLE_OR_PATH = BLENDS_DIR / "roberta_modernbert_ensemble_or.csv"
ENSEMBLE_AND_PATH = BLENDS_DIR / "roberta_modernbert_ensemble_and.csv"
ENSEMBLE_ROBERTA_TIE_PATH = BLENDS_DIR / "roberta_modernbert_ensemble_roberta_tie.csv"
ENSEMBLE_REPORT_PATH = BLENDS_DIR / "roberta_modernbert_ensemble_report.json"


def normalize_label_series(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return np.where(series.to_numpy(), "TRUE", "FALSE")

    normalized = series.astype(str).str.strip().str.upper()
    mapped = normalized.map(
        {
            "TRUE": "TRUE",
            "FALSE": "FALSE",
            "T": "TRUE",
            "F": "FALSE",
            "1": "TRUE",
            "0": "FALSE",
        }
    )
    if mapped.isna().any():
        bad_examples = normalized[mapped.isna()].head(5).tolist()
        raise ValueError(f"{name} contains invalid labels. Examples: {bad_examples}")

    return mapped.to_numpy()


def make_submission(solution_df: pd.DataFrame, labels_true_false: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels_true_false})

    out_df = solution_df.copy()
    out_df["label"] = labels_true_false
    return out_df[solution_df.columns.tolist()]


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    BLENDS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading files ===")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    roberta_df = pd.read_csv(ROBERTA_SUBMISSION_PATH)
    modernbert_df = pd.read_csv(MODERNBERT_SUBMISSION_PATH)

    if roberta_df.columns.tolist() != solution_df.columns.tolist():
        raise ValueError("RoBERTa submission columns do not match solution_format columns.")
    if modernbert_df.columns.tolist() != solution_df.columns.tolist():
        raise ValueError("ModernBERT submission columns do not match solution_format columns.")
    if len(roberta_df) != len(solution_df):
        raise ValueError("RoBERTa submission row count does not match solution_format.")
    if len(modernbert_df) != len(solution_df):
        raise ValueError("ModernBERT submission row count does not match solution_format.")

    print("\n=== Building ensemble ===")
    roberta_labels = normalize_label_series(roberta_df["label"], "RoBERTa submission")
    modernbert_labels = normalize_label_series(modernbert_df["label"], "ModernBERT submission")

    agree_mask = roberta_labels == modernbert_labels
    disagree_mask = ~agree_mask

    roberta_true = roberta_labels == "TRUE"
    modernbert_true = modernbert_labels == "TRUE"

    # OR: TRUE if either model predicts TRUE.
    ensemble_or = np.where(roberta_true | modernbert_true, "TRUE", "FALSE")
    # AND: TRUE only if both models predict TRUE.
    ensemble_and = np.where(roberta_true & modernbert_true, "TRUE", "FALSE")
    # Tie-break toward RoBERTa (kept for reference; often close to RoBERTa submission).
    ensemble_roberta_tie = np.where(agree_mask, roberta_labels, roberta_labels)

    def pred_dist(labels: np.ndarray) -> dict:
        return {
            "pred_FALSE": int((labels == "FALSE").sum()),
            "pred_TRUE": int((labels == "TRUE").sum()),
        }

    agree_count = int(agree_mask.sum())
    disagree_count = int(disagree_mask.sum())

    submission_or = make_submission(solution_df, ensemble_or)
    submission_and = make_submission(solution_df, ensemble_and)
    submission_roberta_tie = make_submission(solution_df, ensemble_roberta_tie)

    submission_or.to_csv(ENSEMBLE_OR_PATH, index=False)
    submission_and.to_csv(ENSEMBLE_AND_PATH, index=False)
    submission_roberta_tie.to_csv(ENSEMBLE_ROBERTA_TIE_PATH, index=False)

    report = {
        "strategy": "hard_label_ensemble_variants",
        "files": {
            "roberta_submission": str(ROBERTA_SUBMISSION_PATH),
            "modernbert_submission": str(MODERNBERT_SUBMISSION_PATH),
            "solution_format": str(SOLUTION_FORMAT_PATH),
            "ensemble_or_submission": str(ENSEMBLE_OR_PATH),
            "ensemble_and_submission": str(ENSEMBLE_AND_PATH),
            "ensemble_roberta_tie_submission": str(ENSEMBLE_ROBERTA_TIE_PATH),
        },
        "row_count": int(len(solution_df)),
        "agreement": {
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "agree_ratio": float(agree_count / len(solution_df)),
        },
        "prediction_distributions": {
            "or_rule": pred_dist(ensemble_or),
            "and_rule": pred_dist(ensemble_and),
            "roberta_tie_rule": pred_dist(ensemble_roberta_tie),
        },
    }
    ENSEMBLE_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved ensemble submission (OR): {ENSEMBLE_OR_PATH}")
    print(f"Saved ensemble submission (AND): {ENSEMBLE_AND_PATH}")
    print(f"Saved ensemble submission (RoBERTa tie): {ENSEMBLE_ROBERTA_TIE_PATH}")
    print(f"Saved ensemble report: {ENSEMBLE_REPORT_PATH}")
    print(f"Agreement: {agree_count} | Disagreement: {disagree_count}")
    print(f"OR distribution -> {pred_dist(ensemble_or)}")
    print(f"AND distribution -> {pred_dist(ensemble_and)}")
    print(f"RoBERTa-tie distribution -> {pred_dist(ensemble_roberta_tie)}")


if __name__ == "__main__":
    main()
