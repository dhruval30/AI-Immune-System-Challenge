#!/usr/bin/env python3
"""
Blend the two best RoBERTa hard-weighted GPU runs.

Dependency install command:
pip install pandas numpy

No training is performed. This script blends probabilities from:

1. Previous best hard-weighted RoBERTa:
   gpu/outputs/roberta_hard_weighted_gpu_b64_final/

2. New best bootstrapped hard-weighted RoBERTa:
   gpu/outputs/roberta_hard_weighted_bootstrap_gpu/

The current best public LB is the bootstrapped fixed-threshold 0.32 submission:
0.93718166. This script creates only five probability-blend candidates, all at
threshold 0.32, so submission volume stays controlled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_hard_bootstrap_blend"

SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

HARD_PROB_PATH = (
    ROOT_DIR
    / "outputs"
    / "roberta_hard_weighted_gpu_b64_final"
    / "roberta_hard_weighted_gpu_b64_test_probabilities.csv"
)
BOOTSTRAP_PROB_PATH = (
    ROOT_DIR
    / "outputs"
    / "roberta_hard_weighted_bootstrap_gpu"
    / "roberta_hard_weighted_bootstrap_gpu_test_probabilities.csv"
)

PROB_COL = "pred_prob_TRUE"
THRESHOLD = 0.32

# Keep this tight. The bootstrapped model is the stronger LB result, so it gets
# at least half the weight in every candidate.
BOOTSTRAP_WEIGHTS = [0.50, 0.60, 0.70, 0.80, 0.90]


def normalize_text(value: object) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_probability_file(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} probability file: {path}")
    df = pd.read_csv(path)
    required = {"text", PROB_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} probability file is missing columns: {sorted(missing)}")
    df = df.copy()
    df["normalized_text"] = df["text"].map(normalize_text)
    df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="coerce")
    if df[PROB_COL].isna().any():
        raise ValueError(f"{name} probability file contains missing/non-numeric probabilities.")
    if not ((df[PROB_COL] >= 0.0) & (df[PROB_COL] <= 1.0)).all():
        raise ValueError(f"{name} probability file has probabilities outside [0, 1].")
    return df


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = labels
    return submission_df[solution_df.columns.tolist()]


def prediction_distribution(labels: np.ndarray) -> dict[str, int]:
    return {
        "pred_FALSE": int((labels == "FALSE").sum()),
        "pred_TRUE": int((labels == "TRUE").sum()),
    }


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Hard + Bootstrap Probability Blend ===")
    print("No training is performed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    hard_df = load_probability_file(HARD_PROB_PATH, "hard-weighted")
    bootstrap_df = load_probability_file(BOOTSTRAP_PROB_PATH, "bootstrapped")

    expected_rows = len(solution_df)
    if len(hard_df) != expected_rows or len(bootstrap_df) != expected_rows:
        raise ValueError(
            "Probability row counts must match solution format. "
            f"solution={expected_rows}, hard={len(hard_df)}, bootstrap={len(bootstrap_df)}"
        )
    if not hard_df["normalized_text"].equals(bootstrap_df["normalized_text"]):
        mismatch = hard_df["normalized_text"] != bootstrap_df["normalized_text"]
        first_bad = int(np.flatnonzero(mismatch.to_numpy())[0])
        raise ValueError(f"Probability files are not text-aligned. First mismatch row: {first_bad}")

    hard_prob = hard_df[PROB_COL].to_numpy(dtype=np.float64)
    bootstrap_prob = bootstrap_df[PROB_COL].to_numpy(dtype=np.float64)

    manifest = []
    for bootstrap_weight in BOOTSTRAP_WEIGHTS:
        hard_weight = 1.0 - bootstrap_weight
        blended_prob = hard_weight * hard_prob + bootstrap_weight * bootstrap_prob
        pred = blended_prob >= THRESHOLD
        labels = np.where(pred, "TRUE", "FALSE")

        weight_name = f"boot{int(round(bootstrap_weight * 100)):03d}_hard{int(round(hard_weight * 100)):03d}"
        out_path = OUTPUT_DIR / f"roberta_hard_bootstrap_blend_{weight_name}_thr0p32.csv"
        make_submission(solution_df, labels).to_csv(out_path, index=False)

        distribution = prediction_distribution(labels)
        changed_vs_hard = int(((hard_prob >= THRESHOLD) != pred).sum())
        changed_vs_bootstrap = int(((bootstrap_prob >= THRESHOLD) != pred).sum())
        avg_abs_delta = float(np.mean(np.abs(bootstrap_prob - hard_prob)))
        max_abs_delta = float(np.max(np.abs(bootstrap_prob - hard_prob)))

        row = {
            "filename": str(out_path),
            "bootstrap_weight": float(bootstrap_weight),
            "hard_weight": float(hard_weight),
            "threshold": float(THRESHOLD),
            "pred_FALSE": distribution["pred_FALSE"],
            "pred_TRUE": distribution["pred_TRUE"],
            "changed_vs_hard_thr0p32": changed_vs_hard,
            "changed_vs_bootstrap_thr0p32": changed_vs_bootstrap,
            "avg_abs_prob_delta_between_sources": avg_abs_delta,
            "max_abs_prob_delta_between_sources": max_abs_delta,
        }
        manifest.append(row)
        print(
            f"Saved {out_path.name} | boot_weight={bootstrap_weight:.2f} "
            f"| TRUE={distribution['pred_TRUE']} | changed_vs_bootstrap={changed_vs_bootstrap}"
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_path = OUTPUT_DIR / "roberta_hard_bootstrap_blend_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    notes_path = OUTPUT_DIR / "roberta_hard_bootstrap_blend_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# RoBERTa Hard + Bootstrap Blend Notes",
                "",
                "No training is performed. This is pure probability blending.",
                "",
                "## Sources",
                "",
                f"- Previous hard-weighted run: `{HARD_PROB_PATH}`",
                f"- Bootstrapped hard-weighted run: `{BOOTSTRAP_PROB_PATH}`",
                "",
                "## Current Best Anchor",
                "",
                "- `roberta_hard_weighted_bootstrap_gpu_submission_thr0p32.csv`",
                "- Public LB: `0.93718166`",
                "",
                "## Candidate Logic",
                "",
                "- Blend probabilities only, not hard labels.",
                "- Use fixed threshold `0.32` because it is the best known LB-calibrated threshold.",
                "- Keep only five candidates.",
                "- Weight bootstrapped model at `0.50`, `0.60`, `0.70`, `0.80`, `0.90`.",
                "",
                "Recommended first submissions: `boot070_hard030`, then `boot080_hard020`, then `boot060_hard040`.",
                "",
                f"Manifest: `{manifest_path}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved notes: {notes_path}")


if __name__ == "__main__":
    main()
