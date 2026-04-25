#!/usr/bin/env python3
# pip install pandas numpy
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
BLENDS_DIR = OUTPUTS_DIR / "blends"

SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
ROBERTA_TEST_PROB_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_test_probabilities.csv"
MODERNBERT_TEST_PROB_PATH = OUTPUTS_DIR / "modernbert_base" / "modernbert_base_test_probabilities.csv"

MANIFEST_PATH = BLENDS_DIR / "roberta_modernbert_prob_blend_manifest.csv"
NOTES_PATH = BLENDS_DIR / "roberta_modernbert_prob_blend_notes.md"

# RoBERTa is stronger, so keep it dominant in blending.
ROBERTA_WEIGHTS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
THRESHOLDS = [0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.45, 0.50]


def load_probabilities(path: Path, model_name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing probability file for {model_name}: {path}")

    df = pd.read_csv(path)
    if "pred_prob_TRUE" not in df.columns:
        raise ValueError(f"{path} must contain 'pred_prob_TRUE' column.")

    probs = pd.to_numeric(df["pred_prob_TRUE"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(probs).any():
        raise ValueError(f"Found NaN/invalid probability values in {path}")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"Probability values outside [0, 1] in {path}")

    return probs


def make_submission(solution_df: pd.DataFrame, labels_true_false: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels_true_false})

    submission_df = solution_df.copy()
    submission_df["label"] = labels_true_false
    return submission_df[solution_df.columns.tolist()]


def slug_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    BLENDS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading artifacts ===")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    roberta_probs = load_probabilities(ROBERTA_TEST_PROB_PATH, "RoBERTa")
    modernbert_probs = load_probabilities(MODERNBERT_TEST_PROB_PATH, "ModernBERT")

    n_rows = len(solution_df)
    if len(roberta_probs) != n_rows:
        raise ValueError(f"RoBERTa probability rows ({len(roberta_probs)}) != solution rows ({n_rows})")
    if len(modernbert_probs) != n_rows:
        raise ValueError(f"ModernBERT probability rows ({len(modernbert_probs)}) != solution rows ({n_rows})")

    print(f"Solution rows: {n_rows}")
    print(f"RoBERTa probs loaded: {len(roberta_probs)}")
    print(f"ModernBERT probs loaded: {len(modernbert_probs)}")

    print("\n=== Generating blended submissions ===")
    manifest_rows: list[dict] = []

    for roberta_weight in ROBERTA_WEIGHTS:
        modernbert_weight = 1.0 - roberta_weight
        blended_probs = (roberta_weight * roberta_probs) + (modernbert_weight * modernbert_probs)

        for threshold in THRESHOLDS:
            pred_true = blended_probs >= threshold
            pred_labels = np.where(pred_true, "TRUE", "FALSE")

            filename = (
                f"roberta_modernbert_blend_r{slug_float(roberta_weight)}"
                f"_m{slug_float(modernbert_weight)}_thr{slug_float(threshold)}.csv"
            )
            out_path = BLENDS_DIR / filename

            submission_df = make_submission(solution_df, pred_labels)
            submission_df.to_csv(out_path, index=False)

            manifest_rows.append(
                {
                    "filename": filename,
                    "roberta_weight": roberta_weight,
                    "modernbert_weight": modernbert_weight,
                    "threshold": threshold,
                    "pred_FALSE count": int((~pred_true).sum()),
                    "pred_TRUE count": int(pred_true.sum()),
                }
            )

    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "filename",
            "roberta_weight",
            "modernbert_weight",
            "threshold",
            "pred_FALSE count",
            "pred_TRUE count",
        ],
    )
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved {len(manifest_rows)} blend submissions.")
    print(f"Saved manifest: {MANIFEST_PATH}")

    notes = f"""# RoBERTa + ModernBERT Probability Blend Notes

## Summary

- No training was performed.
- This script only blends existing test probabilities.
- Blend sources:
- `outputs/roberta_base/roberta_base_test_probabilities.csv`
- `outputs/modernbert_base/modernbert_base_test_probabilities.csv`

## Blend Grid

- RoBERTa weights: `{ROBERTA_WEIGHTS}`
- ModernBERT weight: `1 - roberta_weight`
- Thresholds: `{THRESHOLDS}`
- Total files generated: `{len(manifest_rows)}`

## Practical Starting Point

- Start with RoBERTa-heavy blends:
- `roberta_weight` in `0.85` to `0.95`
- threshold in `0.30` to `0.36`

## Output Files

- Blend submissions: `outputs/blends/roberta_modernbert_blend_*.csv`
- Manifest: `outputs/blends/roberta_modernbert_prob_blend_manifest.csv`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()

