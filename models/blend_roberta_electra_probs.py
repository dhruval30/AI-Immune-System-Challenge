#!/usr/bin/env python3
# pip install pandas numpy
#
# No training is performed. This script only blends existing RoBERTa and
# ELECTRA test probabilities and writes five submission candidates.

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
ELECTRA_TEST_PROB_PATH = OUTPUTS_DIR / "electra_base" / "electra_base_test_probabilities.csv"
ROBERTA_SUBMISSION_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_submission.csv"
ELECTRA_SUBMISSION_PATH = OUTPUTS_DIR / "electra_base" / "electra_base_submission.csv"

MANIFEST_PATH = BLENDS_DIR / "roberta_electra_prob_blend_manifest.csv"
NOTES_PATH = BLENDS_DIR / "roberta_electra_prob_blend_notes.md"

# RoBERTa is the known strong anchor. These five candidates were selected
# because their TRUE counts stay near the known strong RoBERTa distribution
# (~577 TRUE / ~1523 FALSE) and the best previous blend count (~578 TRUE).
BLEND_CANDIDATES = [
    {"name": "count577_r0p90_thr0p34", "roberta_weight": 0.90, "threshold": 0.34},
    {"name": "count577_r0p85_thr0p34", "roberta_weight": 0.85, "threshold": 0.34},
    {"name": "count578_r0p95_thr0p32", "roberta_weight": 0.95, "threshold": 0.32},
    {"name": "count576_r0p95_thr0p34", "roberta_weight": 0.95, "threshold": 0.34},
    {"name": "count578_r0p80_thr0p34", "roberta_weight": 0.80, "threshold": 0.34},
]


def load_probabilities(path: Path, model_name: str) -> tuple[pd.DataFrame, np.ndarray]:
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

    return df, probs


def validate_text_alignment(left_df: pd.DataFrame, right_df: pd.DataFrame) -> None:
    if "text" not in left_df.columns or "text" not in right_df.columns:
        print("Text alignment check skipped because one probability file lacks text column.")
        return

    left_text = left_df["text"].astype(str).str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False).str.strip()
    right_text = right_df["text"].astype(str).str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False).str.strip()
    mismatches = np.where(left_text.to_numpy() != right_text.to_numpy())[0]
    if len(mismatches):
        first = int(mismatches[0])
        raise ValueError(f"RoBERTa/ELECTRA probability text alignment failed at row {first}.")


def load_submission_labels(path: Path, expected_rows: int, label: str) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) != expected_rows or "label" not in df.columns:
        print(f"{label} submission labels skipped because shape/schema did not match.")
        return None
    labels = df["label"].astype(str).str.upper()
    if not set(labels.unique()).issubset({"TRUE", "FALSE"}):
        print(f"{label} submission labels skipped because labels are not TRUE/FALSE.")
        return None
    return labels


def make_submission(solution_df: pd.DataFrame, labels_true_false: np.ndarray) -> pd.DataFrame:
    if not set(np.unique(labels_true_false)).issubset({"TRUE", "FALSE"}):
        raise ValueError("Generated labels must be only TRUE/FALSE.")

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
    roberta_df, roberta_probs = load_probabilities(ROBERTA_TEST_PROB_PATH, "RoBERTa")
    electra_df, electra_probs = load_probabilities(ELECTRA_TEST_PROB_PATH, "ELECTRA")

    n_rows = len(solution_df)
    if len(roberta_probs) != n_rows:
        raise ValueError(f"RoBERTa probability rows ({len(roberta_probs)}) != solution rows ({n_rows})")
    if len(electra_probs) != n_rows:
        raise ValueError(f"ELECTRA probability rows ({len(electra_probs)}) != solution rows ({n_rows})")

    validate_text_alignment(roberta_df, electra_df)

    roberta_labels = load_submission_labels(ROBERTA_SUBMISSION_PATH, n_rows, "RoBERTa")
    electra_labels = load_submission_labels(ELECTRA_SUBMISSION_PATH, n_rows, "ELECTRA")

    print(f"Solution rows: {n_rows}")
    print(f"RoBERTa probs loaded: {len(roberta_probs)}")
    print(f"ELECTRA probs loaded: {len(electra_probs)}")

    print("\n=== Generating five selected blended submissions ===")
    manifest_rows: list[dict] = []

    for candidate in BLEND_CANDIDATES:
        roberta_weight = float(candidate["roberta_weight"])
        electra_weight = 1.0 - roberta_weight
        threshold = float(candidate["threshold"])
        blended_probs = (roberta_weight * roberta_probs) + (electra_weight * electra_probs)

        pred_true = blended_probs >= threshold
        pred_labels = np.where(pred_true, "TRUE", "FALSE")

        filename = (
            f"roberta_electra_blend_{candidate['name']}"
            f"_r{slug_float(roberta_weight)}_e{slug_float(electra_weight)}"
            f"_thr{slug_float(threshold)}.csv"
        )
        out_path = BLENDS_DIR / filename

        submission_df = make_submission(solution_df, pred_labels)
        if submission_df.columns.tolist() != solution_df.columns.tolist():
            raise ValueError(f"Submission columns do not match solution_format for {filename}")
        submission_df.to_csv(out_path, index=False)

        row = {
            "candidate": candidate["name"],
            "filename": filename,
            "roberta_weight": roberta_weight,
            "electra_weight": electra_weight,
            "threshold": threshold,
            "pred_FALSE_count": int((~pred_true).sum()),
            "pred_TRUE_count": int(pred_true.sum()),
        }
        if roberta_labels is not None:
            row["changed_vs_roberta_base"] = int((pd.Series(pred_labels).str.upper() != roberta_labels).sum())
        if electra_labels is not None:
            row["changed_vs_electra_base"] = int((pd.Series(pred_labels).str.upper() != electra_labels).sum())
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved {len(manifest_rows)} selected blend submissions.")
    print(f"Saved manifest: {MANIFEST_PATH}")

    notes = f"""# RoBERTa + ELECTRA Probability Blend Notes

## Summary

- No training was performed.
- This script only blends existing test probabilities.
- This is probability blending, not a router.
- Blend sources:
- `outputs/roberta_base/roberta_base_test_probabilities.csv`
- `outputs/electra_base/electra_base_test_probabilities.csv`

## Why This Blend

RoBERTa is the known strong anchor. ELECTRA may add diversity because its discriminator-style pretraining can react differently to synthetic, corrupted, or unnatural text artifacts.

## Blend Grid

- This script intentionally generates only five selected candidates.
- Candidates: `{BLEND_CANDIDATES}`
- Total files generated: `{len(manifest_rows)}`

## Selection Logic

The candidates keep prediction counts near the known strong RoBERTa distribution (`577 TRUE / 1523 FALSE`) and the best previous blend count (`578 TRUE`).

Use the manifest to choose submission order. Prefer the candidates with the smallest number of changes vs the original RoBERTa submission first.

## Output Files

- Blend submissions: `outputs/blends/roberta_electra_blend_*.csv`
- Manifest: `outputs/blends/roberta_electra_prob_blend_manifest.csv`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
