#!/usr/bin/env python3
# pip install pandas numpy
#
# Build balanced swap submissions around the strong RoBERTa base prediction.
# The total TRUE count is preserved while replacing weak RoBERTa TRUE rows with
# suspicious RoBERTa FALSE rows using independent model/style signals.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / "balanced_swaps"

SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
ROBERTA_PROB_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_test_probabilities.csv"
ROBERTA_SUBMISSION_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_submission.csv"

SIGNAL_FILES = {
    "electra": (OUTPUTS_DIR / "electra_base" / "electra_base_test_probabilities.csv", "pred_prob_TRUE", 0.90),
    "modernbert_base": (OUTPUTS_DIR / "modernbert_base" / "modernbert_base_test_probabilities.csv", "pred_prob_TRUE", 0.70),
    "modernbert_lora_v2": (OUTPUTS_DIR / "modernbert_lora_v2" / "modernbert_lora_v2_test_probabilities.csv", "pred_prob_TRUE", 0.80),
    "roberta_style": (OUTPUTS_DIR / "roberta_style_features" / "roberta_style_features_test_probabilities.csv", "pred_prob_TRUE", 0.75),
    "label_style_calibrated": (
        OUTPUTS_DIR / "roberta_label_style_calibrator" / "roberta_label_style_calibrator_test_scores.csv",
        "calibrated_prob_TRUE",
        0.55,
    ),
    "label_style_rankblend_090": (
        OUTPUTS_DIR / "roberta_label_style_calibrator" / "roberta_label_style_calibrator_test_scores.csv",
        "rank_blend_roberta_0p90",
        0.45,
    ),
    "roberta_cv": (OUTPUTS_DIR / "roberta_base_cv" / "roberta_base_cv_test_probabilities.csv", "pred_prob_TRUE", 0.55),
    "roberta_cv_v2": (OUTPUTS_DIR / "roberta_base_cv_v2" / "roberta_base_cv_v2_test_probabilities.csv", "pred_prob_TRUE", 0.55),
    "hard_specialist": (
        OUTPUTS_DIR / "roberta_routed" / "roberta_hard_specialist_test_probabilities.csv",
        "specialist_prob_TRUE",
        0.45,
    ),
}

SWAP_COUNTS = [3, 5, 8, 12, 16]

MANIFEST_PATH = OUT_DIR / "balanced_swap_manifest.csv"
DIAGNOSTICS_PATH = OUT_DIR / "balanced_swap_row_scores.csv"
NOTES_PATH = OUT_DIR / "balanced_swap_notes.md"


def normalize_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def load_prob_file(path: Path, column: str, expected_rows: int, signal_name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping missing signal {signal_name}: {path}")
        return None

    df = pd.read_csv(path)
    if len(df) != expected_rows:
        print(f"Skipping signal {signal_name}: row count {len(df)} != expected {expected_rows}")
        return None
    if column not in df.columns:
        print(f"Skipping signal {signal_name}: missing column {column}")
        return None

    probs = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if np.isnan(probs).any():
        print(f"Skipping signal {signal_name}: NaN probabilities")
        return None

    # Rank-blend scores can be rank-like instead of calibrated probability. Normalize if needed.
    if probs.min() < 0.0 or probs.max() > 1.0:
        order = probs.argsort().argsort()
        probs = order / max(len(probs) - 1, 1)

    return pd.DataFrame({signal_name: probs})


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if not set(np.unique(labels)).issubset({"TRUE", "FALSE"}):
        raise ValueError("Labels must be TRUE/FALSE.")

    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels})

    submission = solution_df.copy()
    submission["label"] = labels
    return submission[solution_df.columns.tolist()]


def rank01(values: np.ndarray, higher_is_more_true: bool = True) -> np.ndarray:
    if not higher_is_more_true:
        values = -values
    order = values.argsort().argsort().astype(float)
    return order / max(len(values) - 1, 1)


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Balanced RoBERTa Swap Generator ===")
    print("No training is performed. No test labels are used.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    roberta_probs_df = pd.read_csv(ROBERTA_PROB_PATH)
    roberta_submission_df = pd.read_csv(ROBERTA_SUBMISSION_PATH)

    n_rows = len(solution_df)
    if len(roberta_probs_df) != n_rows or len(roberta_submission_df) != n_rows:
        raise ValueError("RoBERTa probabilities/submission must match solution_format row count.")
    if "label" not in roberta_submission_df.columns:
        raise ValueError("RoBERTa submission must contain label column.")

    texts = (
        roberta_probs_df["text"].map(normalize_text).tolist()
        if "text" in roberta_probs_df.columns
        else [""] * n_rows
    )
    roberta_prob = pd.to_numeric(roberta_probs_df["pred_prob_TRUE"], errors="raise").to_numpy(dtype=float)
    base_labels = roberta_submission_df["label"].astype(str).str.upper().to_numpy()
    if not set(np.unique(base_labels)).issubset({"TRUE", "FALSE"}):
        raise ValueError("RoBERTa base labels must be TRUE/FALSE.")

    print(f"Base TRUE count: {int((base_labels == 'TRUE').sum())}")
    print(f"Base FALSE count: {int((base_labels == 'FALSE').sum())}")

    signal_frames = []
    signal_weights = {}
    for signal_name, (path, column, weight) in SIGNAL_FILES.items():
        frame = load_prob_file(path, column, n_rows, signal_name)
        if frame is not None:
            signal_frames.append(frame)
            signal_weights[signal_name] = weight
            print(f"Loaded signal: {signal_name} weight={weight}")

    if not signal_frames:
        raise RuntimeError("No secondary signals loaded.")

    signals_df = pd.concat(signal_frames, axis=1)

    # Consensus score is rank-normalized so models with different calibration can still vote.
    weighted_rank_sum = np.zeros(n_rows, dtype=float)
    total_weight = 0.0
    for signal_name, weight in signal_weights.items():
        weighted_rank_sum += weight * rank01(signals_df[signal_name].to_numpy(dtype=float), higher_is_more_true=True)
        total_weight += weight
    consensus_true_rank = weighted_rank_sum / max(total_weight, 1e-9)

    roberta_true_rank = rank01(roberta_prob, higher_is_more_true=True)
    mean_secondary_prob = signals_df.mean(axis=1).to_numpy(dtype=float)
    max_secondary_prob = signals_df.max(axis=1).to_numpy(dtype=float)
    min_secondary_prob = signals_df.min(axis=1).to_numpy(dtype=float)

    # For RoBERTa FALSE rows, high secondary consensus near the base boundary means likely false negative.
    false_mask = base_labels == "FALSE"
    true_mask = base_labels == "TRUE"

    false_to_true_score = (
        0.45 * consensus_true_rank
        + 0.25 * roberta_true_rank
        + 0.20 * mean_secondary_prob
        + 0.10 * max_secondary_prob
    )

    # For RoBERTa TRUE rows, low secondary consensus and low RoBERTa probability means likely false positive.
    true_to_false_score = (
        0.45 * (1.0 - consensus_true_rank)
        + 0.30 * (1.0 - roberta_true_rank)
        + 0.15 * (1.0 - mean_secondary_prob)
        + 0.10 * (1.0 - min_secondary_prob)
    )

    diagnostics = pd.DataFrame(
        {
            "idx": np.arange(n_rows),
            "text": texts,
            "base_label": base_labels,
            "roberta_prob_TRUE": roberta_prob,
            "consensus_true_rank": consensus_true_rank,
            "mean_secondary_prob": mean_secondary_prob,
            "max_secondary_prob": max_secondary_prob,
            "min_secondary_prob": min_secondary_prob,
            "false_to_true_score": false_to_true_score,
            "true_to_false_score": true_to_false_score,
        }
    )
    diagnostics = pd.concat([diagnostics, signals_df], axis=1)

    false_candidates = diagnostics[false_mask].sort_values("false_to_true_score", ascending=False).copy()
    true_candidates = diagnostics[true_mask].sort_values("true_to_false_score", ascending=False).copy()

    manifest_rows = []
    for swap_count in SWAP_COUNTS:
        if swap_count > len(false_candidates) or swap_count > len(true_candidates):
            continue

        flip_to_true_idx = false_candidates.head(swap_count)["idx"].to_numpy(dtype=int)
        flip_to_false_idx = true_candidates.head(swap_count)["idx"].to_numpy(dtype=int)

        labels = base_labels.copy()
        labels[flip_to_true_idx] = "TRUE"
        labels[flip_to_false_idx] = "FALSE"

        if int((labels == "TRUE").sum()) != int((base_labels == "TRUE").sum()):
            raise RuntimeError("Balanced swap failed to preserve TRUE count.")

        filename = f"roberta_balanced_swap_{swap_count:02d}.csv"
        out_path = OUT_DIR / filename
        make_submission(solution_df, labels).to_csv(out_path, index=False)

        manifest_rows.append(
            {
                "filename": filename,
                "swap_count_each_direction": swap_count,
                "total_changed_rows": 2 * swap_count,
                "pred_TRUE_count": int((labels == "TRUE").sum()),
                "pred_FALSE_count": int((labels == "FALSE").sum()),
                "flip_false_to_true_indices": " ".join(map(str, flip_to_true_idx.tolist())),
                "flip_true_to_false_indices": " ".join(map(str, flip_to_false_idx.tolist())),
                "mean_false_to_true_score": float(false_candidates.head(swap_count)["false_to_true_score"].mean()),
                "mean_true_to_false_score": float(true_candidates.head(swap_count)["true_to_false_score"].mean()),
            }
        )

        diagnostics[f"swap_{swap_count:02d}_action"] = "keep"
        diagnostics.loc[flip_to_true_idx, f"swap_{swap_count:02d}_action"] = "FALSE_to_TRUE"
        diagnostics.loc[flip_to_false_idx, f"swap_{swap_count:02d}_action"] = "TRUE_to_FALSE"

        print(
            f"Saved {filename}: swapped {swap_count} FALSE->TRUE and {swap_count} TRUE->FALSE; "
            f"TRUE count stays {int((labels == 'TRUE').sum())}"
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    write_notes(manifest_df, signal_weights)

    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Saved diagnostics: {DIAGNOSTICS_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


def write_notes(manifest_df: pd.DataFrame, signal_weights: dict[str, float]) -> None:
    lines = [
        "# Balanced RoBERTa Swap Notes",
        "",
        "## Purpose",
        "",
        "These candidates preserve the RoBERTa base TRUE count while swapping suspicious rows in both directions.",
        "",
        "The previous rule override reduced TRUE count and hurt LB. This script avoids that by making balanced swaps.",
        "",
        "No training is performed. No test labels are used.",
        "",
        "## Signals",
        "",
    ]
    for name, weight in signal_weights.items():
        lines.append(f"- `{name}` weight `{weight}`")

    lines.extend(
        [
            "",
            "## Generated Candidates",
            "",
        ]
    )
    if manifest_df.empty:
        lines.append("No candidates generated.")
    else:
        lines.append(manifest_df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Suggested Submission Order",
            "",
            "Start small. Submit `roberta_balanced_swap_03.csv` first, then `05` only if it improves.",
            "",
            "These files are controlled experiments, not guaranteed improvements.",
        ]
    )
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
