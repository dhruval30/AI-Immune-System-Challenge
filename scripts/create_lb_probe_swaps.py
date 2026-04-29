#!/usr/bin/env python3
# pip install pandas numpy
#
# Create balanced one-pair probe submissions from the rows that changed in the
# successful balanced swap. These are for controlled leaderboard feedback:
# each file flips exactly one RoBERTa FALSE->TRUE row and one RoBERTa TRUE->FALSE
# row, preserving the total TRUE count.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INSTALL_CMD = "pip install pandas numpy"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / "lb_probes"

SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
BASE_SUBMISSION_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_submission.csv"
SWAP_SCORES_PATH = OUTPUTS_DIR / "balanced_swaps" / "balanced_swap_row_scores.csv"

MANIFEST_PATH = OUT_DIR / "lb_probe_manifest.csv"
REVIEW_PATH = OUT_DIR / "lb_probe_candidate_review.md"
NOTES_PATH = OUT_DIR / "lb_probe_notes.md"

# From roberta_balanced_swap_03.csv, which improved LB.
FALSE_TO_TRUE_CANDIDATES = [1686, 557, 1538]
TRUE_TO_FALSE_CANDIDATES = [857, 1187, 1311]

# Six probes are enough to separate pair effects without burning all daily slots.
PAIR_PROBES = [
    (1686, 857),
    (557, 1187),
    (1538, 1311),
    (1686, 1187),
    (557, 1311),
    (1538, 857),
]


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if not set(np.unique(labels)).issubset({"TRUE", "FALSE"}):
        raise ValueError("Submission labels must be TRUE/FALSE.")
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels})
    out = solution_df.copy()
    out["label"] = labels
    return out[solution_df.columns.tolist()]


def short_text(text: object, max_len: int = 240) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) > max_len:
        return value[: max_len - 3] + "..."
    return value


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== LB Probe Swap Generator ===")
    print("No training is performed. No test labels are used.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    base_df = pd.read_csv(BASE_SUBMISSION_PATH)
    scores_df = pd.read_csv(SWAP_SCORES_PATH)

    if len(base_df) != len(solution_df):
        raise ValueError("Base submission row count does not match solution_format.")
    if "label" not in base_df.columns:
        raise ValueError("Base submission must contain label column.")

    labels = base_df["label"].astype(str).str.upper().to_numpy()
    if not set(np.unique(labels)).issubset({"TRUE", "FALSE"}):
        raise ValueError("Base labels must be TRUE/FALSE.")

    base_true_count = int((labels == "TRUE").sum())
    manifest_rows = []

    for false_idx, true_idx in PAIR_PROBES:
        if labels[false_idx] != "FALSE":
            raise ValueError(f"Expected row {false_idx} to be FALSE in base submission.")
        if labels[true_idx] != "TRUE":
            raise ValueError(f"Expected row {true_idx} to be TRUE in base submission.")

        probe_labels = labels.copy()
        probe_labels[false_idx] = "TRUE"
        probe_labels[true_idx] = "FALSE"

        if int((probe_labels == "TRUE").sum()) != base_true_count:
            raise RuntimeError("Probe did not preserve TRUE count.")

        filename = f"probe_pair_false{false_idx}_true{true_idx}.csv"
        out_path = OUT_DIR / filename
        make_submission(solution_df, probe_labels).to_csv(out_path, index=False)

        false_row = scores_df.loc[scores_df["idx"] == false_idx].iloc[0].to_dict()
        true_row = scores_df.loc[scores_df["idx"] == true_idx].iloc[0].to_dict()
        manifest_rows.append(
            {
                "filename": filename,
                "false_to_true_idx": false_idx,
                "true_to_false_idx": true_idx,
                "pred_TRUE_count": base_true_count,
                "false_to_true_score": false_row.get("false_to_true_score"),
                "true_to_false_score": true_row.get("true_to_false_score"),
                "false_to_true_roberta_prob": false_row.get("roberta_prob_TRUE"),
                "true_to_false_roberta_prob": true_row.get("roberta_prob_TRUE"),
            }
        )
        print(f"Saved {filename}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    write_review(scores_df, manifest_df)
    write_notes(manifest_df)

    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Saved review: {REVIEW_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


def write_review(scores_df: pd.DataFrame, manifest_df: pd.DataFrame) -> None:
    lines = [
        "# LB Probe Candidate Review",
        "",
        "These are the rows involved in the one-pair leaderboard probes.",
        "",
        "Interpretation: if a probe improves over RoBERTa base, that FALSE->TRUE / TRUE->FALSE pair is likely useful on the public set.",
        "",
        "## Probe Files",
        "",
        manifest_df.to_markdown(index=False),
        "",
        "## FALSE -> TRUE Candidates",
        "",
    ]

    cols = [
        "idx",
        "base_label",
        "roberta_prob_TRUE",
        "consensus_true_rank",
        "mean_secondary_prob",
        "false_to_true_score",
        "text",
    ]
    for idx in FALSE_TO_TRUE_CANDIDATES:
        row = scores_df.loc[scores_df["idx"] == idx].iloc[0]
        lines.append(f"### idx {idx}")
        lines.append("")
        for col in cols[:-1]:
            lines.append(f"- {col}: `{row[col]}`")
        lines.append("")
        lines.append("```text")
        lines.append(short_text(row["text"], max_len=1000))
        lines.append("```")
        lines.append("")

    lines.append("## TRUE -> FALSE Candidates")
    lines.append("")
    cols = [
        "idx",
        "base_label",
        "roberta_prob_TRUE",
        "consensus_true_rank",
        "mean_secondary_prob",
        "true_to_false_score",
        "text",
    ]
    for idx in TRUE_TO_FALSE_CANDIDATES:
        row = scores_df.loc[scores_df["idx"] == idx].iloc[0]
        lines.append(f"### idx {idx}")
        lines.append("")
        for col in cols[:-1]:
            lines.append(f"- {col}: `{row[col]}`")
        lines.append("")
        lines.append("```text")
        lines.append(short_text(row["text"], max_len=1000))
        lines.append("```")
        lines.append("")

    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_notes(manifest_df: pd.DataFrame) -> None:
    lines = [
        "# LB Probe Notes",
        "",
        "## Why This Exists",
        "",
        "`roberta_balanced_swap_03.csv` improved public LB, but we do not know which individual row changes were correct.",
        "",
        "These probes isolate row pairs while preserving RoBERTa's TRUE count. They are designed to learn from leaderboard feedback, not to be final model submissions.",
        "",
        "## Submission Advice",
        "",
        "Submit at most a few probes in one day. Record each returned LB score next to the manifest.",
        "",
        "If a one-pair probe beats RoBERTa base, keep that pair. If it beats swap03, it is especially valuable.",
        "",
        "## Generated Probes",
        "",
        manifest_df.to_markdown(index=False),
        "",
    ]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
