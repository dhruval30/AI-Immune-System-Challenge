#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SOLUTION_FORMAT_PATH = ROOT_DIR / "data" / "solution_format.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TRUE-count calibrated submissions from a probability CSV.")
    parser.add_argument("--prob-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[568, 572, 577, 582, 586])
    parser.add_argument("--prob-column", default="pred_prob_TRUE")
    return parser.parse_args()


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = labels
    return submission_df[solution_df.columns.tolist()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prob_df = pd.read_csv(args.prob_path)
    if args.prob_column not in prob_df.columns:
        raise ValueError(f"{args.prob_path} is missing probability column {args.prob_column!r}.")

    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    probs = prob_df[args.prob_column].to_numpy(dtype=np.float64)
    order = np.argsort(-probs)

    manifest = []
    for count in args.counts:
        if count < 0 or count > len(probs):
            raise ValueError(f"Invalid count {count}; expected 0..{len(probs)}.")
        pred = np.zeros(len(probs), dtype=bool)
        pred[order[:count]] = True
        labels = np.where(pred, "TRUE", "FALSE")
        out_path = args.output_dir / f"{args.name}_top{count}_submission.csv"
        make_submission(solution_df, labels).to_csv(out_path, index=False)
        threshold_floor = float(probs[order[count - 1]]) if count > 0 else 1.0
        manifest.append(
            {
                "count": int(count),
                "threshold_floor": threshold_floor,
                "submission": str(out_path),
            }
        )
        print(f"saved {out_path} TRUE_count={count} threshold_floor~{threshold_floor:.6f}")

    manifest_path = args.output_dir / f"{args.name}_count_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
