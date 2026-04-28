# pip install pandas numpy torch transformers tqdm
#
# Inference-only router:
# - base RoBERTa handles confident rows
# - hard specialist handles only uncertain base-probability rows
# This script does not train and does not use test labels.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INSTALL_CMD = "pip install pandas numpy torch transformers tqdm"

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
EVAL_BATCH_SIZE = 16

ROUTER_BANDS = [
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
    (0.25, 0.75),
]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

BASE_OUTPUT_DIR = OUTPUT_DIR / "roberta_base"
BASE_PROB_PATH = BASE_OUTPUT_DIR / "roberta_base_test_probabilities.csv"
BASE_METRICS_PATH = BASE_OUTPUT_DIR / "roberta_base_metrics.json"
BASE_SUBMISSION_PATH = BASE_OUTPUT_DIR / "roberta_base_submission.csv"

SPECIALIST_OUTPUT_DIR = OUTPUT_DIR / "roberta_hard_specialist"
SPECIALIST_MODEL_DIR = SPECIALIST_OUTPUT_DIR / "best_model"
SPECIALIST_METRICS_PATH = SPECIALIST_OUTPUT_DIR / "roberta_hard_specialist_metrics.json"

ROUTED_OUTPUT_DIR = OUTPUT_DIR / "roberta_routed"
SPECIALIST_TEST_PROB_PATH = ROUTED_OUTPUT_DIR / "roberta_hard_specialist_test_probabilities.csv"
ROUTED_DIAGNOSTICS_PATH = ROUTED_OUTPUT_DIR / "routed_diagnostics.csv"
ROUTED_SUMMARY_PATH = ROUTED_OUTPUT_DIR / "routed_summary.md"


def detect_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    records = []
    total = count_lines(path)
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def normalize_text(value: object) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_texts(values: Iterable[object], desc: str) -> list[str]:
    return [normalize_text(value) for value in tqdm(list(values), desc=desc, unit="rows")]


def validate_solution_format(solution_df: pd.DataFrame, expected_rows: int) -> None:
    if len(solution_df) != expected_rows:
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({expected_rows})."
        )
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")


def validate_base_probabilities(base_prob_df: pd.DataFrame, test_texts: Sequence[str]) -> None:
    required_cols = {"text", "pred_prob_TRUE"}
    missing_cols = sorted(required_cols - set(base_prob_df.columns))
    if missing_cols:
        raise ValueError(f"Base probability file is missing required columns: {missing_cols}")

    if len(base_prob_df) != len(test_texts):
        raise ValueError(
            f"Base probability rows ({len(base_prob_df)}) do not match test rows ({len(test_texts)})."
        )

    if base_prob_df["pred_prob_TRUE"].isna().any():
        raise ValueError("Base probabilities contain missing values.")

    probs = base_prob_df["pred_prob_TRUE"].to_numpy(dtype=float)
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Base probabilities must be between 0 and 1.")

    base_texts = normalize_texts(base_prob_df["text"], desc="Normalizing base probability text")
    mismatches = [i for i, (a, b) in enumerate(zip(test_texts, base_texts)) if a != b]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            "Base probability text alignment failed. "
            f"First mismatch at row {first}: test={test_texts[first]!r}, base={base_texts[first]!r}"
        )


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict, keys: Sequence[str]) -> object | None:
    cursor: object = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def coerce_threshold(value: object, label: str) -> float | None:
    if value is None:
        return None
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{label} threshold must be between 0 and 1. Found: {threshold}")
    return threshold


def extract_threshold(metrics: dict, label: str) -> float:
    candidate_paths = [
        ("threshold_tuning", "selected_threshold"),
        ("best_epoch", "selected_specialist_threshold"),
        ("best_epoch", "best_tuned_threshold"),
        ("validation_final", "threshold"),
        ("selected_threshold",),
        ("specialist_threshold",),
    ]

    for path in candidate_paths:
        threshold = coerce_threshold(nested_get(metrics, path), label)
        if threshold is not None:
            return threshold

    best_epoch = nested_get(metrics, ("best_epoch", "epoch"))
    epoch_metrics = metrics.get("epoch_metrics", [])
    if best_epoch is not None and isinstance(epoch_metrics, list):
        for record in epoch_metrics:
            if int(record.get("epoch", -1)) == int(best_epoch):
                threshold = coerce_threshold(record.get("best_tuned_threshold"), label)
                if threshold is not None:
                    return threshold

    threshold_table = nested_get(metrics, ("threshold_tuning", "table"))
    if isinstance(threshold_table, list) and threshold_table:
        best_record = max(threshold_table, key=lambda row: float(row.get("f1", -1.0)))
        threshold = coerce_threshold(best_record.get("threshold"), label)
        if threshold is not None:
            return threshold

    raise KeyError(f"Could not find selected threshold in {label} metrics.")


def load_specialist_tokenizer() -> tuple[AutoTokenizer, str]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(SPECIALIST_MODEL_DIR), use_fast=True)
        return tokenizer, str(SPECIALIST_MODEL_DIR)
    except Exception as exc:
        print("Warning: could not load tokenizer from specialist checkpoint.")
        print(f"Tokenizer load error: {exc}")
        print("Falling back to roberta-base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        return tokenizer, MODEL_NAME


def tokenize_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])

    return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0)


def run_specialist_inference(texts: Sequence[str], device: torch.device) -> tuple[np.ndarray, str]:
    if not SPECIALIST_MODEL_DIR.exists():
        raise FileNotFoundError(f"Specialist model directory not found: {SPECIALIST_MODEL_DIR}")

    tokenizer, tokenizer_source = load_specialist_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(str(SPECIALIST_MODEL_DIR), num_labels=2)
    model.to(device)
    model.eval()

    input_ids, attention_mask = tokenize_texts(
        texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test text for specialist",
    )

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Specialist test inference", unit="batch"):
            batch_input_ids, batch_attention_mask = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())

    prob_true = np.concatenate(all_probs)
    if np.isnan(prob_true).any():
        raise ValueError("Specialist probabilities contain NaN values.")
    if np.any((prob_true < 0.0) | (prob_true > 1.0)):
        raise ValueError("Specialist probabilities must be between 0 and 1.")

    return prob_true, tokenizer_source


def band_token(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if len(solution_df) != len(labels):
        raise ValueError("Submission labels length does not match solution_format rows.")

    if not set(labels).issubset({"TRUE", "FALSE"}):
        raise ValueError("Final labels must contain only TRUE/FALSE.")

    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = labels

    return submission_df[solution_df.columns.tolist()]


def load_original_base_submission(expected_rows: int) -> pd.Series | None:
    if not BASE_SUBMISSION_PATH.exists():
        return None

    submission_df = pd.read_csv(BASE_SUBMISSION_PATH)
    if len(submission_df) != expected_rows or "label" not in submission_df.columns:
        return None

    labels = submission_df["label"].astype(str).str.upper()
    if not set(labels.unique()).issubset({"TRUE", "FALSE"}):
        return None
    return labels


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Routed RoBERTa Submission Generation ===")
    print("No training is performed.")
    print("No test labels are used.")

    torch.manual_seed(SEED)
    ROUTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    print(f"Using device: {device}")

    print("\n=== Loading inputs ===")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    if "text" not in test_df.columns:
        raise ValueError("test_labeled_comp.jsonl must contain a 'text' column.")

    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_solution_format(solution_df, expected_rows=len(test_texts))

    base_prob_df = pd.read_csv(BASE_PROB_PATH)
    validate_base_probabilities(base_prob_df, test_texts)
    base_prob_true = base_prob_df["pred_prob_TRUE"].to_numpy(dtype=float)

    base_metrics = load_json(BASE_METRICS_PATH, "base")
    specialist_metrics = load_json(SPECIALIST_METRICS_PATH, "specialist")
    base_threshold = extract_threshold(base_metrics, "base")
    specialist_threshold = extract_threshold(specialist_metrics, "specialist")

    print(f"Base threshold: {base_threshold:.2f}")
    print(f"Specialist threshold: {specialist_threshold:.2f}")

    print("\n=== Specialist inference ===")
    specialist_prob_true, tokenizer_source = run_specialist_inference(test_texts, device=device)
    specialist_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "specialist_prob_TRUE": specialist_prob_true,
        }
    )
    specialist_prob_df.to_csv(SPECIALIST_TEST_PROB_PATH, index=False)
    print(f"Saved specialist test probabilities: {SPECIALIST_TEST_PROB_PATH}")
    print(f"Specialist tokenizer source: {tokenizer_source}")

    original_base_labels = load_original_base_submission(expected_rows=len(test_texts))

    print("\n=== Routing submissions ===")
    all_diagnostics = []
    summary_records = []

    for low_cutoff, high_cutoff in ROUTER_BANDS:
        if low_cutoff >= high_cutoff:
            raise ValueError(f"Invalid router band: low={low_cutoff}, high={high_cutoff}")

        route_band = f"low_{band_token(low_cutoff)}_high_{band_token(high_cutoff)}"

        low_mask = base_prob_true < low_cutoff
        high_mask = base_prob_true > high_cutoff
        specialist_mask = ~(low_mask | high_mask)

        final_labels = np.where(base_prob_true >= base_threshold, "TRUE", "FALSE")
        specialist_labels = np.where(specialist_prob_true >= specialist_threshold, "TRUE", "FALSE")
        final_labels[specialist_mask] = specialist_labels[specialist_mask]

        route_used = np.empty(len(test_texts), dtype=object)
        route_used[low_mask] = "base_low_confident_FALSE"
        route_used[high_mask] = "base_high_confident_TRUE"
        route_used[specialist_mask] = "specialist_uncertain_band"

        if set(np.unique(final_labels)) - {"TRUE", "FALSE"}:
            raise ValueError(f"Invalid final labels generated for {route_band}.")

        submission_df = make_submission(solution_df, final_labels)
        if len(submission_df) != len(solution_df):
            raise ValueError(f"Submission row count mismatch for {route_band}.")
        if submission_df.columns.tolist() != solution_df.columns.tolist():
            raise ValueError(f"Submission columns mismatch for {route_band}.")

        submission_path = ROUTED_OUTPUT_DIR / f"submission_routed_{route_band}.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Saved routed submission: {submission_path}")

        pred_true = int((final_labels == "TRUE").sum())
        pred_false = int((final_labels == "FALSE").sum())

        changed_vs_base = None
        if original_base_labels is not None:
            changed_vs_base = int((pd.Series(final_labels).str.upper() != original_base_labels).sum())

        summary_records.append(
            {
                "route_band": route_band,
                "submission_path": str(submission_path),
                "low_cutoff": float(low_cutoff),
                "high_cutoff": float(high_cutoff),
                "base_threshold": float(base_threshold),
                "specialist_threshold": float(specialist_threshold),
                "base_low_rows": int(low_mask.sum()),
                "base_high_rows": int(high_mask.sum()),
                "specialist_rows": int(specialist_mask.sum()),
                "pred_FALSE": pred_false,
                "pred_TRUE": pred_true,
                "changed_vs_roberta_base_submission": changed_vs_base,
            }
        )

        all_diagnostics.append(
            pd.DataFrame(
                {
                    "text": test_texts,
                    "base_prob_TRUE": base_prob_true,
                    "specialist_prob_TRUE": specialist_prob_true,
                    "route_band": route_band,
                    "route_used": route_used,
                    "final_label": final_labels,
                }
            )
        )

    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)
    diagnostics_df.to_csv(ROUTED_DIAGNOSTICS_PATH, index=False)
    print(f"Saved routed diagnostics: {ROUTED_DIAGNOSTICS_PATH}")

    write_summary(summary_records, tokenizer_source)
    print(f"Saved routed summary: {ROUTED_SUMMARY_PATH}")


def write_summary(summary_records: list[dict], tokenizer_source: str) -> None:
    lines = [
        "# Routed RoBERTa Summary",
        "",
        "## Purpose",
        "",
        "This is not probability averaging and not a standard ensemble.",
        "",
        "The router uses base RoBERTa predictions for confident rows and the hard-boundary specialist only inside the base uncertainty band.",
        "",
        "No training is performed by this script. No test labels are used.",
        "",
        "## Inputs",
        "",
        f"- Base probabilities: `{BASE_PROB_PATH.relative_to(ROOT_DIR)}`",
        f"- Base metrics: `{BASE_METRICS_PATH.relative_to(ROOT_DIR)}`",
        f"- Specialist checkpoint: `{SPECIALIST_MODEL_DIR.relative_to(ROOT_DIR)}`",
        f"- Specialist metrics: `{SPECIALIST_METRICS_PATH.relative_to(ROOT_DIR)}`",
        f"- Specialist tokenizer source: `{tokenizer_source}`",
        "",
        "## Outputs",
        "",
        f"- Specialist test probabilities: `{SPECIALIST_TEST_PROB_PATH.relative_to(ROOT_DIR)}`",
        f"- Routed diagnostics: `{ROUTED_DIAGNOSTICS_PATH.relative_to(ROOT_DIR)}`",
        f"- Routed summary: `{ROUTED_SUMMARY_PATH.relative_to(ROOT_DIR)}`",
        "",
        "## Band Results",
        "",
    ]

    for record in summary_records:
        changed = record["changed_vs_roberta_base_submission"]
        changed_text = "unavailable" if changed is None else str(changed)
        lines.extend(
            [
                f"### {record['route_band']}",
                "",
                f"- low_cutoff: `{record['low_cutoff']:.2f}`",
                f"- high_cutoff: `{record['high_cutoff']:.2f}`",
                f"- base_threshold: `{record['base_threshold']:.2f}`",
                f"- specialist_threshold: `{record['specialist_threshold']:.2f}`",
                f"- rows routed to base low: `{record['base_low_rows']}`",
                f"- rows routed to base high: `{record['base_high_rows']}`",
                f"- rows routed to specialist: `{record['specialist_rows']}`",
                f"- final FALSE predictions: `{record['pred_FALSE']}`",
                f"- final TRUE predictions: `{record['pred_TRUE']}`",
                f"- labels changed vs original RoBERTa base submission: `{changed_text}`",
                f"- submission: `{Path(record['submission_path']).relative_to(ROOT_DIR)}`",
                "",
            ]
        )

    ROUTED_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
