from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INSTALL_CMD = "pip install pandas numpy torch transformers tqdm scikit-learn"
SEED = 42
DEFAULT_MAX_LENGTH = 256
EVAL_BATCH_SIZE = 16

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_base"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"
METRICS_PATH = OUTPUT_DIR / "roberta_base_metrics.json"
ORIGINAL_SUBMISSION_PATH = OUTPUT_DIR / "roberta_base_submission.csv"

REGENERATED_SUBMISSION_PATH = OUTPUT_DIR / "verification_regenerated_submission.csv"
VERIFICATION_REPORT_PATH = OUTPUT_DIR / "verification_report.json"



def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def detect_device() -> torch.device:
    # Required order: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)



def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)



def normalize_text(value: object) -> str:
    if pd.isna(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()



def normalize_texts(values: Iterable[object], desc: str) -> List[str]:
    out = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        out.append(normalize_text(value))
    return out



def canonical_label(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Found missing label value.")

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    label = str(value).strip().upper()
    if label in {"TRUE", "FALSE"}:
        return label

    raise ValueError(f"Invalid label value: {value}")



def canonicalize_label_series(series: pd.Series, desc: str) -> pd.Series:
    labels = []
    for value in tqdm(series.tolist(), desc=desc, unit="rows"):
        labels.append(canonical_label(value))
    return pd.Series(labels, index=series.index, dtype="object")



def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    return input_ids, attention_mask



def run_inference(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            probs.append(batch_probs.detach().cpu().numpy())

    return np.concatenate(probs)



def build_submission(solution_format_df: pd.DataFrame, pred_labels: np.ndarray) -> pd.DataFrame:
    if solution_format_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": pred_labels})

    out = solution_format_df.copy()
    out["label"] = pred_labels
    return out[solution_format_df.columns.tolist()]



def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    set_seeds(SEED)
    device = detect_device()
    print(f"Using device: {device}")

    if not BEST_MODEL_DIR.exists():
        raise FileNotFoundError(f"Best model directory not found: {BEST_MODEL_DIR}")

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    if not ORIGINAL_SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"Original submission file not found: {ORIGINAL_SUBMISSION_PATH}")

    print("\n=== Loading files ===")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_format_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    original_submission_df = pd.read_csv(ORIGINAL_SUBMISSION_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    if "text" not in test_df.columns:
        raise ValueError("Test file must contain 'text' column.")

    print("\n=== Validating existing artifacts ===")
    if original_submission_df.shape != solution_format_df.shape:
        raise ValueError(
            f"Submission shape {original_submission_df.shape} does not match solution_format shape {solution_format_df.shape}."
        )

    if original_submission_df.columns.tolist() != solution_format_df.columns.tolist():
        raise ValueError(
            "Submission columns do not exactly match solution_format columns. "
            f"submission={original_submission_df.columns.tolist()}, "
            f"solution_format={solution_format_df.columns.tolist()}"
        )

    if len(test_df) != len(original_submission_df):
        raise ValueError(
            f"Test rows ({len(test_df)}) do not match submission rows ({len(original_submission_df)})."
        )

    if "threshold_tuning" not in metrics or "selected_threshold" not in metrics["threshold_tuning"]:
        raise ValueError("Metrics file missing threshold_tuning.selected_threshold")

    selected_threshold = metrics["threshold_tuning"]["selected_threshold"]
    if not isinstance(selected_threshold, (int, float)):
        raise ValueError(f"Selected threshold is not numeric: {selected_threshold}")
    if not (0.0 <= float(selected_threshold) <= 1.0):
        raise ValueError(f"Selected threshold out of range [0,1]: {selected_threshold}")
    selected_threshold = float(selected_threshold)

    if "label" not in original_submission_df.columns:
        raise ValueError("Submission must contain 'label' column.")

    original_labels = canonicalize_label_series(
        original_submission_df["label"],
        desc="Validating original submission labels",
    )

    unique_labels = set(original_labels.unique().tolist())
    if not unique_labels.issubset({"TRUE", "FALSE"}):
        raise ValueError(f"Submission has invalid labels: {sorted(unique_labels)}")

    max_length = int(metrics.get("config", {}).get("max_length", DEFAULT_MAX_LENGTH))

    print("\n=== Loading saved tokenizer/model ===")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.to(device)

    print("\n=== Tokenizing test text from saved tokenizer ===")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    input_ids, attention_mask = tokenize_texts(
        texts=test_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=256,
        desc="Tokenizing test split",
    )

    test_dataset = TensorDataset(input_ids, attention_mask)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\n=== Running inference from saved checkpoint ===")
    test_prob_true = run_inference(model, test_loader, device=device, desc="Verification inference")

    pred_binary = (test_prob_true >= selected_threshold).astype(int)
    regenerated_labels = np.where(pred_binary == 1, "TRUE", "FALSE")

    regenerated_submission_df = build_submission(solution_format_df, regenerated_labels)
    regenerated_submission_df.to_csv(REGENERATED_SUBMISSION_PATH, index=False)

    regenerated_norm_df = regenerated_submission_df.copy()
    regenerated_norm_df["label"] = canonicalize_label_series(
        regenerated_norm_df["label"],
        desc="Normalizing regenerated labels",
    )

    original_norm_df = original_submission_df.copy()
    original_norm_df["label"] = original_labels

    diff_mask = (original_norm_df != regenerated_norm_df).any(axis=1)
    differing_rows = int(diff_mask.sum())
    exact_match = differing_rows == 0

    original_dist = original_norm_df["label"].value_counts().to_dict()
    regenerated_dist = regenerated_norm_df["label"].value_counts().to_dict()

    report = {
        "device": str(device),
        "selected_threshold": selected_threshold,
        "max_length": max_length,
        "test_rows": int(len(test_df)),
        "solution_format_shape": list(solution_format_df.shape),
        "original_submission_shape": list(original_submission_df.shape),
        "regenerated_submission_shape": list(regenerated_submission_df.shape),
        "columns_match_solution_format": original_submission_df.columns.tolist() == solution_format_df.columns.tolist(),
        "exact_match_with_original_submission": exact_match,
        "differing_rows": differing_rows,
        "differing_row_indices_preview": np.where(diff_mask.to_numpy())[0][:20].tolist(),
        "original_prediction_distribution": original_dist,
        "regenerated_prediction_distribution": regenerated_dist,
        "paths": {
            "test": str(TEST_PATH),
            "solution_format": str(SOLUTION_FORMAT_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "metrics": str(METRICS_PATH),
            "original_submission": str(ORIGINAL_SUBMISSION_PATH),
            "regenerated_submission": str(REGENERATED_SUBMISSION_PATH),
            "verification_report": str(VERIFICATION_REPORT_PATH),
        },
    }

    VERIFICATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Verification Summary ===")
    print(f"Exact submission match: {exact_match}")
    print(f"Differing rows: {differing_rows}")
    print(f"Original prediction distribution: {original_dist}")
    print(f"Regenerated prediction distribution: {regenerated_dist}")
    print(f"Selected threshold used: {selected_threshold:.4f}")
    print(f"Device used: {device}")
    print(f"Saved regenerated submission: {REGENERATED_SUBMISSION_PATH}")
    print(f"Saved verification report: {VERIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
