# pip install pandas numpy scikit-learn tqdm

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm import tqdm

SEED = 42
INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BASE_OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR = BASE_OUTPUT_DIR / "baseline_tfidf_logreg"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "baseline_tfidf_logreg_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "baseline_tfidf_logreg_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "baseline_tfidf_logreg_error_analysis.csv"
SUBMISSION_PATH = OUTPUT_DIR / "baseline_tfidf_logreg_submission.csv"
NOTES_PATH = OUTPUT_DIR / "baseline_tfidf_logreg_notes.md"


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


def normalize_texts(values: Iterable[object], desc: str) -> List[str]:
    normalized = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        if pd.isna(value):
            text = ""
        elif isinstance(value, str):
            text = value
        else:
            text = str(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized.append(text)
    return normalized


def normalize_label(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Found missing label value.")

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"

    raise ValueError(f"Unexpected label value: {value}")


def encode_labels(values: Iterable[object], desc: str) -> np.ndarray:
    encoded = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        normalized = normalize_label(value)
        encoded.append(1 if normalized == "TRUE" else 0)
    return np.asarray(encoded, dtype=np.int32)


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def build_pipeline() -> Pipeline:
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
    )

    char_tfidf = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
    )

    features = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
        solver="saga",
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )


def predict_proba_in_batches(model: Pipeline, texts: List[str], desc: str, batch_size: int = 512) -> np.ndarray:
    probs = np.zeros(len(texts), dtype=np.float64)
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, len(texts))
        probs[start:end] = model.predict_proba(texts[start:end])[:, 1]
    return probs


def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    required_train_cols = {"label", "text"}
    required_test_cols = {"text"}

    missing_train_cols = sorted(required_train_cols - set(train_df.columns))
    if missing_train_cols:
        raise ValueError(f"Train file is missing required columns: {missing_train_cols}")

    missing_test_cols = sorted(required_test_cols - set(test_df.columns))
    if missing_test_cols:
        raise ValueError(f"Test file is missing required columns: {missing_test_cols}")

    if len(solution_df) != len(test_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )

    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")


def main() -> None:
    print("\nDependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    validate_inputs(train_df, test_df, solution_df)
    print("Input schema checks passed.")
    print(f"Expected submission columns: {solution_df.columns.tolist()}")

    print("\n=== Building features ===")
    train_texts = normalize_texts(train_df["text"], desc="Preparing train text")
    test_texts = normalize_texts(test_df["text"], desc="Preparing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print(f"Train split size: {len(X_train)}")
    print(f"Validation split size: {len(X_val)}")

    model = build_pipeline()

    print("\n=== Training model ===")
    model.fit(X_train, y_train)

    print("\n=== Evaluating validation split ===")
    val_prob_true = predict_proba_in_batches(model, X_val, desc="Validation prediction")
    val_pred = (val_prob_true >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred, zero_division=0)
    recall = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, val_prob_true)
    cm = confusion_matrix(y_val, val_pred, labels=[0, 1])

    val_distribution = {
        "pred_FALSE": int((val_pred == 0).sum()),
        "pred_TRUE": int((val_pred == 1).sum()),
    }

    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"ROC AUC:   {roc_auc:.6f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")
    print(f"Validation prediction distribution: {val_distribution}")

    metrics = {
        "model": "TF-IDF (word+char) + LogisticRegression",
        "seed": SEED,
        "validation": {
            "n_samples": int(len(X_val)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": {
                "labels": ["FALSE", "TRUE"],
                "matrix": cm.tolist(),
                "format": "[[TN, FP], [FN, TP]]",
            },
            "prediction_distribution": val_distribution,
        },
        "paths": {
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "submission": str(SUBMISSION_PATH),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    val_pred_df = pd.DataFrame(
        {
            "text": X_val,
            "true_label": decode_labels(y_val),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": val_prob_true,
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    fp_df = val_pred_df[(val_pred_df["true_label"] == "FALSE") & (val_pred_df["pred_label"] == "TRUE")].copy()
    fp_df["error_type"] = "false_positive"
    fp_df["confidence"] = fp_df["pred_prob_TRUE"]

    fn_df = val_pred_df[(val_pred_df["true_label"] == "TRUE") & (val_pred_df["pred_label"] == "FALSE")].copy()
    fn_df["error_type"] = "false_negative"
    fn_df["confidence"] = 1.0 - fn_df["pred_prob_TRUE"]

    top_k = 200
    error_df = pd.concat(
        [
            fp_df.sort_values("confidence", ascending=False).head(top_k),
            fn_df.sort_values("confidence", ascending=False).head(top_k),
        ],
        ignore_index=True,
    )
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Training final model ===")
    final_model = build_pipeline()
    final_model.fit(train_texts, y)

    print("\n=== Generating submission ===")
    test_prob_true = predict_proba_in_batches(final_model, test_texts, desc="Test prediction")
    test_pred = (test_prob_true >= 0.5).astype(int)
    test_pred_labels = decode_labels(test_pred)

    test_distribution = {
        "pred_FALSE": int((test_pred == 0).sum()),
        "pred_TRUE": int((test_pred == 1).sum()),
    }
    print(f"Final test prediction distribution: {test_distribution}")

    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": test_pred_labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = test_pred_labels

    submission_df = submission_df[solution_df.columns.tolist()]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")

    notes = f"""# Baseline TF-IDF + Logistic Regression Notes

## What Was Trained

- Model: word-level TF-IDF + character-level TF-IDF + Logistic Regression (`class_weight=balanced`)
- Seed: {SEED}
- Data: `data/train_labeled_comp.jsonl` for training/validation, `data/test_labeled_comp.jsonl` for inference

## Why This Baseline

This baseline is designed to quickly test whether lexical weirdness, character-level noise, and surface text structure already separate `TRUE` and `FALSE` labels.

It aligns with the current working hypothesis that `TRUE` may capture harmful/unsafe/abnormal language behavior, including incoherence and spam-like artifacts, not only explicit toxicity.

## Validation Metrics

- Accuracy: {accuracy:.6f}
- Precision: {precision:.6f}
- Recall: {recall:.6f}
- F1: {f1:.6f}
- ROC AUC: {roc_auc:.6f}
- Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}
- Validation prediction distribution: {val_distribution}

## Files Created

- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_metrics.json`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_val_predictions.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_error_analysis.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_submission.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_notes.md`

## What To Inspect Next

- Inspect confident false positives and false negatives in `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_error_analysis.csv`.
- Check if errors correlate with incoherence, token noise, formatting artifacts, or ambiguous semantics.
- Decide whether the next iteration should add explicit weirdness/fluency features or threshold tuning before heavier models.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
