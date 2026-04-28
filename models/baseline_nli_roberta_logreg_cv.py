# pip install pandas numpy scikit-learn tqdm sentence-transformers torch

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

SEED = 42
N_SPLITS = 5
MODEL_NAME = "sentence-transformers/nli-roberta-base-v2"
INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm sentence-transformers torch"
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "baseline_nli_roberta_logreg_cv"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

TRAIN_EMB_CACHE_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_train_embeddings.npy"
TEST_EMB_CACHE_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_test_embeddings.npy"
TRAIN_EMB_META_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_train_embeddings_meta.json"
TEST_EMB_META_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_test_embeddings_meta.json"

METRICS_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_metrics.json"
OOF_PRED_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_oof_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_error_analysis.csv"
SUBMISSION_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_submission.csv"
NOTES_PATH = OUTPUT_DIR / "baseline_nli_roberta_logreg_cv_notes.md"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def detect_device() -> str:
    # Requirement: prioritize MPS on Mac, else CPU.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"



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
    normalized = []
    for value in tqdm(list(values), desc=desc, unit="rows"):
        normalized.append(normalize_text(value))
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
        encoded.append(1 if normalize_label(value) == "TRUE" else 0)
    return np.asarray(encoded, dtype=np.int32)



def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")



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



def compute_text_signature(texts: List[str], desc: str) -> str:
    hasher = hashlib.sha256()
    for text in tqdm(texts, desc=desc, unit="rows"):
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()



def load_or_create_embeddings(
    texts: List[str],
    cache_npy_path: Path,
    cache_meta_path: Path,
    model: SentenceTransformer,
    model_name: str,
    batch_size: int,
    split_name: str,
) -> np.ndarray:
    text_signature = compute_text_signature(texts, desc=f"Hashing {split_name} texts")

    if cache_npy_path.exists() and cache_meta_path.exists():
        try:
            cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if (
                cache_meta.get("model_name") == model_name
                and cache_meta.get("num_rows") == len(texts)
                and cache_meta.get("text_signature") == text_signature
            ):
                embeddings = np.load(cache_npy_path)
                if embeddings.shape[0] == len(texts):
                    print(f"Using cached {split_name} embeddings: {cache_npy_path}")
                    return embeddings.astype(np.float32)
        except Exception:
            print(f"Cache validation failed for {split_name}. Regenerating embeddings.")

    print(f"Generating {split_name} embeddings with {model_name}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(cache_npy_path, embeddings)
    cache_meta = {
        "model_name": model_name,
        "num_rows": len(texts),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "text_signature": text_signature,
    }
    cache_meta_path.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")

    print(f"Saved {split_name} embeddings cache: {cache_npy_path}")
    return embeddings



def metric_bundle(y_true: np.ndarray, prob_true: np.ndarray, threshold: float) -> dict:
    pred = (prob_true >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_true)),
        "confusion_matrix": {
            "labels": ["FALSE", "TRUE"],
            "matrix": cm.tolist(),
            "format": "[[TN, FP], [FN, TP]]",
        },
        "prediction_distribution": {
            "pred_FALSE": int((pred == 0).sum()),
            "pred_TRUE": int((pred == 1).sum()),
        },
    }



def tune_threshold_for_f1(y_true: np.ndarray, prob_true: np.ndarray, thresholds: np.ndarray) -> tuple[float, List[dict]]:
    records = []
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in tqdm(thresholds, desc="Threshold search (OOF)", unit="threshold"):
        pred = (prob_true >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        tie_break_current = abs(float(threshold) - 0.5)
        tie_break_best = abs(best_threshold - 0.5)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and tie_break_current < tie_break_best):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, records



def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
        solver="liblinear",
    )



def main() -> None:
    print("\nDependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)

    device = detect_device()
    print(f"\nEmbedding device: {device}")

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

    print("\n=== Preparing text and labels ===")
    train_texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    print("\n=== Loading embedding model ===")
    embed_model = SentenceTransformer(MODEL_NAME, device=device)

    print("\n=== Building/Loading embeddings ===")
    train_embeddings = load_or_create_embeddings(
        texts=train_texts,
        cache_npy_path=TRAIN_EMB_CACHE_PATH,
        cache_meta_path=TRAIN_EMB_META_PATH,
        model=embed_model,
        model_name=MODEL_NAME,
        batch_size=128,
        split_name="train",
    )
    test_embeddings = load_or_create_embeddings(
        texts=test_texts,
        cache_npy_path=TEST_EMB_CACHE_PATH,
        cache_meta_path=TEST_EMB_META_PATH,
        model=embed_model,
        model_name=MODEL_NAME,
        batch_size=128,
        split_name="test",
    )

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    print("\n=== Cross-validation (StratifiedKFold) ===")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    n_train = train_embeddings.shape[0]
    n_test = test_embeddings.shape[0]

    oof_prob_true = np.zeros(n_train, dtype=np.float64)
    oof_seen_mask = np.zeros(n_train, dtype=bool)
    fold_test_prob_true = np.zeros((N_SPLITS, n_test), dtype=np.float64)
    fold_metrics: List[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(skf.split(train_embeddings, y), total=N_SPLITS, desc="CV folds", unit="fold"),
        start=1,
    ):
        print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")
        x_tr = train_embeddings[tr_idx]
        y_tr = y[tr_idx]
        x_val = train_embeddings[val_idx]
        y_val = y[val_idx]

        clf = build_classifier()
        clf.fit(x_tr, y_tr)

        val_prob_true = clf.predict_proba(x_val)[:, 1]
        val_pred = (val_prob_true >= 0.5).astype(int)

        fold_acc = accuracy_score(y_val, val_pred)
        fold_f1 = f1_score(y_val, val_pred, zero_division=0)
        fold_auc = roc_auc_score(y_val, val_prob_true)

        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(tr_idx)),
                "n_val": int(len(val_idx)),
                "accuracy": float(fold_acc),
                "f1": float(fold_f1),
                "roc_auc": float(fold_auc),
            }
        )

        print(
            f"Fold {fold_idx} metrics | "
            f"accuracy={fold_acc:.6f}, f1={fold_f1:.6f}, roc_auc={fold_auc:.6f}"
        )

        oof_prob_true[val_idx] = val_prob_true
        oof_seen_mask[val_idx] = True
        fold_test_prob_true[fold_idx - 1] = clf.predict_proba(test_embeddings)[:, 1]

    if not np.all(oof_seen_mask):
        missing = int((~oof_seen_mask).sum())
        raise RuntimeError(f"OOF predictions missing for {missing} rows.")

    print("\n=== Threshold tuning on OOF ===")
    best_threshold, threshold_table = tune_threshold_for_f1(y, oof_prob_true, THRESHOLD_GRID)
    print(f"Selected threshold (best OOF F1): {best_threshold:.2f}")

    print("\n=== OOF evaluation ===")
    oof_metrics = metric_bundle(y_true=y, prob_true=oof_prob_true, threshold=best_threshold)

    print(f"OOF Accuracy:  {oof_metrics['accuracy']:.6f}")
    print(f"OOF Precision: {oof_metrics['precision']:.6f}")
    print(f"OOF Recall:    {oof_metrics['recall']:.6f}")
    print(f"OOF F1:        {oof_metrics['f1']:.6f}")
    print(f"OOF ROC AUC:   {oof_metrics['roc_auc']:.6f}")
    print(f"OOF confusion matrix [[TN, FP], [FN, TP]]: {oof_metrics['confusion_matrix']['matrix']}")
    print(f"OOF prediction distribution: {oof_metrics['prediction_distribution']}")

    oof_pred = (oof_prob_true >= best_threshold).astype(int)
    oof_pred_df = pd.DataFrame(
        {
            "text": train_texts,
            "true_label": decode_labels(y),
            "oof_pred_label": decode_labels(oof_pred),
            "oof_prob_TRUE": oof_prob_true,
        }
    )
    oof_pred_df.to_csv(OOF_PRED_PATH, index=False)
    print(f"Saved OOF predictions: {OOF_PRED_PATH}")

    print("\n=== OOF error analysis ===")
    error_df = oof_pred_df[oof_pred_df["true_label"] != oof_pred_df["oof_pred_label"]].copy()
    error_df["error_type"] = np.where(
        (error_df["true_label"] == "FALSE") & (error_df["oof_pred_label"] == "TRUE"),
        "false_positive",
        "false_negative",
    )
    error_df["confidence"] = np.where(
        error_df["error_type"] == "false_positive",
        error_df["oof_prob_TRUE"],
        1.0 - error_df["oof_prob_TRUE"],
    )

    top_k = 200
    fp_df = error_df[error_df["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(top_k)
    fn_df = error_df[error_df["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(top_k)
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)

    error_df = error_df[["text", "true_label", "oof_pred_label", "oof_prob_TRUE", "error_type", "confidence"]]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved OOF error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Final test inference (fold probability averaging) ===")
    ensemble_test_prob_true = fold_test_prob_true.mean(axis=0)
    test_pred = (ensemble_test_prob_true >= best_threshold).astype(int)
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

    metrics = {
        "model": "NLI-RoBERTa sentence embeddings + LogisticRegression",
        "seed": SEED,
        "embedding_model": MODEL_NAME,
        "device": device,
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "fold_metrics": fold_metrics,
        },
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_oof_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "oof": oof_metrics,
        "inference": {
            "approach": "mean_probability_across_fold_models",
            "prediction_distribution": test_distribution,
            "ensemble_test_prob_summary": {
                "mean": float(np.mean(ensemble_test_prob_true)),
                "std": float(np.std(ensemble_test_prob_true)),
                "min": float(np.min(ensemble_test_prob_true)),
                "max": float(np.max(ensemble_test_prob_true)),
            },
        },
        "cache": {
            "train_embeddings": str(TRAIN_EMB_CACHE_PATH),
            "test_embeddings": str(TEST_EMB_CACHE_PATH),
            "train_meta": str(TRAIN_EMB_META_PATH),
            "test_meta": str(TEST_EMB_META_PATH),
        },
        "paths": {
            "oof_predictions": str(OOF_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes = f"""# Baseline NLI-RoBERTa Embeddings + Logistic Regression (CV)

## Summary

- Embedding model: `{MODEL_NAME}`
- Classifier: Logistic Regression (`class_weight='balanced'`, `solver='liblinear'`, `max_iter=2000`)
- Validation: 5-fold StratifiedKFold with OOF predictions
- Threshold selection: best OOF F1 over thresholds 0.30 to 0.70
- Inference: fold-model probability averaging on test

## Why This Baseline

This tests whether an NLI-trained RoBERTa sentence embedding model gives better semantic separation than the previous MiniLM embedding baseline.

This is not transformer fine-tuning. The encoder is used only to generate cached sentence embeddings, then a Logistic Regression meta-classifier is trained with 5-fold CV.

Expected tradeoff: this should capture richer sentence-level semantics than MiniLM, but it will be slower and may still miss dataset-specific style/noise artifacts that full supervised fine-tuning can learn.

## Outputs

- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_metrics.json`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_oof_predictions.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_error_analysis.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_submission.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_notes.md`

## Cache Files

- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_train_embeddings.npy`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_test_embeddings.npy`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_train_embeddings_meta.json`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_test_embeddings_meta.json`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
