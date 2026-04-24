# pip install pandas numpy scikit-learn tqdm

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SEED = 42
N_SPLITS = 5
INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "baseline_tfidf_features_logreg_cv"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "baseline_tfidf_features_logreg_cv_metrics.json"
OOF_PRED_PATH = OUTPUT_DIR / "baseline_tfidf_features_logreg_cv_oof_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "baseline_tfidf_features_logreg_cv_error_analysis.csv"
SUBMISSION_PATH = OUTPUT_DIR / "baseline_tfidf_features_logreg_cv_submission.csv"
NOTES_PATH = OUTPUT_DIR / "baseline_tfidf_features_logreg_cv_notes.md"

WORD_RE = re.compile(r"\b\w+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")

ENGINEERED_FEATURE_COLUMNS = [
    "char_length",
    "word_count",
    "unique_word_ratio",
    "avg_word_length",
    "digit_ratio",
    "uppercase_ratio",
    "punctuation_ratio",
    "non_alphanumeric_ratio",
    "newline_count",
    "url_count",
    "repeated_token_count",
    "very_long_token_count",
    "short_sentence_count",
    "long_sentence_count",
    "approx_sentence_count",
]


class EngineeredFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that computes numeric text features."""

    def fit(self, X: Sequence[str], y: np.ndarray | None = None) -> "EngineeredFeatureTransformer":
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        texts = ensure_text_list(X)
        return extract_engineered_feature_matrix(texts, desc="Extracting engineered features")



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



def ensure_text_list(values: Sequence[object]) -> List[str]:
    return [normalize_text(v) for v in values]



def sentence_word_counts(text: str) -> List[int]:
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if not parts and text:
        parts = [text]
    counts = []
    for part in parts:
        counts.append(len(WORD_RE.findall(part)))
    return counts



def extract_engineered_feature_row(text: str) -> List[float]:
    char_length = len(text)
    safe_char_length = max(char_length, 1)

    tokens = WORD_RE.findall(text)
    lower_tokens = [token.lower() for token in tokens]
    word_count = len(lower_tokens)

    unique_word_ratio = (len(set(lower_tokens)) / word_count) if word_count > 0 else 0.0
    avg_word_length = float(np.mean([len(t) for t in lower_tokens])) if word_count > 0 else 0.0

    digit_ratio = sum(ch.isdigit() for ch in text) / safe_char_length

    alpha_count = sum(ch.isalpha() for ch in text)
    uppercase_count = sum(ch.isupper() for ch in text)
    uppercase_ratio = uppercase_count / max(alpha_count, 1)

    punctuation_ratio = sum(ch in string.punctuation for ch in text) / safe_char_length
    non_alphanumeric_ratio = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / safe_char_length

    newline_count = text.count("\n")
    url_count = len(URL_RE.findall(text))

    token_counts = Counter(lower_tokens)
    repeated_token_count = sum(count - 1 for count in token_counts.values() if count > 1)
    very_long_token_count = sum(len(token) >= 15 for token in lower_tokens)

    sentence_counts = sentence_word_counts(text)
    approx_sentence_count = len(sentence_counts)
    short_sentence_count = sum(count <= 5 for count in sentence_counts if count > 0)
    long_sentence_count = sum(count >= 25 for count in sentence_counts)

    return [
        float(char_length),
        float(word_count),
        float(unique_word_ratio),
        float(avg_word_length),
        float(digit_ratio),
        float(uppercase_ratio),
        float(punctuation_ratio),
        float(non_alphanumeric_ratio),
        float(newline_count),
        float(url_count),
        float(repeated_token_count),
        float(very_long_token_count),
        float(short_sentence_count),
        float(long_sentence_count),
        float(approx_sentence_count),
    ]



def extract_engineered_feature_matrix(texts: Sequence[str], desc: str) -> np.ndarray:
    matrix = []
    for text in tqdm(texts, desc=desc, unit="rows"):
        matrix.append(extract_engineered_feature_row(text))
    return np.asarray(matrix, dtype=np.float64)



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

    numeric_pipeline = Pipeline(
        steps=[
            ("engineered", EngineeredFeatureTransformer()),
            ("scaler", StandardScaler()),
        ]
    )

    combined_features = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
            ("engineered_features", numeric_pipeline),
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
            ("features", combined_features),
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

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        records.append(row)

        tie_break_current = abs(float(threshold) - 0.5)
        tie_break_best = abs(best_threshold - 0.5)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and tie_break_current < tie_break_best):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, records



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

    print("\n=== Building base arrays ===")
    train_texts = normalize_texts(train_df["text"], desc="Preparing train text")
    test_texts = normalize_texts(test_df["text"], desc="Preparing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    train_texts_array = np.asarray(train_texts, dtype=object)
    n_train = len(train_texts)
    n_test = len(test_texts)

    print(f"Train rows: {n_train}")
    print(f"Test rows:  {n_test}")

    print("\n=== Cross-validation (StratifiedKFold) ===")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_prob_true = np.zeros(n_train, dtype=np.float64)
    oof_seen_mask = np.zeros(n_train, dtype=bool)
    fold_test_prob_true = np.zeros((N_SPLITS, n_test), dtype=np.float64)
    fold_metrics: List[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(train_texts_array, y), total=N_SPLITS, desc="CV folds", unit="fold"),
        start=1,
    ):
        print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")

        x_fold_train = train_texts_array[train_idx].tolist()
        x_fold_val = train_texts_array[val_idx].tolist()
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]

        fold_model = build_pipeline()

        print(f"Training fold model {fold_idx}")
        fold_model.fit(x_fold_train, y_fold_train)

        val_prob_true = predict_proba_in_batches(
            fold_model,
            x_fold_val,
            desc=f"Fold {fold_idx} validation prediction",
        )
        val_pred = (val_prob_true >= 0.5).astype(int)

        fold_accuracy = accuracy_score(y_fold_val, val_pred)
        fold_f1 = f1_score(y_fold_val, val_pred, zero_division=0)
        fold_roc_auc = roc_auc_score(y_fold_val, val_prob_true)

        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "accuracy": float(fold_accuracy),
                "f1": float(fold_f1),
                "roc_auc": float(fold_roc_auc),
            }
        )

        print(
            f"Fold {fold_idx} metrics | "
            f"accuracy={fold_accuracy:.6f}, f1={fold_f1:.6f}, roc_auc={fold_roc_auc:.6f}"
        )

        oof_prob_true[val_idx] = val_prob_true
        oof_seen_mask[val_idx] = True

        fold_test_prob_true[fold_idx - 1] = predict_proba_in_batches(
            fold_model,
            test_texts,
            desc=f"Fold {fold_idx} test prediction",
        )

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
    full_feature_matrix = extract_engineered_feature_matrix(
        train_texts,
        desc="Building engineered features for OOF error analysis",
    )
    feature_df = pd.DataFrame(full_feature_matrix, columns=ENGINEERED_FEATURE_COLUMNS)

    error_df = pd.concat([oof_pred_df.reset_index(drop=True), feature_df], axis=1)
    error_df = error_df[error_df["true_label"] != error_df["oof_pred_label"]].copy()

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

    leading_cols = ["text", "true_label", "oof_pred_label", "oof_prob_TRUE", "error_type", "confidence"]
    error_df = error_df[leading_cols + ENGINEERED_FEATURE_COLUMNS]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved OOF error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Training final model on full data ===")
    final_model = build_pipeline()
    final_model.fit(train_texts, y)

    print("\n=== Generating test predictions ===")
    full_model_test_prob_true = predict_proba_in_batches(
        final_model,
        test_texts,
        desc="Full-data model test prediction",
    )

    # Ensemble approach: average probabilities from fold models.
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
        "model": "TF-IDF (word+char+engineered numeric) + LogisticRegression",
        "seed": SEED,
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "fold_metrics": fold_metrics,
        },
        "engineered_features": ENGINEERED_FEATURE_COLUMNS,
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
            "full_model_trained_on_all_data": True,
            "full_model_test_prob_summary": {
                "mean": float(np.mean(full_model_test_prob_true)),
                "std": float(np.std(full_model_test_prob_true)),
                "min": float(np.min(full_model_test_prob_true)),
                "max": float(np.max(full_model_test_prob_true)),
            },
            "ensemble_test_prob_summary": {
                "mean": float(np.mean(ensemble_test_prob_true)),
                "std": float(np.std(ensemble_test_prob_true)),
                "min": float(np.min(ensemble_test_prob_true)),
                "max": float(np.max(ensemble_test_prob_true)),
            },
            "prediction_distribution": test_distribution,
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

    notes = f"""# Baseline TF-IDF + Engineered Features + Logistic Regression (CV)

## What Changed

- Replaced single split validation with 5-fold Stratified K-Fold (`shuffle=True`, `random_state=42`).
- Generated OOF predictions across all training rows.
- Tuned classification threshold on OOF probabilities by maximizing F1 over thresholds 0.30 to 0.70.
- Used OOF predictions for error analysis.
- Trained a full-data model and used fold-model probability averaging for final test inference.

## Pipeline

- Word TF-IDF (`ngram_range=(1,2)`, `min_df=2`, `max_features=100000`)
- Char TF-IDF (`analyzer='char_wb'`, `ngram_range=(3,5)`, `min_df=2`, `max_features=100000`)
- Engineered numeric features ({len(ENGINEERED_FEATURE_COLUMNS)} features) + `StandardScaler`
- Logistic Regression (`class_weight='balanced'`, `solver='saga'`, `max_iter=2000`, `random_state=42`)

## Why This Setup

This configuration improves validation stability and reduces dependence on one lucky/unlucky split.
It directly tests whether explicit surface abnormality/fluency features improve signal over TF-IDF alone.

## Outputs

- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_metrics.json`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_oof_predictions.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_error_analysis.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_submission.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_notes.md`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
