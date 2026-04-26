#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"

SEED = 42
N_SPLITS = 5
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_label_style_calibrator"

ROBERTA_VAL_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv"
ROBERTA_TEST_PROB_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_test_probabilities.csv"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_metrics.json"
TEST_SCORE_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_test_scores.csv"
VAL_SCORE_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_val_scores.csv"
COEFFICIENTS_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_coefficients.csv"
MANIFEST_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_submission_manifest.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_label_style_calibrator_notes.md"

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
TARGET_TRUE_COUNTS = [574, 577, 578, 580]

FEATURE_SETS = {
    "roberta_only": ["roberta_logit"],
    "core_style": [
        "roberta_logit",
        "char_length",
        "word_count",
        "sentence_count",
        "avg_sentence_word_count",
        "unique_word_ratio",
        "stopword_ratio",
        "spam_keyword_count",
        "risk_keyword_count",
        "prompt_template_flag",
        "decode_error_flag",
    ],
    "noise_style": [
        "roberta_logit",
        "char_length",
        "word_count",
        "avg_word_length",
        "max_word_length",
        "sentence_count",
        "unique_word_ratio",
        "stopword_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "punctuation_ratio",
        "non_ascii_ratio",
        "non_alnum_ratio",
        "code_symbol_count",
        "weird_token_ratio",
        "repeated_token_count",
        "very_long_token_count",
        "url_domain_count",
        "spam_keyword_count",
        "risk_keyword_count",
        "prompt_template_flag",
        "decode_error_flag",
    ],
}

SUBMISSION_VARIANTS = [
    ("rankblend0p90", "rank_blend_roberta_0p90"),
    ("rankblend0p85", "rank_blend_roberta_0p85"),
    ("rankblend0p80", "rank_blend_roberta_0p80"),
    ("calibrated", "calibrated_prob_TRUE"),
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

SPAM_TERMS = {
    "affiliate",
    "buy",
    "click",
    "coupon",
    "deal",
    "free",
    "offer",
    "order",
    "product",
    "products",
    "promo",
    "sale",
    "shopping",
    "subscribe",
}

RISK_TERMS = {
    "attack",
    "combat",
    "danger",
    "exploit",
    "fight",
    "harm",
    "hate",
    "kill",
    "threat",
    "weapon",
}

PROMPT_PATTERN = re.compile(
    r"based on the passage above|answer:|question:|could you|can you|what is|how did|why did|paraphrase",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|io|gov|edu)\b", re.IGNORECASE)
CODE_PATTERN = re.compile(
    r"\{|\}|\[|\]|=>|==|</|/>|function\b|var\b|const\b|let\b|def\b|class\b|model\(|user_story|setvar",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+|\n+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value}")


def encode_label(value: object) -> int:
    return 1 if normalize_label(value) == "TRUE" else 0


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def safe_logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(prob / (1 - prob))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def count_adjacent_repeats(tokens: list[str]) -> int:
    return sum(1 for prev, cur in zip(tokens, tokens[1:]) if prev == cur)


def text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    chars = len(text)
    tokens = WORD_PATTERN.findall(text)
    lower_tokens = [token.lower() for token in tokens]
    token_count = len(tokens)
    alpha_count = sum(ch.isalpha() for ch in text)
    uppercase_count = sum(ch.isupper() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    punctuation_count = sum(ch in string.punctuation for ch in text)
    whitespace_count = sum(ch.isspace() for ch in text)
    non_ascii_count = sum(ord(ch) > 127 for ch in text)
    alnum_count = sum(ch.isalnum() for ch in text)
    sentence_parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    sentence_count = len(sentence_parts)
    token_lengths = [len(token) for token in tokens]
    spam_keyword_count = sum(token in SPAM_TERMS for token in lower_tokens)
    risk_keyword_count = sum(token in RISK_TERMS for token in lower_tokens)

    weird_tokens = 0
    for token in tokens:
        has_digit = any(ch.isdigit() for ch in token)
        has_alpha = any(ch.isalpha() for ch in token)
        has_non_ascii = any(ord(ch) > 127 for ch in token)
        has_repeated_chars = bool(REPEATED_CHAR_PATTERN.search(token))
        has_internal_punct = bool(re.search(r"[_/\\-]", token))
        if (has_digit and has_alpha) or has_non_ascii or has_repeated_chars or has_internal_punct:
            weird_tokens += 1

    return {
        "char_length": float(chars),
        "word_count": float(token_count),
        "avg_word_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "max_word_length": float(max(token_lengths)) if token_lengths else 0.0,
        "unique_word_ratio": safe_ratio(len(set(lower_tokens)), token_count),
        "stopword_ratio": safe_ratio(sum(token in STOPWORDS for token in lower_tokens), token_count),
        "digit_ratio": safe_ratio(digit_count, chars),
        "uppercase_ratio": safe_ratio(uppercase_count, alpha_count),
        "punctuation_ratio": safe_ratio(punctuation_count, chars),
        "whitespace_ratio": safe_ratio(whitespace_count, chars),
        "non_ascii_ratio": safe_ratio(non_ascii_count, chars),
        "non_alnum_ratio": safe_ratio(chars - alnum_count, chars),
        "newline_count": float(text.count("\n")),
        "url_domain_count": float(len(URL_PATTERN.findall(text))),
        "code_symbol_count": float(len(CODE_PATTERN.findall(text))),
        "decode_error_flag": float("[DECODE ERROR]" in text),
        "prompt_template_flag": float(bool(PROMPT_PATTERN.search(text))),
        "spam_keyword_count": float(spam_keyword_count),
        "risk_keyword_count": float(risk_keyword_count),
        "repeated_token_count": float(count_adjacent_repeats(lower_tokens)),
        "very_long_token_count": float(sum(length >= 20 for length in token_lengths)),
        "weird_token_ratio": safe_ratio(weird_tokens, token_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_word_count": safe_ratio(token_count, sentence_count),
    }


def make_feature_frame(texts: Iterable[str], desc: str) -> pd.DataFrame:
    rows = []
    for text in tqdm(list(texts), desc=desc, unit="rows"):
        rows.append(text_features(text))
    return pd.DataFrame(rows)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ROBERTA_VAL_PATH.exists():
        raise FileNotFoundError(f"Missing RoBERTa validation predictions: {ROBERTA_VAL_PATH}")
    if not ROBERTA_TEST_PROB_PATH.exists():
        raise FileNotFoundError(f"Missing RoBERTa test probabilities: {ROBERTA_TEST_PROB_PATH}")

    val_df = pd.read_csv(ROBERTA_VAL_PATH)
    test_df = pd.read_csv(ROBERTA_TEST_PROB_PATH)
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)

    required_val = {"text", "true_label", "pred_prob_TRUE"}
    required_test = {"text", "pred_prob_TRUE"}
    if missing := sorted(required_val - set(val_df.columns)):
        raise ValueError(f"Validation predictions missing columns: {missing}")
    if missing := sorted(required_test - set(test_df.columns)):
        raise ValueError(f"Test probabilities missing columns: {missing}")
    if len(test_df) != len(solution_df):
        raise ValueError(f"Test probability rows ({len(test_df)}) do not match solution rows ({len(solution_df)}).")
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a label column.")

    val_df["text"] = val_df["text"].map(normalize_text)
    test_df["text"] = test_df["text"].map(normalize_text)
    val_df["true_label_norm"] = val_df["true_label"].map(normalize_label)
    val_df["y"] = val_df["true_label"].map(encode_label).astype(int)
    val_df["roberta_prob_TRUE"] = pd.to_numeric(val_df["pred_prob_TRUE"], errors="raise").astype(float)
    test_df["roberta_prob_TRUE"] = pd.to_numeric(test_df["pred_prob_TRUE"], errors="raise").astype(float)
    return val_df, test_df, solution_df


def add_features(df: pd.DataFrame, desc: str) -> pd.DataFrame:
    features = make_feature_frame(df["text"], desc=desc)
    out = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["roberta_logit"] = safe_logit(out["roberta_prob_TRUE"].to_numpy())
    return out


def make_pipeline(c_value: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_candidate(
    data: pd.DataFrame,
    feature_cols: list[str],
    c_value: float,
    class_weight: str | None,
) -> dict:
    y = data["y"].to_numpy(dtype=int)
    x = data[feature_cols].to_numpy(dtype=float)
    oof_prob = np.zeros(len(data), dtype=float)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        model = make_pipeline(c_value=c_value, class_weight=class_weight)
        model.fit(x[train_idx], y[train_idx])
        oof_prob[val_idx] = model.predict_proba(x[val_idx])[:, 1]

    best_threshold, threshold_table = tune_threshold(y, oof_prob)
    pred = (oof_prob >= best_threshold).astype(int)
    return {
        "feature_set": None,
        "features": feature_cols,
        "c": float(c_value),
        "class_weight": class_weight,
        "selected_threshold": float(best_threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_prob)),
        "oof_prob": oof_prob,
        "threshold_table": threshold_table,
    }


def tune_threshold(y_true: np.ndarray, prob_true: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        pred = (prob_true >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and abs(float(threshold) - 0.32) < abs(best_threshold - 0.32)):
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, rows


def select_calibrator(val_features: pd.DataFrame) -> tuple[dict, list[dict]]:
    candidates = []
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    class_weights: list[str | None] = [None, "balanced"]

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        for c_value in c_values:
            for class_weight in class_weights:
                result = evaluate_candidate(
                    data=val_features,
                    feature_cols=feature_cols,
                    c_value=c_value,
                    class_weight=class_weight,
                )
                result["feature_set"] = feature_set_name
                candidates.append(result)

    candidates.sort(key=lambda item: (item["f1"], item["roc_auc"]), reverse=True)
    best = candidates[0]

    serializable = []
    for item in candidates:
        serializable.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"oof_prob", "threshold_table"}
            }
            | {"threshold_table": item["threshold_table"]}
        )
    return best, serializable


def fit_final_model(best: dict, val_features: pd.DataFrame) -> Pipeline:
    x = val_features[best["features"]].to_numpy(dtype=float)
    y = val_features["y"].to_numpy(dtype=int)
    model = make_pipeline(c_value=best["c"], class_weight=best["class_weight"])
    model.fit(x, y)
    return model


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray, out_path: Path) -> None:
    if solution_df.columns.tolist() == ["label"]:
        submission = pd.DataFrame({"label": labels})
    else:
        submission = solution_df.copy()
        submission["label"] = labels
    submission = submission[solution_df.columns.tolist()]
    submission.to_csv(out_path, index=False)


def labels_from_top_k(score: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k > len(score):
        raise ValueError(f"Invalid top_k={top_k} for {len(score)} rows.")
    order = np.argsort(-score)
    pred = np.zeros(len(score), dtype=int)
    pred[order[:top_k]] = 1
    return decode_labels(pred)


def build_test_scores(test_features: pd.DataFrame, final_model: Pipeline, best: dict) -> pd.DataFrame:
    x_test = test_features[best["features"]].to_numpy(dtype=float)
    calibrated_prob = final_model.predict_proba(x_test)[:, 1]
    roberta_prob = test_features["roberta_prob_TRUE"].to_numpy(dtype=float)
    roberta_rank = percentile_rank(roberta_prob)
    calibrated_rank = percentile_rank(calibrated_prob)

    scores = pd.DataFrame(
        {
            "text": test_features["text"],
            "roberta_prob_TRUE": roberta_prob,
            "calibrated_prob_TRUE": calibrated_prob,
            "roberta_rank": roberta_rank,
            "calibrated_rank": calibrated_rank,
            "rank_blend_roberta_0p90": 0.90 * roberta_rank + 0.10 * calibrated_rank,
            "rank_blend_roberta_0p85": 0.85 * roberta_rank + 0.15 * calibrated_rank,
            "rank_blend_roberta_0p80": 0.80 * roberta_rank + 0.20 * calibrated_rank,
        }
    )
    return scores


def save_coefficients(final_model: Pipeline, best: dict) -> None:
    logreg = final_model.named_steps["logreg"]
    coefs = pd.DataFrame(
        {
            "feature": best["features"],
            "coefficient": logreg.coef_[0],
            "abs_coefficient": np.abs(logreg.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefs.to_csv(COEFFICIENTS_PATH, index=False)


def write_submission_variants(test_scores: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, score_col in SUBMISSION_VARIANTS:
        score = test_scores[score_col].to_numpy(dtype=float)
        for top_k in TARGET_TRUE_COUNTS:
            out_name = f"roberta_label_style_{variant_name}_top{top_k}_submission.csv"
            out_path = OUTPUT_DIR / out_name
            labels = labels_from_top_k(score, top_k=top_k)
            make_submission(solution_df, labels, out_path)
            rows.append(
                {
                    "filename": out_name,
                    "path": str(out_path),
                    "score_column": score_col,
                    "variant": variant_name,
                    "selection_method": "top_k",
                    "top_k_true": int(top_k),
                    "pred_TRUE": int((labels == "TRUE").sum()),
                    "pred_FALSE": int((labels == "FALSE").sum()),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading RoBERTa artifacts ===")
    val_df, test_df, solution_df = load_inputs()
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Solution columns: {solution_df.columns.tolist()}")

    print("\n=== Building label-style features ===")
    val_features = add_features(val_df, desc="Validation features")
    test_features = add_features(test_df, desc="Test features")

    print("\n=== Selecting lightweight calibrator on RoBERTa validation rows ===")
    best, all_candidates = select_calibrator(val_features)
    print(
        "Selected calibrator: "
        f"feature_set={best['feature_set']} C={best['c']} class_weight={best['class_weight']} "
        f"cv_f1={best['f1']:.6f} cv_auc={best['roc_auc']:.6f} threshold={best['selected_threshold']:.2f}"
    )

    print("\n=== Fitting final calibrator ===")
    final_model = fit_final_model(best, val_features)
    save_coefficients(final_model, best)
    print(f"Saved coefficients: {COEFFICIENTS_PATH}")

    val_score_df = val_features[["text", "true_label_norm", "roberta_prob_TRUE"]].copy()
    val_score_df["calibrator_oof_prob_TRUE"] = best["oof_prob"]
    val_score_df.to_csv(VAL_SCORE_PATH, index=False)
    print(f"Saved validation calibrator scores: {VAL_SCORE_PATH}")

    print("\n=== Scoring test rows ===")
    test_scores = build_test_scores(test_features, final_model, best)
    test_scores.to_csv(TEST_SCORE_PATH, index=False)
    print(f"Saved test scores: {TEST_SCORE_PATH}")

    print("\n=== Writing curated top-K submission variants ===")
    manifest = write_submission_variants(test_scores, solution_df)
    print(f"Saved submission manifest: {MANIFEST_PATH}")

    metrics_payload = {
        "model": "RoBERTa probability + deterministic label-style feature calibrator",
        "seed": SEED,
        "inputs": {
            "roberta_val_predictions": str(ROBERTA_VAL_PATH),
            "roberta_test_probabilities": str(ROBERTA_TEST_PROB_PATH),
            "solution_format": str(SOLUTION_FORMAT_PATH),
        },
        "selection": {
            "n_splits": N_SPLITS,
            "threshold_grid": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "target_true_counts": TARGET_TRUE_COUNTS,
            "best_feature_set": best["feature_set"],
            "best_features": best["features"],
            "best_c": float(best["c"]),
            "best_class_weight": best["class_weight"],
            "best_cv_f1": float(best["f1"]),
            "best_cv_auc": float(best["roc_auc"]),
            "best_threshold": float(best["selected_threshold"]),
        },
        "candidate_results": all_candidates,
        "submission_manifest": manifest.to_dict(orient="records"),
        "paths": {
            "metrics": str(METRICS_PATH),
            "val_scores": str(VAL_SCORE_PATH),
            "test_scores": str(TEST_SCORE_PATH),
            "coefficients": str(COEFFICIENTS_PATH),
            "manifest": str(MANIFEST_PATH),
            "notes": str(NOTES_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes = f"""# RoBERTa Label-Style Calibrator Notes

## Purpose

This is a lightweight correction layer around the existing strong RoBERTa single-split model. It does not train a transformer.

The calibrator uses RoBERTa validation probabilities plus deterministic text-style features from the EDA pass. The goal is to learn small ranking corrections for abnormal/generated/corrupted text patterns that RoBERTa sometimes misclassifies.

## Selected Calibrator

- Feature set: `{best['feature_set']}`
- C: `{best['c']}`
- Class weight: `{best['class_weight']}`
- Internal CV F1: `{best['f1']:.6f}`
- Internal CV ROC AUC: `{best['roc_auc']:.6f}`
- Internal selected threshold: `{best['selected_threshold']:.2f}`

## Submission Strategy

Submissions are written as top-K variants, not raw threshold variants. This avoids probability calibration mismatch and uses the leaderboard observation that good submissions cluster around 574-580 TRUE predictions.

Recommended first file to try:

`outputs/roberta_label_style_calibrator/roberta_label_style_rankblend0p85_top578_submission.csv`

## Outputs

- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_metrics.json`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_val_scores.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_test_scores.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_coefficients.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_submission_manifest.csv`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
