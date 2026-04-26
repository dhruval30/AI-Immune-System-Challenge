#!/usr/bin/env python3
"""
Inspect the dataset and existing RoBERTa artifacts to recommend hyperparameter changes.

Dependency install command:
pip install pandas numpy scikit-learn tqdm transformers

This is not a training script. It does not fine-tune a model and it does not create
a submission. It measures token-length/truncation behavior and existing RoBERTa
validation errors so the next edit to models/train_roberta_base.py is evidence-based.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm transformers"

SEED = 42
MODEL_NAME = "roberta-base"
CURRENT_MAX_LENGTH = 256
CANDIDATE_MAX_LENGTHS = [128, 192, 256, 320, 384, 448, 512]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_hyperparam_inspection"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

LOCAL_TOKENIZER_DIR = ROOT_DIR / "outputs" / "roberta_base" / "best_model"
ROBERTA_METRICS_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_metrics.json"
ROBERTA_VAL_PRED_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv"

TRAIN_LENGTHS_PATH = OUTPUT_DIR / "train_token_lengths.csv"
TEST_LENGTHS_PATH = OUTPUT_DIR / "test_token_lengths.csv"
TOKEN_STATS_PATH = OUTPUT_DIR / "token_length_stats_by_label.csv"
COVERAGE_PATH = OUTPUT_DIR / "max_length_coverage.csv"
ERROR_LENGTH_PATH = OUTPUT_DIR / "roberta_error_length_analysis.csv"
RECOMMENDATION_JSON_PATH = OUTPUT_DIR / "roberta_hyperparam_recommendations.json"
RECOMMENDATION_MD_PATH = OUTPUT_DIR / "roberta_hyperparam_recommendations.md"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total = count_lines(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(f"Train file must contain text and label columns; found {list(train_df.columns)}")
    if "text" not in test_df.columns:
        raise ValueError(f"Test file must contain text column; found {list(test_df.columns)}")
    if len(test_df) != len(solution_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )
    if "label" not in solution_df.columns:
        raise ValueError(f"solution_format.csv must contain label column; found {list(solution_df.columns)}")


def load_tokenizer() -> AutoTokenizer:
    if LOCAL_TOKENIZER_DIR.exists():
        print(f"Loading tokenizer from local RoBERTa checkpoint: {LOCAL_TOKENIZER_DIR}")
        try:
            return AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_DIR, use_fast=True)
        except Exception as exc:
            print(f"Could not load local tokenizer with use_fast=True: {exc}")
            try:
                return AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_DIR, use_fast=False)
            except Exception as slow_exc:
                print(f"Could not load local tokenizer with use_fast=False: {slow_exc}")
                print(f"Falling back to tokenizer from model name: {MODEL_NAME}")
    print(f"Local tokenizer not found. Loading tokenizer from Hugging Face model name: {MODEL_NAME}")
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as exc:
        print(f"Could not load {MODEL_NAME} with use_fast=True: {exc}")
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


def token_lengths(texts: list[str], tokenizer: AutoTokenizer, desc: str) -> np.ndarray:
    lengths: list[int] = []
    batch_size = 256
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


def basic_text_stats(texts: list[str]) -> pd.DataFrame:
    word_re = re.compile(r"\S+")
    rows = []
    for text in tqdm(texts, desc="Computing basic text stats", unit="rows"):
        words = word_re.findall(text)
        rows.append(
            {
                "char_length": len(text),
                "word_count": len(words),
                "newline_count": text.count("\n"),
            }
        )
    return pd.DataFrame(rows)


def describe_values(values: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    if len(values) == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": 0.0,
            f"{prefix}std": 0.0,
            f"{prefix}min": 0.0,
            f"{prefix}p50": 0.0,
            f"{prefix}p75": 0.0,
            f"{prefix}p90": 0.0,
            f"{prefix}p95": 0.0,
            f"{prefix}p99": 0.0,
            f"{prefix}max": 0.0,
        }
    return {
        f"{prefix}count": int(len(values)),
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}std": float(np.std(values)),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}p50": float(np.percentile(values, 50)),
        f"{prefix}p75": float(np.percentile(values, 75)),
        f"{prefix}p90": float(np.percentile(values, 90)),
        f"{prefix}p95": float(np.percentile(values, 95)),
        f"{prefix}p99": float(np.percentile(values, 99)),
        f"{prefix}max": float(np.max(values)),
    }


def build_length_stats(train_lengths_df: pd.DataFrame, test_lengths_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in ["FALSE", "TRUE"]:
        subset = train_lengths_df[train_lengths_df["label"] == label]
        row = {"dataset": "train", "label": label}
        row.update(describe_values(subset["token_length"].to_numpy(), prefix="token_"))
        row.update(describe_values(subset["char_length"].to_numpy(), prefix="char_"))
        row.update(describe_values(subset["word_count"].to_numpy(), prefix="word_"))
        rows.append(row)

    train_row = {"dataset": "train", "label": "ALL"}
    train_row.update(describe_values(train_lengths_df["token_length"].to_numpy(), prefix="token_"))
    train_row.update(describe_values(train_lengths_df["char_length"].to_numpy(), prefix="char_"))
    train_row.update(describe_values(train_lengths_df["word_count"].to_numpy(), prefix="word_"))
    rows.append(train_row)

    test_row = {"dataset": "test", "label": "UNKNOWN"}
    test_row.update(describe_values(test_lengths_df["token_length"].to_numpy(), prefix="token_"))
    test_row.update(describe_values(test_lengths_df["char_length"].to_numpy(), prefix="char_"))
    test_row.update(describe_values(test_lengths_df["word_count"].to_numpy(), prefix="word_"))
    rows.append(test_row)

    return pd.DataFrame(rows)


def coverage_for_lengths(
    token_lengths_array: np.ndarray,
    dataset: str,
    label: str,
    special_token_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    safe_denominator = np.maximum(token_lengths_array, 1)
    for max_length in CANDIDATE_MAX_LENGTHS:
        content_capacity = max_length - special_token_count
        truncated = token_lengths_array > content_capacity
        retained_ratio = np.minimum(token_lengths_array, content_capacity) / safe_denominator
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "max_length": int(max_length),
                "content_capacity_without_special_tokens": int(content_capacity),
                "truncated_count": int(truncated.sum()),
                "total_count": int(len(token_lengths_array)),
                "truncated_pct": float(100.0 * truncated.mean()) if len(token_lengths_array) else 0.0,
                "mean_retained_token_pct": float(100.0 * np.mean(retained_ratio)) if len(token_lengths_array) else 100.0,
                "p05_retained_token_pct": float(100.0 * np.percentile(retained_ratio, 5))
                if len(token_lengths_array)
                else 100.0,
            }
        )
    return rows


def build_coverage_table(
    train_lengths_df: pd.DataFrame,
    test_lengths_df: pd.DataFrame,
    special_token_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(
        coverage_for_lengths(
            train_lengths_df["token_length"].to_numpy(),
            dataset="train",
            label="ALL",
            special_token_count=special_token_count,
        )
    )
    for label in ["FALSE", "TRUE"]:
        subset = train_lengths_df[train_lengths_df["label"] == label]
        rows.extend(
            coverage_for_lengths(
                subset["token_length"].to_numpy(),
                dataset="train",
                label=label,
                special_token_count=special_token_count,
            )
        )
    rows.extend(
        coverage_for_lengths(
            test_lengths_df["token_length"].to_numpy(),
            dataset="test",
            label="UNKNOWN",
            special_token_count=special_token_count,
        )
    )
    return pd.DataFrame(rows)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_existing_roberta_summary() -> dict[str, Any]:
    metrics = read_json_if_exists(ROBERTA_METRICS_PATH)
    if not metrics:
        return {
            "available": False,
            "reason": f"Metrics file not found at {ROBERTA_METRICS_PATH}",
        }

    epoch_history = metrics.get("epoch_metrics_threshold_0_5", [])
    best_epoch_payload = metrics.get("best_epoch", {})
    best_epoch = best_epoch_payload.get("epoch") if isinstance(best_epoch_payload, dict) else None
    selected_threshold = metrics.get("threshold_tuning", {}).get("selected_threshold")
    final_metrics = metrics.get("validation_final", {})
    test_distribution = metrics.get("test_prediction_distribution", {})

    return {
        "available": True,
        "metrics_path": str(ROBERTA_METRICS_PATH),
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "validation_final_f1": final_metrics.get("f1"),
        "validation_final_auc": final_metrics.get("roc_auc"),
        "test_prediction_distribution": test_distribution,
        "epoch_history": epoch_history,
    }


def analyze_roberta_errors(
    tokenizer: AutoTokenizer,
    special_token_count: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not ROBERTA_VAL_PRED_PATH.exists():
        return None, {
            "available": False,
            "reason": f"Validation prediction file not found at {ROBERTA_VAL_PRED_PATH}",
        }

    val_df = pd.read_csv(ROBERTA_VAL_PRED_PATH)
    required_columns = {"text", "true_label", "pred_label", "pred_prob_TRUE"}
    missing_columns = sorted(required_columns - set(val_df.columns))
    if missing_columns:
        return None, {
            "available": False,
            "reason": f"Validation prediction file is missing columns: {missing_columns}",
        }

    val_texts = [normalize_text(value) for value in val_df["text"].tolist()]
    val_token_lengths = token_lengths(val_texts, tokenizer, "Tokenizing RoBERTa validation rows")
    val_df = val_df.copy()
    # pandas reads TRUE/FALSE columns as bool by default, so normalize before comparisons.
    val_df["true_label_norm"] = [normalize_label(value) for value in val_df["true_label"].tolist()]
    val_df["pred_label_norm"] = [normalize_label(value) for value in val_df["pred_label"].tolist()]
    val_df["token_length"] = val_token_lengths
    val_df["truncated_at_256"] = val_df["token_length"] > (CURRENT_MAX_LENGTH - special_token_count)
    val_df["truncated_at_384"] = val_df["token_length"] > (384 - special_token_count)
    val_df["outcome"] = "correct"
    val_df.loc[
        (val_df["true_label_norm"] == "TRUE") & (val_df["pred_label_norm"] == "TRUE"),
        "outcome",
    ] = "true_positive"
    val_df.loc[
        (val_df["true_label_norm"] == "FALSE") & (val_df["pred_label_norm"] == "FALSE"),
        "outcome",
    ] = "true_negative"
    val_df.loc[
        (val_df["true_label_norm"] == "FALSE") & (val_df["pred_label_norm"] == "TRUE"),
        "outcome",
    ] = "false_positive"
    val_df.loc[
        (val_df["true_label_norm"] == "TRUE") & (val_df["pred_label_norm"] == "FALSE"),
        "outcome",
    ] = "false_negative"

    rows: list[dict[str, Any]] = []
    for outcome, subset in val_df.groupby("outcome"):
        row = {"outcome": outcome}
        row.update(describe_values(subset["token_length"].to_numpy(), prefix="token_"))
        row["truncated_at_256_count"] = int(subset["truncated_at_256"].sum())
        row["truncated_at_256_pct"] = float(100.0 * subset["truncated_at_256"].mean()) if len(subset) else 0.0
        row["truncated_at_384_count"] = int(subset["truncated_at_384"].sum())
        row["truncated_at_384_pct"] = float(100.0 * subset["truncated_at_384"].mean()) if len(subset) else 0.0
        row["mean_pred_prob_TRUE"] = float(subset["pred_prob_TRUE"].mean()) if len(subset) else 0.0
        rows.append(row)

    error_stats_df = pd.DataFrame(rows).sort_values("outcome")
    error_stats_df.to_csv(ERROR_LENGTH_PATH, index=False)

    false_negatives = val_df[val_df["outcome"] == "false_negative"]
    true_positives = val_df[val_df["outcome"] == "true_positive"]
    summary = {
        "available": True,
        "validation_rows": int(len(val_df)),
        "error_length_analysis_path": str(ERROR_LENGTH_PATH),
        "false_negative_count": int(len(false_negatives)),
        "false_positive_count": int((val_df["outcome"] == "false_positive").sum()),
        "false_negative_truncated_at_256_pct": float(100.0 * false_negatives["truncated_at_256"].mean())
        if len(false_negatives)
        else 0.0,
        "true_positive_truncated_at_256_pct": float(100.0 * true_positives["truncated_at_256"].mean())
        if len(true_positives)
        else 0.0,
        "false_negative_token_p95": float(np.percentile(false_negatives["token_length"], 95))
        if len(false_negatives)
        else 0.0,
        "true_positive_token_p95": float(np.percentile(true_positives["token_length"], 95))
        if len(true_positives)
        else 0.0,
    }
    return error_stats_df, summary


def coverage_lookup(
    coverage_df: pd.DataFrame,
    dataset: str,
    label: str,
    max_length: int,
    column: str = "truncated_pct",
) -> float:
    subset = coverage_df[
        (coverage_df["dataset"] == dataset)
        & (coverage_df["label"] == label)
        & (coverage_df["max_length"] == max_length)
    ]
    if subset.empty:
        return 0.0
    return float(subset.iloc[0][column])


def choose_max_length(
    coverage_df: pd.DataFrame,
    error_summary: dict[str, Any],
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    train_256 = coverage_lookup(coverage_df, "train", "ALL", 256)
    train_true_256 = coverage_lookup(coverage_df, "train", "TRUE", 256)
    test_256 = coverage_lookup(coverage_df, "test", "UNKNOWN", 256)
    train_384 = coverage_lookup(coverage_df, "train", "ALL", 384)
    train_true_384 = coverage_lookup(coverage_df, "train", "TRUE", 384)
    test_384 = coverage_lookup(coverage_df, "test", "UNKNOWN", 384)

    fn_trunc_256 = float(error_summary.get("false_negative_truncated_at_256_pct", 0.0))
    tp_trunc_256 = float(error_summary.get("true_positive_truncated_at_256_pct", 0.0))
    fn_p95 = float(error_summary.get("false_negative_token_p95", 0.0))

    any_256_truncation = max(train_256, train_true_256, test_256) > 0.0
    meaningful_256_truncation = max(train_256, train_true_256, test_256) >= 2.0
    fn_length_signal = fn_trunc_256 >= max(5.0, tp_trunc_256 + 3.0) or fn_p95 > 256

    if any_256_truncation and not meaningful_256_truncation and not fn_length_signal:
        reasons.append(
            "Use max_length=384 for the next controlled larger-context experiment. "
            "Pure coverage says 256 is efficient, but 256 does truncate some train/test rows, "
            "and repeating the current 256 setup is not an informative next run."
        )
        reasons.append(
            "Do not jump straight to 512 first: 384 removes nearly all truncation with lower cost and memory risk."
        )
        return 384, reasons

    if not any_256_truncation and not fn_length_signal:
        reasons.append(
            "Keep max_length=256 unless the coverage table shows unexpectedly high truncation; "
            "current evidence does not make longer context the main bottleneck."
        )
        return 256, reasons

    reduction_at_384 = max(train_256 - train_384, train_true_256 - train_true_384, test_256 - test_384)
    if reduction_at_384 >= 1.0 and max(train_384, train_true_384, test_384) <= 3.0:
        reasons.append(
            "Use max_length=384: it materially reduces truncation from 256 while avoiding the full 512-token cost."
        )
        return 384, reasons

    if max(train_384, train_true_384, test_384) > 3.0:
        reasons.append(
            "Use max_length=512: even 384 still leaves meaningful truncation. Expect slower training."
        )
        return 512, reasons

    reasons.append(
        "Use max_length=384 as the conservative larger-context test; 512 is not justified unless 384 still truncates many rows."
    )
    return 384, reasons


def recommend_hyperparams(
    recommended_max_length: int,
    roberta_summary: dict[str, Any],
    coverage_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []

    if recommended_max_length <= 256:
        train_batch_size = 8
        grad_accum = 2
        eval_batch_size = 16
    elif recommended_max_length <= 384:
        train_batch_size = 4
        grad_accum = 4
        eval_batch_size = 8
        reasons.append(
            "For max_length=384, use batch_size=4 and grad_accum=4 to preserve effective batch size 16 on MPS."
        )
    else:
        train_batch_size = 4
        grad_accum = 4
        eval_batch_size = 8
        reasons.append(
            "For max_length=512, batch_size=4 is the safer MPS choice; use gradient checkpointing only if memory fails."
        )

    best_epoch = roberta_summary.get("best_epoch") if roberta_summary.get("available") else None
    epoch_history = roberta_summary.get("epoch_history", []) if roberta_summary.get("available") else []
    if best_epoch == 1:
        epochs = 2
        reasons.append(
            "Existing RoBERTa selected epoch 1 as best, so a larger-context run should avoid long overfit cycles; use 2 epochs with best-checkpoint saving."
        )
    else:
        epochs = 3
        reasons.append("Existing metrics do not clearly force shorter training; keep 3 epochs with best-checkpoint saving.")

    if len(epoch_history) >= 2:
        first = epoch_history[0]
        last = epoch_history[-1]
        first_f1 = first.get("f1")
        last_f1 = last.get("f1")
        first_loss = first.get("val_loss")
        last_loss = last.get("val_loss")
        if first_f1 is not None and last_f1 is not None and last_f1 < first_f1:
            reasons.append("Validation F1 fell after epoch 1 in the anchor run; do not increase epochs first.")
        if first_loss is not None and last_loss is not None and last_loss > first_loss:
            reasons.append("Validation loss increased after epoch 1 in the anchor run; longer training is likely overfitting.")

    config = {
        "MAX_LENGTH": int(recommended_max_length),
        "EPOCHS": int(epochs),
        "TRAIN_BATCH_SIZE": int(train_batch_size),
        "EVAL_BATCH_SIZE": int(eval_batch_size),
        "GRADIENT_ACCUMULATION_STEPS": int(grad_accum),
        "LEARNING_RATE": 1e-5,
        "WEIGHT_DECAY": 0.01,
        "ADAM_EPS": 1e-8,
        "effective_train_batch_size": int(train_batch_size * grad_accum),
        "optional_if_memory_fails": {
            "enable_gradient_checkpointing": recommended_max_length >= 384,
            "reduce_train_batch_size_to": 2 if recommended_max_length >= 384 else None,
            "increase_gradient_accumulation_to": 8 if recommended_max_length >= 384 else None,
        },
        "efficient_baseline_reference": {
            "MAX_LENGTH": 256,
            "reason": (
                "256 remains the efficient baseline because only a small percentage of rows are truncated. "
                "The recommended larger value is for the next experiment, not because 256 is objectively broken."
            ),
            "train_truncated_pct_at_256": coverage_lookup(coverage_df, "train", "ALL", 256),
            "test_truncated_pct_at_256": coverage_lookup(coverage_df, "test", "UNKNOWN", 256),
        },
    }
    return config, reasons


def write_markdown_report(
    recommendation: dict[str, Any],
    token_stats_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    roberta_summary: dict[str, Any],
    error_summary: dict[str, Any],
    max_length_reasons: list[str],
    hyperparam_reasons: list[str],
) -> None:
    recommended = recommendation["recommended_config"]
    current_coverage = coverage_df[coverage_df["max_length"] == CURRENT_MAX_LENGTH].copy()
    recommended_coverage = coverage_df[coverage_df["max_length"] == recommended["MAX_LENGTH"]].copy()

    current_table = current_coverage[
        ["dataset", "label", "max_length", "truncated_count", "total_count", "truncated_pct"]
    ].to_markdown(index=False)
    recommended_table = recommended_coverage[
        ["dataset", "label", "max_length", "truncated_count", "total_count", "truncated_pct"]
    ].to_markdown(index=False)
    stats_table = token_stats_df[
        ["dataset", "label", "token_count", "token_mean", "token_p90", "token_p95", "token_p99", "token_max"]
    ].to_markdown(index=False)

    patch_lines = "\n".join(
        [
            f"MAX_LENGTH = {recommended['MAX_LENGTH']}",
            f"EPOCHS = {recommended['EPOCHS']}",
            f"TRAIN_BATCH_SIZE = {recommended['TRAIN_BATCH_SIZE']}",
            f"EVAL_BATCH_SIZE = {recommended['EVAL_BATCH_SIZE']}",
            f"GRADIENT_ACCUMULATION_STEPS = {recommended['GRADIENT_ACCUMULATION_STEPS']}",
            f"LEARNING_RATE = {recommended['LEARNING_RATE']}",
            f"WEIGHT_DECAY = {recommended['WEIGHT_DECAY']}",
            f"ADAM_EPS = {recommended['ADAM_EPS']}",
        ]
    )

    max_length_reason_text = "\n".join(f"- {reason}" for reason in max_length_reasons)
    hyperparam_reason_text = "\n".join(f"- {reason}" for reason in hyperparam_reasons)

    roberta_section = "Existing RoBERTa metrics were not available."
    if roberta_summary.get("available"):
        roberta_section = f"""- Best epoch: `{roberta_summary.get("best_epoch")}`
- Selected threshold: `{roberta_summary.get("selected_threshold")}`
- Validation F1: `{roberta_summary.get("validation_final_f1")}`
- Validation ROC AUC: `{roberta_summary.get("validation_final_auc")}`
- Test distribution: `{roberta_summary.get("test_prediction_distribution")}`"""

    error_section = "RoBERTa validation prediction analysis was not available."
    if error_summary.get("available"):
        error_section = f"""- False negatives: `{error_summary.get("false_negative_count")}`
- False positives: `{error_summary.get("false_positive_count")}`
- False-negative truncation at 256: `{error_summary.get("false_negative_truncated_at_256_pct"):.2f}%`
- True-positive truncation at 256: `{error_summary.get("true_positive_truncated_at_256_pct"):.2f}%`
- False-negative token p95: `{error_summary.get("false_negative_token_p95"):.2f}`
- True-positive token p95: `{error_summary.get("true_positive_token_p95"):.2f}`"""

    text = f"""# RoBERTa Hyperparameter Inspection

This report inspects token lengths, truncation risk, and existing RoBERTa validation errors. It does not train a model.

## Recommended Edit For `models/train_roberta_base.py`

```python
{patch_lines}
```

Effective train batch size: `{recommended["effective_train_batch_size"]}`.

## Why

{max_length_reason_text}
{hyperparam_reason_text}

## Existing RoBERTa Anchor

{roberta_section}

## RoBERTa Error Length Signal

{error_section}

## Token Length Stats

{stats_table}

## Current 256 Coverage

{current_table}

## Recommended Max Length Coverage

{recommended_table}

## Output Files

- `{TRAIN_LENGTHS_PATH.relative_to(ROOT_DIR)}`
- `{TEST_LENGTHS_PATH.relative_to(ROOT_DIR)}`
- `{TOKEN_STATS_PATH.relative_to(ROOT_DIR)}`
- `{COVERAGE_PATH.relative_to(ROOT_DIR)}`
- `{ERROR_LENGTH_PATH.relative_to(ROOT_DIR)}`
- `{RECOMMENDATION_JSON_PATH.relative_to(ROOT_DIR)}`
- `{RECOMMENDATION_MD_PATH.relative_to(ROOT_DIR)}`

## Practical Next Step

If the recommendation is `MAX_LENGTH=384` or `512`, create a separate RoBERTa training script or output directory for that run instead of overwriting the current best `outputs/roberta_base/` artifacts.
"""
    RECOMMENDATION_MD_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Hyperparameter Inspection ===")
    print("This script does not train and does not create a submission.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_inputs(train_df, test_df, solution_df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    train_texts = [normalize_text(value) for value in tqdm(train_df["text"].tolist(), desc="Normalizing train text")]
    test_texts = [normalize_text(value) for value in tqdm(test_df["text"].tolist(), desc="Normalizing test text")]
    labels = [normalize_label(value) for value in tqdm(train_df["label"].tolist(), desc="Normalizing train labels")]

    print("\n=== Loading tokenizer ===")
    tokenizer = load_tokenizer()
    special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    print(f"Special tokens added per single sequence: {special_token_count}")

    print("\n=== Measuring token lengths ===")
    train_token_lengths = token_lengths(train_texts, tokenizer, "Tokenizing train without truncation")
    test_token_lengths = token_lengths(test_texts, tokenizer, "Tokenizing test without truncation")
    train_basic_stats = basic_text_stats(train_texts)
    test_basic_stats = basic_text_stats(test_texts)

    train_lengths_df = pd.DataFrame(
        {
            "row_id": np.arange(len(train_texts)),
            "label": labels,
            "token_length": train_token_lengths,
        }
    )
    train_lengths_df = pd.concat([train_lengths_df, train_basic_stats], axis=1)

    test_lengths_df = pd.DataFrame(
        {
            "row_id": np.arange(len(test_texts)),
            "token_length": test_token_lengths,
        }
    )
    test_lengths_df = pd.concat([test_lengths_df, test_basic_stats], axis=1)

    for max_length in CANDIDATE_MAX_LENGTHS:
        capacity = max_length - special_token_count
        train_lengths_df[f"truncated_at_{max_length}"] = train_lengths_df["token_length"] > capacity
        test_lengths_df[f"truncated_at_{max_length}"] = test_lengths_df["token_length"] > capacity

    train_lengths_df.to_csv(TRAIN_LENGTHS_PATH, index=False)
    test_lengths_df.to_csv(TEST_LENGTHS_PATH, index=False)
    print(f"Saved train token lengths: {TRAIN_LENGTHS_PATH}")
    print(f"Saved test token lengths: {TEST_LENGTHS_PATH}")

    token_stats_df = build_length_stats(train_lengths_df, test_lengths_df)
    coverage_df = build_coverage_table(train_lengths_df, test_lengths_df, special_token_count)
    token_stats_df.to_csv(TOKEN_STATS_PATH, index=False)
    coverage_df.to_csv(COVERAGE_PATH, index=False)
    print(f"Saved token stats: {TOKEN_STATS_PATH}")
    print(f"Saved max-length coverage: {COVERAGE_PATH}")

    print("\n=== Reading existing RoBERTa artifacts ===")
    roberta_summary = get_existing_roberta_summary()
    _, error_summary = analyze_roberta_errors(tokenizer, special_token_count)
    if roberta_summary.get("available"):
        print(f"Existing RoBERTa best epoch: {roberta_summary.get('best_epoch')}")
        print(f"Existing RoBERTa selected threshold: {roberta_summary.get('selected_threshold')}")
    else:
        print(roberta_summary.get("reason"))
    if error_summary.get("available"):
        print(f"Saved RoBERTa error length analysis: {ERROR_LENGTH_PATH}")
    else:
        print(error_summary.get("reason"))

    print("\n=== Building recommendation ===")
    recommended_max_length, max_length_reasons = choose_max_length(coverage_df, error_summary)
    recommended_config, hyperparam_reasons = recommend_hyperparams(
        recommended_max_length,
        roberta_summary,
        coverage_df,
    )

    recommendation = {
        "script_type": "inspection_only_no_training",
        "model_anchor": MODEL_NAME,
        "current_config": {
            "MAX_LENGTH": CURRENT_MAX_LENGTH,
            "known_best_lb_anchor": "outputs/roberta_base/roberta_base_submission.csv -> 0.90909091",
        },
        "recommended_config": recommended_config,
        "max_length_reasons": max_length_reasons,
        "hyperparam_reasons": hyperparam_reasons,
        "roberta_summary": roberta_summary,
        "error_summary": error_summary,
        "paths": {
            "train_token_lengths": str(TRAIN_LENGTHS_PATH),
            "test_token_lengths": str(TEST_LENGTHS_PATH),
            "token_stats": str(TOKEN_STATS_PATH),
            "coverage": str(COVERAGE_PATH),
            "error_length_analysis": str(ERROR_LENGTH_PATH),
            "markdown_report": str(RECOMMENDATION_MD_PATH),
        },
    }
    RECOMMENDATION_JSON_PATH.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    write_markdown_report(
        recommendation=recommendation,
        token_stats_df=token_stats_df,
        coverage_df=coverage_df,
        roberta_summary=roberta_summary,
        error_summary=error_summary,
        max_length_reasons=max_length_reasons,
        hyperparam_reasons=hyperparam_reasons,
    )

    print(f"Saved recommendation JSON: {RECOMMENDATION_JSON_PATH}")
    print(f"Saved recommendation report: {RECOMMENDATION_MD_PATH}")
    print("\nRecommended config:")
    for key, value in recommended_config.items():
        if key in {"optional_if_memory_fails", "efficient_baseline_reference"}:
            continue
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
