#!/usr/bin/env python3
"""
Train a hard-weighted RoBERTa CV ensemble on GPU.

This keeps the same recipe that produced the strong single-split GPU run:
- roberta-base
- max_length 384
- 6 epochs
- train batch size 16
- hard-example sample weights

The difference is that it trains across stratified folds and averages test
probabilities across fold checkpoints. That makes the submission less dependent
on one lucky validation split without changing the core model family.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train_roberta_hard_weighted_gpu import (
    INSTALL_CMD,
    abnormal_score,
    build_weight_frame,
    compute_feature_frame,
    compute_metrics,
    configure_cuda_runtime,
    decode_labels,
    detect_device,
    encode_labels,
    evaluate_model,
    fit_abnormal_thresholds,
    load_jsonl,
    normalize_texts,
    set_seeds,
    tokenize_texts,
    tune_threshold_for_f1,
    validate_inputs,
)


SEED = 42
MODEL_NAME = "/workspace/gpu/roberta-base"
MAX_LENGTH = 384
EPOCHS = 6
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
N_SPLITS = 5

DEFAULT_OUTPUT_NAME = "roberta_hard_weighted_cv_gpu"
DEFAULT_COUNT_CANDIDATES = [568, 572, 577, 582, 584, 586, 590, 595]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-weighted RoBERTa CV ensemble for GPU.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--count-candidates", type=int, nargs="+", default=DEFAULT_COUNT_CANDIDATES)
    parser.add_argument("--save-fold-models", action="store_true")
    return parser.parse_args()


def build_paths(output_name: str) -> dict[str, Path]:
    output_dir = ROOT_DIR / "outputs" / output_name
    return {
        "output_dir": output_dir,
        "fold_model_dir": output_dir / "fold_models",
        "metrics": output_dir / f"{output_name}_metrics.json",
        "fold_metrics": output_dir / f"{output_name}_fold_metrics.csv",
        "oof_predictions": output_dir / f"{output_name}_oof_predictions.csv",
        "error_analysis": output_dir / f"{output_name}_error_analysis.csv",
        "test_probabilities": output_dir / f"{output_name}_test_probabilities.csv",
        "submission": output_dir / f"{output_name}_submission.csv",
        "threshold_table": output_dir / f"{output_name}_threshold_table.csv",
        "candidate_manifest": output_dir / f"{output_name}_count_candidate_manifest.csv",
        "notes": output_dir / f"{output_name}_notes.md",
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_optimizer(
    model: AutoModelForSequenceClassification,
    learning_rate: float,
    adam_eps: float,
    weight_decay: float,
) -> AdamW:
    no_decay_terms = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        eps=adam_eps,
    )


def build_threshold_grid(args: argparse.Namespace) -> np.ndarray:
    stop = args.threshold_max + (args.threshold_step / 2.0)
    return np.round(np.arange(args.threshold_min, stop, args.threshold_step), 4)


def make_solution_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = labels
    return submission_df[solution_df.columns.tolist()]


def build_artifact_stratification(
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    n_splits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = fit_abnormal_thresholds(feature_df)
    scores = np.asarray([abnormal_score(row, thresholds) for _, row in feature_df.iterrows()], dtype=np.int64)
    buckets = np.where(scores <= 1, "low", np.where(scores == 2, "mid", "high"))
    keys = np.asarray([f"{label}_{bucket}" for label, bucket in zip(labels, buckets, strict=True)])

    counts = pd.Series(keys).value_counts().sort_index()
    if int(counts.min()) < n_splits:
        fallback_keys = labels.astype(str)
        fallback_counts = pd.Series(fallback_keys).value_counts().sort_index()
        return fallback_keys, {
            "mode": "label_only",
            "reason": "artifact bucket count below n_splits",
            "artifact_key_counts": counts.to_dict(),
            "fallback_key_counts": fallback_counts.to_dict(),
        }

    return keys, {
        "mode": "label_plus_artifact_bucket",
        "artifact_key_counts": counts.to_dict(),
        "feature_thresholds_for_stratification": thresholds,
    }


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )


def clone_state_dict_to_cpu(model: AutoModelForSequenceClassification) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_fold(
    *,
    fold_number: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    paths: dict[str, Path],
    device: torch.device,
    tokenizer: AutoTokenizer,
    local_files_only: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    test_loader: DataLoader,
    pin_memory: bool,
) -> dict[str, Any]:
    fold_seed = int(args.seed + fold_number)
    set_seeds(fold_seed)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    fold_train_features = feature_df.iloc[train_idx].reset_index(drop=True)
    fold_val_features = feature_df.iloc[val_idx].reset_index(drop=True)
    feature_thresholds = fit_abnormal_thresholds(fold_train_features)

    train_weight_df = build_weight_frame(y_train, fold_train_features, feature_thresholds)
    val_weight_df = build_weight_frame(y_val, fold_val_features, feature_thresholds)
    raw_train_weights = train_weight_df["raw_sample_weight"].to_numpy(dtype=np.float32)
    train_weights = raw_train_weights / max(float(raw_train_weights.mean()), 1e-12)

    train_dataset = TensorDataset(
        input_ids[train_idx],
        attention_mask[train_idx],
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(train_weights, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        input_ids[val_idx],
        attention_mask[val_idx],
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        seed=fold_seed,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=fold_seed,
        pin_memory=pin_memory,
    )

    print(f"\n===== Fold {fold_number}/{args.n_splits} =====")
    print(f"Fold seed: {fold_seed}")
    print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")
    print(f"Mean normalized train weight: {train_weights.mean():.6f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        local_files_only=local_files_only,
    )
    model.to(device)

    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_loss = math.nan
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_unweighted_loss = 0.0
        running_weight_mean = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Fold {fold_number} Epoch {epoch}/{args.epochs} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            batch_input_ids, batch_attention_mask, batch_labels, sample_weights = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            sample_weights = sample_weights.to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            per_row_loss = F.cross_entropy(outputs.logits, batch_labels, reduction="none")
            loss = (per_row_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                non_finite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(
                    train_loss="non_finite",
                    skipped_loss=non_finite_loss_batches,
                    skipped_grad=non_finite_grad_steps,
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                continue

            (loss / args.grad_accum).backward()

            batch_size = batch_input_ids.size(0)
            running_loss += loss_value * batch_size
            running_unweighted_loss += float(per_row_loss.mean().detach().cpu().item()) * batch_size
            running_weight_mean += float(sample_weights.mean().detach().cpu().item()) * batch_size
            seen_examples += batch_size

            if (step % args.grad_accum == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                grad_norm_value = (
                    float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                )
                if not math.isfinite(grad_norm_value):
                    non_finite_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(
                        train_loss=f"{(running_loss / max(seen_examples, 1)):.4f}",
                        skipped_loss=non_finite_loss_batches,
                        skipped_grad=non_finite_grad_steps,
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{(running_loss / max(seen_examples, 1)):.4f}",
                raw_ce=f"{(running_unweighted_loss / max(seen_examples, 1)):.4f}",
                w_mean=f"{(running_weight_mean / max(seen_examples, 1)):.3f}",
                skipped_loss=non_finite_loss_batches,
                skipped_grad=non_finite_grad_steps,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(seen_examples, 1)
        train_unweighted_loss = running_unweighted_loss / max(seen_examples, 1)

        val_loss, val_prob_true, val_true = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Fold {fold_number} Epoch {epoch}/{args.epochs} - validation",
            has_labels=True,
        )
        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["fold"] = int(fold_number)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_weighted_loss"] = float(train_loss)
        val_metrics_05["train_unweighted_loss"] = float(train_unweighted_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        epoch_history.append(val_metrics_05)

        print(
            f"Fold {fold_number} Epoch {epoch} | train_weighted_loss={train_loss:.6f} "
            f"| train_raw_ce={train_unweighted_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch = int(epoch)
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch_val_loss = float(val_metrics_05["val_loss"])
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            best_state_dict = clone_state_dict_to_cpu(model)
            print(f"Captured fold {fold_number} best state at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_state_dict is None or best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError(f"Fold {fold_number} did not produce a best checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    if args.save_fold_models:
        fold_model_dir = paths["fold_model_dir"] / f"fold_{fold_number}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fold_model_dir)
        tokenizer.save_pretrained(fold_model_dir)
        print(f"Saved fold model: {fold_model_dir}")

    _, test_prob_true, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        desc=f"Fold {fold_number} - test inference",
        has_labels=False,
    )

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold": int(fold_number),
        "val_idx": val_idx,
        "val_prob_true": best_epoch_val_probs,
        "val_true": best_epoch_val_labels,
        "val_abnormal_score": val_weight_df["abnormal_score"].to_numpy(dtype=np.int64),
        "val_hard_category": val_weight_df["hard_category"].to_numpy(),
        "val_raw_sample_weight": val_weight_df["raw_sample_weight"].to_numpy(dtype=np.float32),
        "test_prob_true": test_prob_true,
        "feature_thresholds": feature_thresholds,
        "best_epoch": int(best_epoch),
        "best_epoch_f1_at_0_5": float(best_epoch_f1),
        "best_epoch_val_loss": float(best_epoch_val_loss),
        "warmup_steps": int(warmup_steps),
        "epoch_history": epoch_history,
        "weight_summary": (
            train_weight_df.assign(
                label=decode_labels(y_train),
                normalized_sample_weight=train_weights,
            )
            .groupby(["label", "hard_category"])
            .agg(
                rows=("hard_category", "size"),
                mean_abnormal_score=("abnormal_score", "mean"),
                mean_raw_weight=("raw_sample_weight", "mean"),
                mean_normalized_weight=("normalized_sample_weight", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def write_count_candidates(
    *,
    output_dir: Path,
    output_name: str,
    solution_df: pd.DataFrame,
    probs: np.ndarray,
    counts: list[int],
) -> list[dict[str, Any]]:
    candidate_dir = output_dir / "count_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-probs)
    manifest = []
    for count in counts:
        if count < 0 or count > len(probs):
            raise ValueError(f"Invalid count {count}; expected 0..{len(probs)}.")
        pred = np.zeros(len(probs), dtype=np.int64)
        pred[order[:count]] = 1
        labels = decode_labels(pred)
        out_path = candidate_dir / f"{output_name}_top{count}_submission.csv"
        make_solution_submission(solution_df, labels).to_csv(out_path, index=False)
        threshold_floor = float(probs[order[count - 1]]) if count > 0 else 1.0
        manifest.append(
            {
                "count": int(count),
                "threshold_floor": threshold_floor,
                "submission": str(out_path),
            }
        )
        print(f"Saved count candidate: {out_path} TRUE_count={count}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa Hard-Weighted CV GPU Ensemble ===")

    paths = build_paths(args.output_name)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    set_seeds(args.seed)
    configure_cuda_runtime()
    device = detect_device()
    pin_memory = device.type == "cuda"
    local_files_only = Path(args.model_name).exists()
    threshold_grid = build_threshold_grid(args)

    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Local files only: {local_files_only}")
    print(f"Output dir: {paths['output_dir']}")
    print(f"Folds: {args.n_splits}")
    print(f"Save fold models: {args.save_fold_models}")

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_inputs(train_df, test_df, solution_df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    print("\n=== Preparing text, labels, and features ===")
    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    labels = encode_labels(train_df["label"], desc="Encoding labels")
    train_features = compute_feature_frame(texts, "Computing train hard features")
    test_features = compute_feature_frame(test_texts, "Computing test hard features")
    global_thresholds = fit_abnormal_thresholds(train_features)
    test_scores = np.asarray([abnormal_score(row, global_thresholds) for _, row in test_features.iterrows()])

    stratify_keys, stratification_meta = build_artifact_stratification(labels, train_features, args.n_splits)
    print(f"Stratification mode: {stratification_meta['mode']}")

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )

    print("\n=== Tokenizing once for all folds ===")
    input_ids, attention_mask = tokenize_texts(
        texts,
        tokenizer,
        max_length=args.max_length,
        batch_size=256,
        desc="Tokenizing train",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer,
        max_length=args.max_length,
        batch_size=256,
        desc="Tokenizing test",
    )
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(labels), dtype=np.float64)
    oof_fold = np.zeros(len(labels), dtype=np.int64)
    oof_abnormal_score = np.zeros(len(labels), dtype=np.int64)
    oof_hard_category = np.empty(len(labels), dtype=object)
    oof_raw_weight = np.zeros(len(labels), dtype=np.float32)
    fold_test_probs = []
    fold_summaries = []
    fold_epoch_rows = []
    fold_weight_rows = []

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), stratify_keys), start=1):
        fold_result = train_one_fold(
            fold_number=fold_number,
            train_idx=train_idx,
            val_idx=val_idx,
            args=args,
            paths=paths,
            device=device,
            tokenizer=tokenizer,
            local_files_only=local_files_only,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            feature_df=train_features,
            test_loader=test_loader,
            pin_memory=pin_memory,
        )

        oof_probs[val_idx] = fold_result["val_prob_true"]
        oof_fold[val_idx] = fold_number
        oof_abnormal_score[val_idx] = fold_result["val_abnormal_score"]
        oof_hard_category[val_idx] = fold_result["val_hard_category"]
        oof_raw_weight[val_idx] = fold_result["val_raw_sample_weight"]
        fold_test_probs.append(fold_result["test_prob_true"])

        fold_summaries.append(
            {
                "fold": fold_number,
                "best_epoch": fold_result["best_epoch"],
                "best_epoch_f1_at_0_5": fold_result["best_epoch_f1_at_0_5"],
                "best_epoch_val_loss": fold_result["best_epoch_val_loss"],
                "warmup_steps": fold_result["warmup_steps"],
            }
        )
        for row in fold_result["epoch_history"]:
            flat_row = dict(row)
            flat_row.pop("confusion_matrix", None)
            flat_row.pop("prediction_distribution", None)
            fold_epoch_rows.append(flat_row)
        for row in fold_result["weight_summary"]:
            fold_weight_rows.append({"fold": fold_number, **row})

    print("\n=== OOF threshold tuning ===")
    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=labels,
        prob_true=oof_probs,
        thresholds=threshold_grid,
    )
    final_oof_metrics = compute_metrics(labels, oof_probs, threshold=best_threshold)
    print(f"Selected OOF threshold: {best_threshold:.4f}")
    print(f"OOF F1: {final_oof_metrics['f1']:.6f}")

    threshold_df = pd.DataFrame(threshold_table)
    threshold_df.to_csv(paths["threshold_table"], index=False)
    print(f"Saved threshold table: {paths['threshold_table']}")

    oof_pred = (oof_probs >= best_threshold).astype(np.int64)
    oof_df = pd.DataFrame(
        {
            "row_index": np.arange(len(labels)),
            "fold": oof_fold,
            "text": texts,
            "true_label": decode_labels(labels),
            "pred_label": decode_labels(oof_pred),
            "pred_prob_TRUE": oof_probs,
            "abnormal_score": oof_abnormal_score,
            "hard_category": oof_hard_category,
            "raw_sample_weight_if_trained": oof_raw_weight,
        }
    )
    oof_df.to_csv(paths["oof_predictions"], index=False)
    print(f"Saved OOF predictions: {paths['oof_predictions']}")

    errors = oof_df[oof_df["true_label"] != oof_df["pred_label"]].copy()
    errors["error_type"] = np.where(
        (errors["true_label"] == "FALSE") & (errors["pred_label"] == "TRUE"),
        "false_positive",
        "false_negative",
    )
    errors["confidence"] = np.where(
        errors["error_type"] == "false_positive",
        errors["pred_prob_TRUE"],
        1.0 - errors["pred_prob_TRUE"],
    )
    fp_df = errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(300)
    fn_df = errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(300)
    pd.concat([fp_df, fn_df], ignore_index=True).to_csv(paths["error_analysis"], index=False)
    print(f"Saved error analysis: {paths['error_analysis']}")

    print("\n=== Averaging fold test probabilities ===")
    fold_test_prob_array = np.vstack(fold_test_probs)
    mean_test_probs = fold_test_prob_array.mean(axis=0)
    std_test_probs = fold_test_prob_array.std(axis=0)
    test_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "pred_prob_TRUE": mean_test_probs,
            "pred_prob_TRUE_std": std_test_probs,
            "abnormal_score_global": test_scores,
        }
    )
    for fold_number in range(1, args.n_splits + 1):
        test_prob_df[f"fold_{fold_number}_prob_TRUE"] = fold_test_prob_array[fold_number - 1]
    test_prob_df.to_csv(paths["test_probabilities"], index=False)
    print(f"Saved test probabilities: {paths['test_probabilities']}")

    test_pred = (mean_test_probs >= best_threshold).astype(np.int64)
    test_labels = decode_labels(test_pred)
    submission_df = make_solution_submission(solution_df, test_labels)
    submission_df.to_csv(paths["submission"], index=False)
    test_distribution = {
        "pred_FALSE": int((test_pred == 0).sum()),
        "pred_TRUE": int((test_pred == 1).sum()),
    }
    print(f"Saved threshold submission: {paths['submission']}")
    print(f"Threshold submission distribution: {test_distribution}")

    candidate_manifest = write_count_candidates(
        output_dir=paths["output_dir"],
        output_name=args.output_name,
        solution_df=solution_df,
        probs=mean_test_probs,
        counts=args.count_candidates,
    )
    pd.DataFrame(candidate_manifest).to_csv(paths["candidate_manifest"], index=False)
    print(f"Saved count candidate manifest: {paths['candidate_manifest']}")

    fold_metrics_df = pd.DataFrame(fold_summaries)
    fold_metrics_df.to_csv(paths["fold_metrics"], index=False)
    print(f"Saved fold metrics: {paths['fold_metrics']}")

    metrics_payload = {
        "model": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "config": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_train_batch_size": args.train_batch_size * args.grad_accum,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "max_grad_norm": args.max_grad_norm,
            "n_splits": args.n_splits,
            "loss": "sample_weighted_cross_entropy",
            "fold_checkpoint_selection": "best_validation_f1_at_threshold_0_5",
            "final_threshold_selection": "best_oof_f1",
            "save_fold_models": args.save_fold_models,
        },
        "stratification": stratification_meta,
        "fold_summaries": fold_summaries,
        "fold_epoch_metrics": fold_epoch_rows,
        "fold_weight_summaries": fold_weight_rows,
        "threshold_tuning": {
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "oof_final": final_oof_metrics,
        "test_prediction_distribution": test_distribution,
        "count_candidates": candidate_manifest,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["metrics"].write_text(json.dumps(metrics_payload, indent=2, default=json_default), encoding="utf-8")
    print(f"Saved metrics: {paths['metrics']}")

    notes_text = f"""# RoBERTa Hard-Weighted CV GPU Notes

## Goal

This run keeps the successful hard-weighted RoBERTa recipe, but replaces the single
train/validation split with `{args.n_splits}` stratified folds and averages test probabilities.

## Core Configuration

- Model: `{args.model_name}`
- Max length: `{args.max_length}`
- Epochs per fold: `{args.epochs}`
- Train batch size: `{args.train_batch_size}`
- Eval batch size: `{args.eval_batch_size}`
- Gradient accumulation: `{args.grad_accum}`
- Effective train batch size: `{args.train_batch_size * args.grad_accum}`
- Learning rate: `{args.learning_rate}`
- Fold models saved: `{args.save_fold_models}`

## Selection

- Per-fold checkpoint: best validation F1 at threshold `0.5`
- Final threshold: best full OOF F1 over `{args.threshold_min}` to `{args.threshold_max}`
- Selected threshold: `{best_threshold:.4f}`
- OOF F1: `{final_oof_metrics['f1']:.6f}`
- OOF ROC AUC: `{final_oof_metrics['roc_auc']:.6f}`
- Threshold submission TRUE count: `{test_distribution['pred_TRUE']}`

## Main Outputs

- `{paths['submission']}`
- `{paths['test_probabilities']}`
- `{paths['oof_predictions']}`
- `{paths['metrics']}`
- `{paths['candidate_manifest']}`

The `count_candidates/` directory contains top-N TRUE submissions around the known useful
TRUE-count region from the single-split run. Treat those as secondary candidates; the main
non-tuned output is the OOF-threshold submission.
"""
    paths["notes"].write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {paths['notes']}")
    print("\nRun complete.")


if __name__ == "__main__":
    main()
