#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from roberta_dapt_pipeline_common import (
    assign_artifact_buckets,
    build_text_feature_df,
    compute_metrics,
    decode_labels,
    detect_device,
    encode_labels,
    fit_artifact_thresholds,
    load_jsonl,
    make_artifact_stratify_labels,
    make_occurrence_keys,
    make_submission,
    normalize_label,
    normalize_texts,
    set_seeds,
    tokenize_texts,
    tune_threshold_for_f1,
    validate_competition_inputs,
)

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

MODEL_NAME = "roberta-base"
MAX_LENGTH = 384
N_SPLITS = 5
SPLIT_SEEDS = [42]
EPOCHS = 4
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 8e-6
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
CLASS_WEIGHT_FALSE = 1.0
CLASS_WEIGHT_TRUE = 1.5
LABEL_SMOOTHING = 0.03
DISTILL_ALPHA = 0.35
EARLY_STOPPING_PATIENCE = 2
WARMUP_RATIO = 0.10
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01
TOKENIZE_BATCH_SIZE = 256
PSEUDO_WEIGHT = 0.40

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_dapt_cv_distill"
CHECKPOINTS_DIR = OUTPUT_DIR / "fold_checkpoints"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

DAPT_MODEL_DIR = ROOT_DIR / "outputs" / "roberta_dapt_mlm" / "best_model"
TEACHER_TARGETS_PATH = ROOT_DIR / "outputs" / "roberta_teacher_targets" / "roberta_teacher_soft_targets_train.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_metrics.json"
OOF_PRED_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_oof_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_notes.md"
FOLD_ASSIGNMENTS_PATH = OUTPUT_DIR / "roberta_dapt_cv_distill_fold_assignments.csv"


class SequenceDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        soft_targets: torch.Tensor,
        sample_weights: torch.Tensor,
        distill_mask: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.soft_targets = soft_targets
        self.sample_weights = sample_weights
        self.distill_mask = distill_mask

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "soft_targets": self.soft_targets[index],
            "sample_weights": self.sample_weights[index],
            "distill_mask": self.distill_mask[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Honest RoBERTa DAPT CV training with distillation.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--pretrained-dir", type=Path, default=DAPT_MODEL_DIR)
    parser.add_argument("--teacher-path", type=Path, default=TEACHER_TARGETS_PATH)
    parser.add_argument("--pseudo-label-path", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=SPLIT_SEEDS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS)
    parser.add_argument("--class-weight-false", type=float, default=CLASS_WEIGHT_FALSE)
    parser.add_argument("--class-weight-true", type=float, default=CLASS_WEIGHT_TRUE)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--distill-alpha", type=float, default=DISTILL_ALPHA)
    parser.add_argument("--pseudo-weight", type=float, default=PSEUDO_WEIGHT)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--threshold-min", type=float, default=THRESHOLD_MIN)
    parser.add_argument("--threshold-max", type=float, default=THRESHOLD_MAX)
    parser.add_argument("--threshold-step", type=float, default=THRESHOLD_STEP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def resolve_base_model_source(pretrained_dir: Path, model_name: str) -> tuple[str, str]:
    if pretrained_dir.exists():
        return str(pretrained_dir), "dapt_checkpoint"
    return model_name, "base_model"


def build_optimizer(
    model: AutoModelForSequenceClassification,
    learning_rate: float,
    weight_decay: float,
    adam_eps: float,
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


def one_hot_with_label_smoothing(
    labels: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

    target = torch.zeros((labels.shape[0], 2), device=labels.device, dtype=torch.float32)
    target.scatter_(1, labels.unsqueeze(1), 1.0)
    if label_smoothing == 0.0:
        return target

    off_value = label_smoothing / 2.0
    on_value = 1.0 - label_smoothing + off_value
    return target * (on_value - off_value) + off_value


def per_example_soft_cross_entropy(logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_distribution * log_probs).sum(dim=-1)


def compute_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor,
    distill_mask: torch.Tensor,
    class_weight_false: float,
    class_weight_true: float,
    label_smoothing: float,
    distill_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_targets = one_hot_with_label_smoothing(labels=labels, label_smoothing=label_smoothing)
    hard_loss = per_example_soft_cross_entropy(logits, hard_targets)

    class_weights = torch.where(
        labels == 1,
        torch.full_like(sample_weights, fill_value=class_weight_true),
        torch.full_like(sample_weights, fill_value=class_weight_false),
    )
    weighted_hard_loss = hard_loss * class_weights * sample_weights

    distill_distribution = torch.stack(
        [1.0 - soft_targets.clamp(1e-5, 1.0 - 1e-5), soft_targets.clamp(1e-5, 1.0 - 1e-5)],
        dim=1,
    )
    distill_loss = per_example_soft_cross_entropy(logits, distill_distribution) * sample_weights * distill_mask

    total_loss = weighted_hard_loss + (distill_alpha * distill_loss)
    return total_loss.mean(), weighted_hard_loss.mean(), distill_loss.mean()


def compute_eval_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight_false: float,
    class_weight_true: float,
    label_smoothing: float,
) -> torch.Tensor:
    hard_targets = one_hot_with_label_smoothing(labels=labels, label_smoothing=label_smoothing)
    hard_loss = per_example_soft_cross_entropy(logits, hard_targets)
    class_weights = torch.where(
        labels == 1,
        torch.full((labels.shape[0],), fill_value=class_weight_true, device=labels.device),
        torch.full((labels.shape[0],), fill_value=class_weight_false, device=labels.device),
    )
    return (hard_loss * class_weights).mean()


def train_one_epoch(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    grad_accum: int,
    class_weight_false: float,
    class_weight_true: float,
    label_smoothing: float,
    distill_alpha: float,
    epoch_label: str,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_hard_loss = 0.0
    total_distill_loss = 0.0
    seen_batches = 0

    progress = tqdm(dataloader, desc=epoch_label, unit="batch")
    for step, batch in enumerate(progress, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        soft_targets = batch["soft_targets"].to(device)
        sample_weights = batch["sample_weights"].to(device)
        distill_mask = batch["distill_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, hard_loss, distill_loss = compute_training_loss(
            logits=outputs.logits,
            labels=labels,
            soft_targets=soft_targets,
            sample_weights=sample_weights,
            distill_mask=distill_mask,
            class_weight_false=class_weight_false,
            class_weight_true=class_weight_true,
            label_smoothing=label_smoothing,
            distill_alpha=distill_alpha,
        )

        if not math.isfinite(float(loss.detach().cpu().item())):
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / grad_accum).backward()

        total_loss += float(loss.detach().cpu().item())
        total_hard_loss += float(hard_loss.detach().cpu().item())
        total_distill_loss += float(distill_loss.detach().cpu().item())
        seen_batches += 1

        if (step % grad_accum == 0) or (step == len(dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        progress.set_postfix(
            total_loss=f"{(total_loss / max(seen_batches, 1)):.4f}",
            hard=f"{(total_hard_loss / max(seen_batches, 1)):.4f}",
            distill=f"{(total_distill_loss / max(seen_batches, 1)):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return {
        "train_loss": float(total_loss / max(seen_batches, 1)),
        "train_hard_loss": float(total_hard_loss / max(seen_batches, 1)),
        "train_distill_loss": float(total_distill_loss / max(seen_batches, 1)),
    }


def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
    class_weight_false: float,
    class_weight_true: float,
    label_smoothing: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    prob_chunks = []
    label_chunks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = compute_eval_loss(
                logits=outputs.logits,
                labels=labels,
                class_weight_false=class_weight_false,
                class_weight_true=class_weight_true,
                label_smoothing=label_smoothing,
            )

            prob_true = torch.softmax(outputs.logits, dim=1)[:, 1]
            prob_chunks.append(prob_true.detach().cpu().numpy())
            label_chunks.append(labels.detach().cpu().numpy())
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    return (
        float(total_loss / max(total_batches, 1)),
        np.concatenate(prob_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
    )


def predict_probabilities(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    model.eval()
    prob_chunks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prob_true = torch.softmax(outputs.logits, dim=1)[:, 1]
            prob_chunks.append(prob_true.detach().cpu().numpy())

    return np.concatenate(prob_chunks, axis=0)


def build_unlabeled_loader(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dummy_labels = torch.zeros((input_ids.shape[0],), dtype=torch.long)
    dummy_soft_targets = torch.full((input_ids.shape[0],), fill_value=0.5, dtype=torch.float32)
    dummy_sample_weights = torch.ones((input_ids.shape[0],), dtype=torch.float32)
    dummy_distill_mask = torch.zeros((input_ids.shape[0],), dtype=torch.float32)
    dataset = SequenceDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=dummy_labels,
        soft_targets=dummy_soft_targets,
        sample_weights=dummy_sample_weights,
        distill_mask=dummy_distill_mask,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def align_teacher_targets(
    teacher_path: Path | None,
    train_texts: list[str],
    train_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if teacher_path is None or not teacher_path.exists():
        return (
            np.full(len(train_texts), np.nan, dtype=np.float64),
            np.ones(len(train_texts), dtype=np.float64),
            np.zeros(len(train_texts), dtype=np.float32),
        )

    teacher_df = pd.read_csv(teacher_path)
    if "teacher_prob_TRUE" not in teacher_df.columns:
        raise ValueError(f"{teacher_path} must contain teacher_prob_TRUE.")

    target_keys = make_occurrence_keys(train_texts, train_labels)
    aligned_prob = np.full(len(train_texts), np.nan, dtype=np.float64)
    aligned_sample_weight = np.ones(len(train_texts), dtype=np.float64)
    aligned_mask = np.zeros(len(train_texts), dtype=np.float32)

    if "row_index" in teacher_df.columns and len(teacher_df) == len(train_texts):
        row_indices = teacher_df["row_index"].to_numpy(dtype=np.int64)
        aligned_prob[row_indices] = teacher_df["teacher_prob_TRUE"].to_numpy(dtype=np.float64)
        if "sample_weight" in teacher_df.columns:
            aligned_sample_weight[row_indices] = teacher_df["sample_weight"].to_numpy(dtype=np.float64)
        teacher_count = teacher_df["teacher_count"].to_numpy(dtype=np.int64) if "teacher_count" in teacher_df.columns else np.ones(len(teacher_df), dtype=np.int64)
        aligned_mask[row_indices] = (teacher_count > 0).astype(np.float32)
        return aligned_prob, aligned_sample_weight, aligned_mask

    source_keys = make_occurrence_keys(
        teacher_df["text"].tolist(),
        [normalize_label(value) for value in teacher_df["label"].tolist()] if "label" in teacher_df.columns else None,
    )
    target_lookup = {key: idx for idx, key in enumerate(target_keys)}
    for source_idx, key in enumerate(source_keys):
        target_idx = target_lookup.get(key)
        if target_idx is None:
            continue
        aligned_prob[target_idx] = float(teacher_df.iloc[source_idx]["teacher_prob_TRUE"])
        if "sample_weight" in teacher_df.columns:
            aligned_sample_weight[target_idx] = float(teacher_df.iloc[source_idx]["sample_weight"])
        teacher_count = int(teacher_df.iloc[source_idx]["teacher_count"]) if "teacher_count" in teacher_df.columns else 1
        aligned_mask[target_idx] = 1.0 if teacher_count > 0 else 0.0

    return aligned_prob, aligned_sample_weight, aligned_mask


def align_pseudo_labels(
    pseudo_path: Path | None,
    test_texts: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if pseudo_path is None:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo label file not found: {pseudo_path}")

    pseudo_df = pd.read_csv(pseudo_path)
    if "pseudo_label" not in pseudo_df.columns or "pseudo_prob_TRUE" not in pseudo_df.columns or "text" not in pseudo_df.columns:
        raise ValueError(f"{pseudo_path} must contain text, pseudo_label, and pseudo_prob_TRUE columns.")

    if "test_row_index" in pseudo_df.columns:
        pseudo_df = pseudo_df.sort_values("test_row_index").reset_index(drop=True)
        texts = pseudo_df["text"].tolist()
    else:
        target_keys = make_occurrence_keys(test_texts)
        source_keys = make_occurrence_keys(pseudo_df["text"].tolist())
        target_lookup = {key: idx for idx, key in enumerate(target_keys)}
        pseudo_df["__target_idx"] = [target_lookup[key] for key in source_keys]
        pseudo_df = pseudo_df.sort_values("__target_idx").reset_index(drop=True)
        texts = pseudo_df["text"].tolist()

    label_int = np.asarray([1 if normalize_label(value) == "TRUE" else 0 for value in pseudo_df["pseudo_label"].tolist()], dtype=np.int64)
    prob_true = pseudo_df["pseudo_prob_TRUE"].to_numpy(dtype=np.float64)
    return texts, label_int, prob_true


def build_dataset(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: np.ndarray,
    soft_targets: np.ndarray,
    sample_weights: np.ndarray,
    distill_mask: np.ndarray,
) -> SequenceDataset:
    return SequenceDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.as_tensor(labels, dtype=torch.long),
        soft_targets=torch.as_tensor(soft_targets, dtype=torch.float32),
        sample_weights=torch.as_tensor(sample_weights, dtype=torch.float32),
        distill_mask=torch.as_tensor(distill_mask, dtype=torch.float32),
    )


def write_notes(metrics: dict) -> None:
    seed_lines = []
    for seed_run in metrics["seed_runs"]:
        seed_lines.append(
            f"- split seed `{seed_run['split_seed']}`: threshold `{seed_run['oof_final']['threshold']:.2f}`, "
            f"F1 `{seed_run['oof_final']['f1']:.6f}`, ROC AUC `{seed_run['oof_final']['roc_auc']:.6f}`"
        )

    notes = f"""# RoBERTa DAPT CV Distill Notes

## Purpose

This script is the new honest mainline:
- optional in-domain DAPT checkpoint initialization
- artifact-aware stratified CV
- longer-context RoBERTa fine-tuning
- teacher soft-target distillation
- optional balanced pseudo-label augmentation

## Configuration

- Base model: `{metrics['model_source']['model_name']}`
- Source mode: `{metrics['model_source']['source_mode']}`
- Loaded from: `{metrics['model_source']['load_path']}`
- Max length: `{metrics['config']['max_length']}`
- Folds: `{metrics['config']['n_splits']}`
- Split seeds: `{metrics['config']['split_seeds']}`
- Epochs: `{metrics['config']['epochs']}`
- Distill alpha: `{metrics['config']['distill_alpha']}`
- Label smoothing: `{metrics['config']['label_smoothing']}`
- Class weights: `[FALSE={metrics['config']['class_weight_false']}, TRUE={metrics['config']['class_weight_true']}]`
- Pseudo labels: `{metrics['config']['pseudo_label_path']}`

## Seed Runs

{chr(10).join(seed_lines)}

## Final OOF

- Threshold: `{metrics['oof_final']['threshold']:.2f}`
- Accuracy: `{metrics['oof_final']['accuracy']:.6f}`
- Precision: `{metrics['oof_final']['precision']:.6f}`
- Recall: `{metrics['oof_final']['recall']:.6f}`
- F1: `{metrics['oof_final']['f1']:.6f}`
- ROC AUC: `{metrics['oof_final']['roc_auc']:.6f}`

## Outputs

- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_oof_predictions.csv`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_test_probabilities.csv`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_submission.csv`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_metrics.json`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_notes.md`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Honest RoBERTa DAPT CV Distillation ===")

    set_seeds(args.seed)
    device = detect_device()
    print(f"device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    base_model_source, source_mode = resolve_base_model_source(args.pretrained_dir, args.model_name)
    print(f"model source: {base_model_source} ({source_mode})")

    thresholds = np.round(np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step), 2)

    print("\n=== Loading competition data ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_competition_inputs(train_df, test_df, solution_df)

    train_texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    train_labels = [normalize_label(value) for value in train_df["label"].tolist()]
    y = encode_labels(train_labels, desc="Encoding train labels")

    print("\n=== Loading teacher targets ===")
    teacher_prob_true, teacher_sample_weights, teacher_distill_mask = align_teacher_targets(
        teacher_path=args.teacher_path,
        train_texts=train_texts,
        train_labels=train_labels,
    )

    print("\n=== Loading pseudo labels ===")
    pseudo_texts, pseudo_labels, pseudo_prob_true = align_pseudo_labels(
        pseudo_path=args.pseudo_label_path,
        test_texts=test_texts,
    )

    print("\n=== Building artifact-aware stratification ===")
    feature_df = build_text_feature_df(train_texts, desc="Extracting train text features")
    artifact_thresholds = fit_artifact_thresholds(feature_df)
    artifact_buckets = assign_artifact_buckets(feature_df, artifact_thresholds)
    stratify_labels = make_artifact_stratify_labels(train_labels, artifact_buckets, min_count=max(args.n_splits, 2))

    fold_assignment_df = pd.DataFrame(
        {
            "row_index": np.arange(len(train_texts), dtype=np.int64),
            "text": train_texts,
            "label": train_labels,
            "artifact_bucket": artifact_buckets,
            "stratify_label": stratify_labels,
        }
    )

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_source, use_fast=True)

    print("\n=== Tokenizing train/test text ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=TOKENIZE_BATCH_SIZE,
        desc="Tokenizing train text",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=TOKENIZE_BATCH_SIZE,
        desc="Tokenizing test text",
    )

    pseudo_input_ids = torch.zeros((0, args.max_length), dtype=torch.long)
    pseudo_attention_mask = torch.zeros((0, args.max_length), dtype=torch.long)
    if pseudo_texts:
        print("\n=== Tokenizing pseudo-labeled text ===")
        pseudo_input_ids, pseudo_attention_mask = tokenize_texts(
            pseudo_texts,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=TOKENIZE_BATCH_SIZE,
            desc="Tokenizing pseudo-labeled text",
        )

    test_loader = build_unlabeled_loader(
        input_ids=test_input_ids,
        attention_mask=test_attention_mask,
        batch_size=args.eval_batch_size,
        device=device,
    )

    all_seed_oof = np.zeros((len(train_texts), len(args.split_seeds)), dtype=np.float64)
    all_seed_test = np.zeros((len(test_texts), len(args.split_seeds)), dtype=np.float64)
    seed_runs = []

    for seed_idx, split_seed in enumerate(args.split_seeds):
        print(f"\n=== CV split seed {split_seed} ===")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=split_seed)
        seed_oof = np.zeros(len(train_texts), dtype=np.float64)
        seed_oof_seen = np.zeros(len(train_texts), dtype=bool)
        seed_test_sum = np.zeros(len(test_texts), dtype=np.float64)
        fold_summaries = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_texts, stratify_labels), start=1):
            print(f"\n--- split_seed={split_seed}, fold={fold_idx}/{args.n_splits} ---")
            fold_assignment_df.loc[val_idx, f"split_seed_{split_seed}_fold"] = fold_idx

            fold_dir = CHECKPOINTS_DIR / f"seed_{split_seed}" / f"fold_{fold_idx}"
            best_model_dir = fold_dir / "best_model"
            best_state_path = fold_dir / "best_state_dict.pt"
            best_meta_path = fold_dir / "best_metadata.json"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                base_model_source,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model.to(device)

            train_soft_targets = np.nan_to_num(teacher_prob_true[train_idx], nan=0.5)
            train_weights = teacher_sample_weights[train_idx].copy()
            train_distill = teacher_distill_mask[train_idx].copy()

            fold_train_input_ids = train_input_ids[train_idx]
            fold_train_attention = train_attention_mask[train_idx]
            fold_train_labels = y[train_idx]

            if len(pseudo_labels) > 0:
                fold_train_input_ids = torch.cat([fold_train_input_ids, pseudo_input_ids], dim=0)
                fold_train_attention = torch.cat([fold_train_attention, pseudo_attention_mask], dim=0)
                fold_train_labels = np.concatenate([fold_train_labels, pseudo_labels], axis=0)
                train_soft_targets = np.concatenate([train_soft_targets, pseudo_prob_true], axis=0)
                train_weights = np.concatenate(
                    [train_weights, np.full(len(pseudo_labels), fill_value=args.pseudo_weight, dtype=np.float64)],
                    axis=0,
                )
                train_distill = np.concatenate([train_distill, np.ones(len(pseudo_labels), dtype=np.float32)], axis=0)

            train_dataset = build_dataset(
                input_ids=fold_train_input_ids,
                attention_mask=fold_train_attention,
                labels=fold_train_labels,
                soft_targets=train_soft_targets,
                sample_weights=train_weights,
                distill_mask=train_distill,
            )

            val_dataset = build_dataset(
                input_ids=train_input_ids[val_idx],
                attention_mask=train_attention_mask[val_idx],
                labels=y[val_idx],
                soft_targets=np.full(len(val_idx), 0.5, dtype=np.float64),
                sample_weights=np.ones(len(val_idx), dtype=np.float64),
                distill_mask=np.zeros(len(val_idx), dtype=np.float32),
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(split_seed + fold_idx),
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )

            optimizer = build_optimizer(
                model=model,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                adam_eps=args.adam_eps,
            )
            steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
            total_training_steps = max(steps_per_epoch * args.epochs, 1)
            warmup_steps = int(args.warmup_ratio * total_training_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )

            best_epoch = -1
            best_val_auc = -1.0
            best_val_loss = math.inf
            best_val_probs = None
            best_val_true = None
            best_epoch_history = None
            patience_counter = 0
            epoch_history = []

            for epoch in range(1, args.epochs + 1):
                train_history = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    grad_accum=args.grad_accum,
                    class_weight_false=args.class_weight_false,
                    class_weight_true=args.class_weight_true,
                    label_smoothing=args.label_smoothing,
                    distill_alpha=args.distill_alpha,
                    epoch_label=f"seed={split_seed} fold={fold_idx} epoch={epoch} train",
                )
                val_loss, val_prob_true, val_true = evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    desc=f"seed={split_seed} fold={fold_idx} epoch={epoch} val",
                    class_weight_false=args.class_weight_false,
                    class_weight_true=args.class_weight_true,
                    label_smoothing=args.label_smoothing,
                )
                val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_history["train_loss"],
                    "train_hard_loss": train_history["train_hard_loss"],
                    "train_distill_loss": train_history["train_distill_loss"],
                    "val_loss": val_loss,
                    "val_f1_threshold_0_5": val_metrics_05["f1"],
                    "val_roc_auc": val_metrics_05["roc_auc"],
                    "val_precision_threshold_0_5": val_metrics_05["precision"],
                    "val_recall_threshold_0_5": val_metrics_05["recall"],
                }
                epoch_history.append(epoch_record)

                improved = (val_metrics_05["roc_auc"] > best_val_auc) or (
                    np.isclose(val_metrics_05["roc_auc"], best_val_auc) and val_loss < best_val_loss
                )

                if improved:
                    best_epoch = epoch
                    best_val_auc = float(val_metrics_05["roc_auc"])
                    best_val_loss = float(val_loss)
                    best_val_probs = val_prob_true.copy()
                    best_val_true = val_true.copy()
                    best_epoch_history = epoch_record
                    patience_counter = 0

                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    torch.save(
                        {
                            "epoch": epoch,
                            "val_loss": best_val_loss,
                            "val_roc_auc": best_val_auc,
                            "model_state_dict": model.state_dict(),
                        },
                        best_state_path,
                    )
                    best_meta_path.write_text(json.dumps(epoch_record, indent=2), encoding="utf-8")
                else:
                    patience_counter += 1

                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} for split_seed={split_seed}, fold={fold_idx}.")
                    break

            if best_val_probs is None or best_val_true is None or best_epoch_history is None:
                raise RuntimeError(f"No best checkpoint recorded for split_seed={split_seed}, fold={fold_idx}.")

            seed_oof[val_idx] = best_val_probs
            seed_oof_seen[val_idx] = True

            best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
            best_model.to(device)
            fold_test_prob = predict_probabilities(
                model=best_model,
                dataloader=test_loader,
                device=device,
                desc=f"seed={split_seed} fold={fold_idx} test",
            )
            seed_test_sum += fold_test_prob

            fold_best_metrics = compute_metrics(best_val_true, best_val_probs, threshold=0.5)
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_roc_auc": best_val_auc,
                    "best_val_f1_threshold_0_5": fold_best_metrics["f1"],
                    "best_val_precision_threshold_0_5": fold_best_metrics["precision"],
                    "best_val_recall_threshold_0_5": fold_best_metrics["recall"],
                    "val_rows": int(len(val_idx)),
                    "train_rows": int(len(train_idx)),
                    "pseudo_rows_added": int(len(pseudo_labels)),
                    "epoch_history": epoch_history,
                    "paths": {
                        "best_model_dir": str(best_model_dir),
                        "best_state_dict": str(best_state_path),
                        "best_metadata": str(best_meta_path),
                    },
                }
            )

        if not np.all(seed_oof_seen):
            missing = int((~seed_oof_seen).sum())
            raise RuntimeError(f"Split seed {split_seed} is missing {missing} OOF rows.")

        seed_test_mean = seed_test_sum / args.n_splits
        all_seed_oof[:, seed_idx] = seed_oof
        all_seed_test[:, seed_idx] = seed_test_mean

        seed_best_threshold, seed_threshold_table = tune_threshold_for_f1(
            y_true=y,
            prob_true=seed_oof,
            thresholds=thresholds,
            desc=f"Threshold search (split seed {split_seed})",
            tie_break_target=0.5,
        )
        seed_metrics = compute_metrics(y_true=y, prob_true=seed_oof, threshold=seed_best_threshold)
        seed_runs.append(
            {
                "split_seed": split_seed,
                "folds": fold_summaries,
                "threshold_tuning": {
                    "selection_objective": "max_seed_oof_f1",
                    "selected_threshold": float(seed_best_threshold),
                    "table": seed_threshold_table,
                },
                "oof_final": seed_metrics,
            }
        )

    avg_oof_prob = all_seed_oof.mean(axis=1)
    avg_test_prob = all_seed_test.mean(axis=1)

    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=y,
        prob_true=avg_oof_prob,
        thresholds=thresholds,
        desc="Threshold search (mean OOF)",
        tie_break_target=0.5,
    )
    final_oof_metrics = compute_metrics(y_true=y, prob_true=avg_oof_prob, threshold=best_threshold)

    oof_pred = (avg_oof_prob >= best_threshold).astype(int)
    oof_payload = {
        "row_index": np.arange(len(train_texts), dtype=np.int64),
        "text": train_texts,
        "true_label": train_labels,
        "artifact_bucket": artifact_buckets,
        "oof_prob_TRUE_mean": avg_oof_prob,
        "oof_pred_label": decode_labels(oof_pred),
    }
    for seed_idx, split_seed in enumerate(args.split_seeds):
        oof_payload[f"oof_prob_TRUE_seed_{split_seed}"] = all_seed_oof[:, seed_idx]
    oof_df = pd.DataFrame(oof_payload)
    oof_df.to_csv(OOF_PRED_PATH, index=False)

    errors = oof_df[oof_df["true_label"] != oof_df["oof_pred_label"]].copy()
    errors["error_type"] = np.where(
        (errors["true_label"] == "FALSE") & (errors["oof_pred_label"] == "TRUE"),
        "FALSE_to_TRUE",
        "TRUE_to_FALSE",
    )
    errors["confidence"] = np.where(
        errors["oof_pred_label"] == "TRUE",
        errors["oof_prob_TRUE_mean"],
        1.0 - errors["oof_prob_TRUE_mean"],
    )
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(ERROR_ANALYSIS_PATH, index=False)

    test_prob_payload = {
        "text": test_texts,
        "pred_prob_TRUE": avg_test_prob,
    }
    for seed_idx, split_seed in enumerate(args.split_seeds):
        test_prob_payload[f"pred_prob_TRUE_seed_{split_seed}"] = all_seed_test[:, seed_idx]
    test_prob_df = pd.DataFrame(test_prob_payload)
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)

    test_pred = (avg_test_prob >= best_threshold).astype(int)
    submission_df = make_submission(solution_df, decode_labels(test_pred))
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    fold_assignment_df.to_csv(FOLD_ASSIGNMENTS_PATH, index=False)

    metrics = {
        "model_source": {
            "model_name": args.model_name,
            "source_mode": source_mode,
            "load_path": base_model_source,
        },
        "device": str(device),
        "config": {
            "max_length": args.max_length,
            "n_splits": args.n_splits,
            "split_seeds": args.split_seeds,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "class_weight_false": args.class_weight_false,
            "class_weight_true": args.class_weight_true,
            "label_smoothing": args.label_smoothing,
            "distill_alpha": args.distill_alpha,
            "pseudo_weight": args.pseudo_weight,
            "early_stopping_patience": args.early_stopping_patience,
            "warmup_ratio": args.warmup_ratio,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "teacher_path": str(args.teacher_path) if args.teacher_path else None,
            "pseudo_label_path": str(args.pseudo_label_path) if args.pseudo_label_path else None,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "data": {
            "train_rows": len(train_texts),
            "test_rows": len(test_texts),
            "teacher_rows_with_soft_targets": int(teacher_distill_mask.sum()),
            "pseudo_rows_added": int(len(pseudo_labels)),
        },
        "artifact_stratification": {
            "thresholds": artifact_thresholds,
            "bucket_distribution": pd.Series(artifact_buckets).value_counts().to_dict(),
            "label_bucket_distribution": pd.Series(stratify_labels).value_counts().to_dict(),
        },
        "seed_runs": seed_runs,
        "threshold_tuning": {
            "selection_objective": "max_mean_oof_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "oof_final": final_oof_metrics,
        "test_prediction_distribution": {
            "pred_FALSE": int((test_pred == 0).sum()),
            "pred_TRUE": int((test_pred == 1).sum()),
        },
        "paths": {
            "metrics": str(METRICS_PATH),
            "oof_predictions": str(OOF_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
            "fold_assignments": str(FOLD_ASSIGNMENTS_PATH),
            "fold_checkpoints_dir": str(CHECKPOINTS_DIR),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_notes(metrics)

    print(f"Saved OOF predictions: {OOF_PRED_PATH}")
    print(f"Saved test probabilities: {TEST_PROB_PATH}")
    print(f"Saved submission: {SUBMISSION_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
