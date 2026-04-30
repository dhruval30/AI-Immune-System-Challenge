#!/usr/bin/env python3
"""
Train one stronger RoBERTa-base model with DAPT + hard-weighted fine-tuning.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This is not an ensemble and not a backbone swap.

Pipeline:
1. Continue masked-language-model pretraining of roberta-base on provided competition text only.
2. Fine-tune the adapted RoBERTa as a binary classifier using the hard-weighted recipe that produced the best GPU run.
3. Save tuned-threshold and fixed-threshold-0.32 submissions.

The intent is to make the same RoBERTa-base model stronger before supervised training, rather than adding brittle side features or changing model families.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from train_roberta_hard_weighted_gpu import (
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

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
LOCAL_MODEL_NAME = "/workspace/gpu/roberta-base"
HF_MODEL_NAME = "roberta-base"
MODEL_NAME = LOCAL_MODEL_NAME if Path(LOCAL_MODEL_NAME).exists() else HF_MODEL_NAME

MAX_LENGTH = 384
TOKENIZE_BATCH_SIZE = 256

# DAPT / MLM settings. Keep this moderate: enough to adapt to competition style, not enough to memorize.
USE_TEST_TEXT_FOR_DAPT = True
DAPT_EPOCHS = 2
DAPT_TRAIN_BATCH_SIZE = 16
DAPT_EVAL_BATCH_SIZE = 32
DAPT_GRADIENT_ACCUMULATION_STEPS = 1
DAPT_LEARNING_RATE = 2e-5
DAPT_WEIGHT_DECAY = 0.01
DAPT_ADAM_EPS = 1e-8
DAPT_MLM_PROBABILITY = 0.15
DAPT_EVAL_SPLIT = 0.05

# Supervised hard-weighted fine-tuning settings.
CLASSIFIER_EPOCHS = 8
CLASSIFIER_TRAIN_BATCH_SIZE = 16
CLASSIFIER_EVAL_BATCH_SIZE = 64
CLASSIFIER_GRADIENT_ACCUMULATION_STEPS = 1
CLASSIFIER_LEARNING_RATE = 8e-6
CLASSIFIER_WEIGHT_DECAY = 0.01
CLASSIFIER_ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
FIXED_SUBMISSION_THRESHOLD = 0.32
THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_dapt_hard_weighted_gpu"
DAPT_BEST_MODEL_DIR = OUTPUT_DIR / "dapt_best_model"
DAPT_LAST_MODEL_DIR = OUTPUT_DIR / "dapt_last_model"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_DICT_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_best_state_dict.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_submission.csv"
SUBMISSION_THR032_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_submission_thr0p32.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_notes.md"
WEIGHT_SUMMARY_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_weight_summary.csv"
FEATURE_THRESHOLDS_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_feature_thresholds.json"
DAPT_SPLIT_META_PATH = OUTPUT_DIR / "roberta_dapt_hard_weighted_gpu_dapt_split_meta.json"


class MLMDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.special_tokens_mask = special_tokens_mask

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "special_tokens_mask": self.special_tokens_mask[index],
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


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float, adam_eps: float) -> AdamW:
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


def tokenize_mlm_texts(
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []
    all_special_masks = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
        all_special_masks.append(encoded["special_tokens_mask"])
    return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0), torch.cat(all_special_masks, dim=0)


def evaluate_mlm(model: AutoModelForMaskedLM, dataloader: DataLoader, device: torch.device, desc: str) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss_value = float(outputs.loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite MLM eval loss during {desc}.")
            batch_size = int(batch["input_ids"].shape[0])
            total_loss += loss_value * batch_size
            total_examples += batch_size
    return total_loss / max(total_examples, 1)


def run_dapt(
    *,
    tokenizer,
    train_texts: list[str],
    test_texts: list[str],
    device: torch.device,
    pin_memory: bool,
) -> dict[str, Any]:
    print("\n=== Stage 1: Domain-Adaptive MLM Pretraining ===")
    dapt_texts = train_texts + test_texts if USE_TEST_TEXT_FOR_DAPT else train_texts
    indices = np.arange(len(dapt_texts))
    mlm_train_idx, mlm_eval_idx = train_test_split(
        indices,
        test_size=DAPT_EVAL_SPLIT,
        random_state=SEED,
        shuffle=True,
    )
    mlm_train_texts = [dapt_texts[i] for i in mlm_train_idx]
    mlm_eval_texts = [dapt_texts[i] for i in mlm_eval_idx]

    split_meta = {
        "seed": SEED,
        "use_test_text_for_dapt": USE_TEST_TEXT_FOR_DAPT,
        "combined_rows": len(dapt_texts),
        "mlm_train_rows": len(mlm_train_texts),
        "mlm_eval_rows": len(mlm_eval_texts),
        "eval_split": DAPT_EVAL_SPLIT,
    }
    DAPT_SPLIT_META_PATH.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    train_input_ids, train_attention_mask, train_special_mask = tokenize_mlm_texts(
        mlm_train_texts,
        tokenizer,
        MAX_LENGTH,
        TOKENIZE_BATCH_SIZE,
        "Tokenizing DAPT train text",
    )
    eval_input_ids, eval_attention_mask, eval_special_mask = tokenize_mlm_texts(
        mlm_eval_texts,
        tokenizer,
        MAX_LENGTH,
        TOKENIZE_BATCH_SIZE,
        "Tokenizing DAPT eval text",
    )

    train_dataset = MLMDataset(train_input_ids, train_attention_mask, train_special_mask)
    eval_dataset = MLMDataset(eval_input_ids, eval_attention_mask, eval_special_mask)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=DAPT_MLM_PROBABILITY,
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=DAPT_TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=DAPT_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin_memory,
    )

    local_files_only = Path(MODEL_NAME).exists()
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, local_files_only=local_files_only)
    mlm_model.to(device)

    optimizer = build_optimizer(mlm_model, DAPT_LEARNING_RATE, DAPT_WEIGHT_DECAY, DAPT_ADAM_EPS)
    steps_per_epoch = math.ceil(len(train_loader) / DAPT_GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * DAPT_EPOCHS
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_eval_loss = float("inf")
    best_epoch = -1
    dapt_history = []

    for epoch in range(1, DAPT_EPOCHS + 1):
        mlm_model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0
        total_examples = 0
        skipped_batches = 0

        progress = tqdm(train_loader, desc=f"DAPT epoch {epoch}/{DAPT_EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = mlm_model(**batch)
            loss = outputs.loss
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(train_loss="non_finite", skipped=skipped_batches)
                continue

            (loss / DAPT_GRADIENT_ACCUMULATION_STEPS).backward()
            batch_size = int(batch["input_ids"].shape[0])
            total_train_loss += loss_value * batch_size
            total_examples += batch_size

            if (step % DAPT_GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                if not math.isfinite(grad_norm_value):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{total_train_loss / max(total_examples, 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                skipped=skipped_batches,
            )

        train_loss = total_train_loss / max(total_examples, 1)
        eval_loss = evaluate_mlm(mlm_model, eval_loader, device, f"DAPT epoch {epoch}/{DAPT_EPOCHS} - eval")
        dapt_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "skipped_batches": int(skipped_batches),
        }
        dapt_history.append(dapt_record)
        print(f"DAPT epoch {epoch} | train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = float(eval_loss)
            best_epoch = int(epoch)
            mlm_model.save_pretrained(DAPT_BEST_MODEL_DIR)
            tokenizer.save_pretrained(DAPT_BEST_MODEL_DIR)
            print(f"Saved DAPT best model at epoch {best_epoch} with eval_loss={best_eval_loss:.6f}")

    mlm_model.save_pretrained(DAPT_LAST_MODEL_DIR)
    tokenizer.save_pretrained(DAPT_LAST_MODEL_DIR)
    mlm_model.to("cpu")
    del mlm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "history": dapt_history,
        "split": split_meta,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }


def write_weight_summary(train_weight_df: pd.DataFrame, y_train: np.ndarray, train_weights: np.ndarray) -> None:
    summary_df = train_weight_df.copy()
    summary_df["label"] = decode_labels(y_train)
    summary_df["normalized_sample_weight"] = train_weights
    grouped = (
        summary_df.groupby(["label", "hard_category"])
        .agg(
            rows=("hard_category", "size"),
            mean_abnormal_score=("abnormal_score", "mean"),
            mean_raw_weight=("raw_sample_weight", "mean"),
            mean_normalized_weight=("normalized_sample_weight", "mean"),
        )
        .reset_index()
        .sort_values(["label", "hard_category"])
    )
    grouped.to_csv(WEIGHT_SUMMARY_PATH, index=False)


def build_classifier_optimizer(model: AutoModelForSequenceClassification) -> AdamW:
    return build_optimizer(model, CLASSIFIER_LEARNING_RATE, CLASSIFIER_WEIGHT_DECAY, CLASSIFIER_ADAM_EPS)


def make_submission(solution_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = labels
    return submission_df[solution_df.columns.tolist()]


def write_notes(metrics_payload: dict[str, Any]) -> None:
    notes = f"""# RoBERTa DAPT + Hard-Weighted Notes

## Why This Run Exists

The best single-model family has been RoBERTa-base with hard-example weighting. Model-family swaps and TF-IDF side branches did not transfer. This script keeps RoBERTa-base, but makes the model stronger before supervision using domain-adaptive masked-language-model pretraining on the provided competition text only.

This is one final model, not an ensemble.

## Pipeline

1. DAPT MLM on competition text.
2. Hard-weighted supervised fine-tuning from the DAPT checkpoint.
3. Save tuned-threshold and fixed `0.32` submissions.

## DAPT Configuration

- Base model: `{MODEL_NAME}`
- Uses test text for unsupervised DAPT: `{USE_TEST_TEXT_FOR_DAPT}`
- DAPT epochs: `{DAPT_EPOCHS}`
- DAPT LR: `{DAPT_LEARNING_RATE}`
- MLM probability: `{DAPT_MLM_PROBABILITY}`
- Best DAPT eval loss: `{metrics_payload['dapt']['best_eval_loss']:.6f}`

## Fine-Tuning Configuration

- Max length: `{MAX_LENGTH}`
- Classifier epochs: `{CLASSIFIER_EPOCHS}`
- Train batch size: `{CLASSIFIER_TRAIN_BATCH_SIZE}`
- Gradient accumulation: `{CLASSIFIER_GRADIENT_ACCUMULATION_STEPS}`
- Effective train batch size: `{CLASSIFIER_TRAIN_BATCH_SIZE * CLASSIFIER_GRADIENT_ACCUMULATION_STEPS}`
- Classifier LR: `{CLASSIFIER_LEARNING_RATE}`
- Loss: `sample_weighted_cross_entropy`
- Fixed submission threshold: `{FIXED_SUBMISSION_THRESHOLD}`

## Comparison Targets

- RoBERTa base public LB: `0.90909091`
- Hard-weighted RoBERTa GPU public LB: `0.9347079`
- R-Drop hard-weighted RoBERTa GPU public LB: `0.93542757`

## Outputs

- `{DAPT_BEST_MODEL_DIR}/`
- `{BEST_MODEL_DIR}/`
- `{METRICS_PATH}`
- `{VAL_PRED_PATH}`
- `{ERROR_ANALYSIS_PATH}`
- `{TEST_PROB_PATH}`
- `{SUBMISSION_PATH}`
- `{SUBMISSION_THR032_PATH}`
- `{WEIGHT_SUMMARY_PATH}`
- `{FEATURE_THRESHOLDS_PATH}`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa DAPT + Hard-Weighted Training ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DAPT_BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DAPT_LAST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    configure_cuda_runtime()
    device = detect_device()
    pin_memory = device.type == "cuda"
    local_files_only = Path(MODEL_NAME).exists()
    print(f"Using device: {device}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Local files only: {local_files_only}")
    print(f"Use test text for unsupervised DAPT: {USE_TEST_TEXT_FOR_DAPT}")

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, desc="Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_inputs(train_df, test_df, solution_df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    print("\n=== Preparing text and labels ===")
    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=local_files_only)

    dapt_metrics = run_dapt(
        tokenizer=tokenizer,
        train_texts=texts,
        test_texts=test_texts,
        device=device,
        pin_memory=pin_memory,
    )

    print("\n=== Stage 2: Hard-Weighted Supervised Fine-Tuning ===")
    idx = np.arange(len(texts))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")

    train_features = compute_feature_frame(train_texts, "Computing train split hard features")
    val_features = compute_feature_frame(val_texts, "Computing validation hard features")
    test_features = compute_feature_frame(test_texts, "Computing test hard features")
    thresholds = fit_abnormal_thresholds(train_features)
    FEATURE_THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Saved hard-feature thresholds: {FEATURE_THRESHOLDS_PATH}")

    train_weight_df = build_weight_frame(y_train, train_features, thresholds)
    val_weight_df = build_weight_frame(y_val, val_features, thresholds)
    raw_train_weights = train_weight_df["raw_sample_weight"].to_numpy(dtype=np.float32)
    train_weights = raw_train_weights / max(float(raw_train_weights.mean()), 1e-12)
    write_weight_summary(train_weight_df, y_train, train_weights)
    print(f"Saved weight summary: {WEIGHT_SUMMARY_PATH}")
    print(f"Mean normalized train weight: {train_weights.mean():.6f}")

    test_scores = [abnormal_score(row, thresholds) for _, row in test_features.iterrows()]

    print("\n=== Loading DAPT checkpoint as classifier ===")
    classifier_tokenizer = AutoTokenizer.from_pretrained(DAPT_BEST_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(DAPT_BEST_MODEL_DIR, num_labels=2)
    model.to(device)

    print("\n=== Tokenizing supervised splits ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts, classifier_tokenizer, MAX_LENGTH, TOKENIZE_BATCH_SIZE, "Tokenizing train split"
    )
    val_input_ids, val_attention_mask = tokenize_texts(
        val_texts, classifier_tokenizer, MAX_LENGTH, TOKENIZE_BATCH_SIZE, "Tokenizing validation split"
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts, classifier_tokenizer, MAX_LENGTH, TOKENIZE_BATCH_SIZE, "Tokenizing test split"
    )

    train_dataset = TensorDataset(
        train_input_ids,
        train_attention_mask,
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(train_weights, dtype=torch.float32),
    )
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CLASSIFIER_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = build_classifier_optimizer(model)
    steps_per_epoch = math.ceil(len(train_loader) / CLASSIFIER_GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * CLASSIFIER_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Effective train batch size: {CLASSIFIER_TRAIN_BATCH_SIZE * CLASSIFIER_GRADIENT_ACCUMULATION_STEPS}")

    print("\n=== Supervised training ===")
    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    epoch_history = []

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_unweighted_loss = 0.0
        running_weight_mean = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{CLASSIFIER_EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels, sample_weights = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            per_row_loss = F.cross_entropy(outputs.logits, labels, reduction="none")
            loss = (per_row_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                non_finite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(train_loss="non_finite", skipped_loss=non_finite_loss_batches)
                continue

            (loss / CLASSIFIER_GRADIENT_ACCUMULATION_STEPS).backward()
            batch_size = input_ids.size(0)
            running_loss += loss_value * batch_size
            running_unweighted_loss += float(per_row_loss.mean().detach().cpu().item()) * batch_size
            running_weight_mean += float(sample_weights.mean().detach().cpu().item()) * batch_size
            seen_examples += batch_size

            if (step % CLASSIFIER_GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                if not math.isfinite(grad_norm_value):
                    non_finite_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{running_loss / max(seen_examples, 1):.4f}",
                raw_ce=f"{running_unweighted_loss / max(seen_examples, 1):.4f}",
                w_mean=f"{running_weight_mean / max(seen_examples, 1):.3f}",
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
            desc=f"Epoch {epoch}/{CLASSIFIER_EPOCHS} - validation",
            has_labels=True,
        )
        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_weighted_loss"] = float(train_loss)
        val_metrics_05["train_unweighted_loss"] = float(train_unweighted_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        epoch_history.append(val_metrics_05)

        print(
            f"Epoch {epoch} | train_weighted_loss={train_loss:.6f} "
            f"| train_raw_ce={train_unweighted_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            model.save_pretrained(BEST_MODEL_DIR)
            classifier_tokenizer.save_pretrained(BEST_MODEL_DIR)
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_f1": best_epoch_f1,
                    "model_state_dict": model.state_dict(),
                    "feature_thresholds": thresholds,
                    "dapt_best_model_dir": str(DAPT_BEST_MODEL_DIR),
                },
                BEST_STATE_DICT_PATH,
            )
            print(f"Saved new best classifier at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError("Best validation predictions were not captured.")

    print("\n=== Threshold tuning on validation probabilities ===")
    best_threshold, threshold_table = tune_threshold_for_f1(best_epoch_val_labels, best_epoch_val_probs, THRESHOLD_GRID)
    print(f"Best threshold by validation F1: {best_threshold:.2f}")
    final_val_metrics = compute_metrics(best_epoch_val_labels, best_epoch_val_probs, threshold=best_threshold)

    val_pred = (best_epoch_val_probs >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": decode_labels(best_epoch_val_labels),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": best_epoch_val_probs,
            "abnormal_score": val_weight_df["abnormal_score"].to_numpy(),
            "hard_category": val_weight_df["hard_category"].to_numpy(),
            "raw_sample_weight_if_trained": val_weight_df["raw_sample_weight"].to_numpy(),
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    errors = val_pred_df[val_pred_df["true_label"] != val_pred_df["pred_label"]].copy()
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
    error_df = pd.concat(
        [
            errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(200),
            errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(200),
        ],
        ignore_index=True,
    )
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Loading best classifier checkpoint for test inference ===")
    best_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    best_model.to(device)
    _, test_prob_true, _ = evaluate_model(best_model, test_loader, device, "Test inference", has_labels=False)
    test_prob_df = pd.DataFrame({"text": test_texts, "pred_prob_TRUE": test_prob_true, "abnormal_score": test_scores})
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {TEST_PROB_PATH}")

    tuned_pred = (test_prob_true >= best_threshold).astype(int)
    tuned_distribution = {"pred_FALSE": int((tuned_pred == 0).sum()), "pred_TRUE": int((tuned_pred == 1).sum())}
    make_submission(solution_df, decode_labels(tuned_pred)).to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved tuned-threshold submission: {SUBMISSION_PATH}")
    print(f"Tuned-threshold prediction distribution: {tuned_distribution}")

    fixed_pred = (test_prob_true >= FIXED_SUBMISSION_THRESHOLD).astype(int)
    fixed_distribution = {"pred_FALSE": int((fixed_pred == 0).sum()), "pred_TRUE": int((fixed_pred == 1).sum())}
    make_submission(solution_df, decode_labels(fixed_pred)).to_csv(SUBMISSION_THR032_PATH, index=False)
    print(f"Saved fixed threshold {FIXED_SUBMISSION_THRESHOLD:.2f} submission: {SUBMISSION_THR032_PATH}")
    print(f"Fixed-threshold prediction distribution: {fixed_distribution}")

    metrics_payload = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
        "dapt": dapt_metrics,
        "config": {
            "max_length": MAX_LENGTH,
            "use_test_text_for_dapt": USE_TEST_TEXT_FOR_DAPT,
            "dapt_epochs": DAPT_EPOCHS,
            "dapt_learning_rate": DAPT_LEARNING_RATE,
            "dapt_mlm_probability": DAPT_MLM_PROBABILITY,
            "classifier_epochs": CLASSIFIER_EPOCHS,
            "classifier_train_batch_size": CLASSIFIER_TRAIN_BATCH_SIZE,
            "classifier_eval_batch_size": CLASSIFIER_EVAL_BATCH_SIZE,
            "classifier_gradient_accumulation_steps": CLASSIFIER_GRADIENT_ACCUMULATION_STEPS,
            "classifier_effective_train_batch_size": CLASSIFIER_TRAIN_BATCH_SIZE * CLASSIFIER_GRADIENT_ACCUMULATION_STEPS,
            "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
            "weight_decay": CLASSIFIER_WEIGHT_DECAY,
            "adam_eps": CLASSIFIER_ADAM_EPS,
            "fixed_submission_threshold": FIXED_SUBMISSION_THRESHOLD,
            "loss": "sample_weighted_cross_entropy_after_dapt_mlm",
            "weight_strategy": {
                "hard_clean_TRUE": 2.5,
                "medium_clean_TRUE": 1.5,
                "easy_noisy_TRUE": 0.85,
                "hard_messy_FALSE": 2.0,
                "medium_messy_FALSE": 1.4,
                "easy_clean_FALSE": 0.85,
                "normalization": "divide train split raw weights by train split mean weight",
            },
            "feature_thresholds": thresholds,
        },
        "split": {
            "train_rows": int(len(train_texts)),
            "val_rows": int(len(val_texts)),
            "test_rows": int(len(test_texts)),
            "stratified": True,
            "random_state": SEED,
        },
        "epoch_metrics_threshold_0_5": epoch_history,
        "best_epoch": {
            "epoch": int(best_epoch),
            "val_f1_threshold_0_5": float(best_epoch_f1),
            "checkpoint_dir": str(BEST_MODEL_DIR),
            "state_dict_path": str(BEST_STATE_DICT_PATH),
        },
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_validation_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "validation_final": final_val_metrics,
        "test_prediction_distribution": tuned_distribution,
        "fixed_threshold_test_prediction_distribution": fixed_distribution,
        "paths": {
            "dapt_best_model_dir": str(DAPT_BEST_MODEL_DIR),
            "dapt_last_model_dir": str(DAPT_LAST_MODEL_DIR),
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "submission_thr0p32": str(SUBMISSION_THR032_PATH),
            "notes": str(NOTES_PATH),
            "weight_summary": str(WEIGHT_SUMMARY_PATH),
            "feature_thresholds": str(FEATURE_THRESHOLDS_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_DICT_PATH),
        },
        "comparison_target": {
            "roberta_base_public_lb": 0.90909091,
            "hard_weighted_gpu_public_lb": 0.9347079,
            "rdrop_hard_weighted_gpu_public_lb": 0.93542757,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, default=json_default), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")
    write_notes(metrics_payload)
    print(f"Saved notes: {NOTES_PATH}")
    print("\nRun complete.")


if __name__ == "__main__":
    main()
