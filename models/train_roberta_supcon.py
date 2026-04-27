#!/usr/bin/env python3
"""
Train RoBERTa with supervised contrastive loss plus classification loss.

Dependency install command:
pip install pandas numpy scikit-learn tqdm torch transformers accelerate

This script is a separate experiment cloned from the successful roberta-base
setup. It does not modify outputs/roberta_base. The model learns with:

    total_loss = CrossEntropyLoss + SUPCON_WEIGHT * SupervisedContrastiveLoss

The contrastive objective is intentionally light so it regularizes the embedding
space without overpowering classification on a noisy binary dataset.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0

SUPCON_WEIGHT = 0.05
SUPCON_TEMPERATURE = 0.07
PROJECTION_DIM = 128
DROPOUT = 0.10

THRESHOLD_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_supcon"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
BEST_STATE_DICT_PATH = OUTPUT_DIR / "roberta_supcon_best_state_dict.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_supcon_metrics.json"
VAL_PRED_PATH = OUTPUT_DIR / "roberta_supcon_val_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_supcon_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_supcon_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_supcon_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_supcon_notes.md"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, desc: str) -> pd.DataFrame:
    total = count_lines(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total, desc=desc, unit="lines"):
            line = line.strip()
            if line:
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
    return np.asarray(encoded, dtype=np.int64)


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    required_train_cols = {"label", "text"}
    required_test_cols = {"text"}
    missing_train_cols = sorted(required_train_cols - set(train_df.columns))
    missing_test_cols = sorted(required_test_cols - set(test_df.columns))
    if missing_train_cols:
        raise ValueError(f"Train file is missing required columns: {missing_train_cols}")
    if missing_test_cols:
        raise ValueError(f"Test file is missing required columns: {missing_test_cols}")
    if len(solution_df) != len(test_df):
        raise ValueError(
            f"solution_format rows ({len(solution_df)}) do not match test rows ({len(test_df)})."
        )
    if "label" not in solution_df.columns:
        raise ValueError("solution_format.csv must contain a 'label' column.")


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


class TextDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.labels is None:
            return self.input_ids[index], self.attention_mask[index]
        return self.input_ids[index], self.attention_mask[index], self.labels[index]


class BalancedBinaryBatchSampler(Sampler[list[int]]):
    """Build batches with equal FALSE/TRUE examples for stable SupCon positives."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError("BalancedBinaryBatchSampler requires an even batch_size.")
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 2
        self.seed = int(seed)
        self.epoch = 0
        self.false_indices = np.where(self.labels == 0)[0]
        self.true_indices = np.where(self.labels == 1)[0]
        if len(self.false_indices) == 0 or len(self.true_indices) == 0:
            raise ValueError("Both classes must be present for balanced contrastive batches.")
        self.num_batches = int(math.ceil(max(len(self.false_indices), len(self.true_indices)) / self.per_class))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        false_pool = rng.permutation(self.false_indices)
        true_pool = rng.permutation(self.true_indices)
        false_needed = self.num_batches * self.per_class
        true_needed = self.num_batches * self.per_class
        if len(false_pool) < false_needed:
            extra = rng.choice(self.false_indices, size=false_needed - len(false_pool), replace=True)
            false_pool = np.concatenate([false_pool, extra])
        if len(true_pool) < true_needed:
            extra = rng.choice(self.true_indices, size=true_needed - len(true_pool), replace=True)
            true_pool = np.concatenate([true_pool, extra])

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.per_class
            end = start + self.per_class
            batch = np.concatenate([false_pool[start:end], true_pool[start:end]])
            rng.shuffle(batch)
            yield batch.tolist()


class RobertaSupConClassifier(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        projection = self.projector(cls_embedding)
        return {
            "logits": logits,
            "embedding": cls_embedding,
            "projection": projection,
        }


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    device = projections.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    features = F.normalize(projections, p=2, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask, device=device)
    logits_mask.fill_diagonal_(0.0)
    positives_mask = mask * logits_mask
    positives_per_row = positives_mask.sum(dim=1)

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    valid_rows = positives_per_row > 0
    if not torch.any(valid_rows):
        return projections.new_tensor(0.0)

    mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / positives_per_row.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid_rows].mean()
    return loss


def compute_metrics(y_true: np.ndarray, prob_true: np.ndarray, threshold: float) -> dict:
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
    for threshold in tqdm(thresholds, desc="Threshold search (validation)", unit="threshold"):
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


def build_optimizer(model: RobertaSupConClassifier, learning_rate: float, adam_eps: float) -> AdamW:
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
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        eps=adam_eps,
    )


def evaluate_model(
    model: RobertaSupConClassifier,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
    has_labels: bool,
) -> tuple[float | None, np.ndarray, np.ndarray | None]:
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            if has_labels:
                input_ids, attention_mask, labels = batch
                labels = labels.to(device)
            else:
                input_ids, attention_mask = batch
                labels = None
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            if has_labels and labels is not None:
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                loss = ce_loss_fn(outputs["logits"], labels)
                total_loss += loss.item() * batch_size
                total_examples += batch_size
    prob_true = np.concatenate(all_probs)
    if has_labels:
        y_true = np.concatenate(all_labels)
        avg_loss = total_loss / max(total_examples, 1)
        return avg_loss, prob_true, y_true
    return None, prob_true, None


def save_best_checkpoint(
    model: RobertaSupConClassifier,
    tokenizer: AutoTokenizer,
    best_epoch: int,
    best_epoch_f1: float,
) -> None:
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    model.encoder.save_pretrained(BEST_MODEL_DIR / "encoder")
    torch.save(
        {
            "epoch": int(best_epoch),
            "best_val_f1": float(best_epoch_f1),
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "projection_dim": PROJECTION_DIM,
            "dropout": DROPOUT,
            "supcon_weight": SUPCON_WEIGHT,
            "supcon_temperature": SUPCON_TEMPERATURE,
        },
        BEST_STATE_DICT_PATH,
    )


def load_best_model(device: torch.device) -> RobertaSupConClassifier:
    model = RobertaSupConClassifier(
        model_name=MODEL_NAME,
        projection_dim=PROJECTION_DIM,
        dropout=DROPOUT,
    )
    checkpoint = torch.load(BEST_STATE_DICT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)
    device = detect_device()
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Loss: CE + {SUPCON_WEIGHT} * SupCon")
    print(f"SupCon temperature: {SUPCON_TEMPERATURE}")

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
    texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    y = encode_labels(train_df["label"], desc="Encoding labels")

    idx = np.arange(len(texts))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")
    print(f"Train class distribution: FALSE={(y_train == 0).sum()} TRUE={(y_train == 1).sum()}")

    print("\n=== Loading tokenizer/model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = RobertaSupConClassifier(
        model_name=MODEL_NAME,
        projection_dim=PROJECTION_DIM,
        dropout=DROPOUT,
    )
    model.to(device)

    print("\n=== Tokenizing ===")
    train_input_ids, train_attention_mask = tokenize_texts(
        train_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing train split",
    )
    val_input_ids, val_attention_mask = tokenize_texts(
        val_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing validation split",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test split",
    )

    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    train_dataset = TextDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TextDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TextDataset(test_input_ids, test_attention_mask)

    pin_memory = device.type == "cuda"
    train_sampler = BalancedBinaryBatchSampler(y_train, batch_size=TRAIN_BATCH_SIZE, seed=SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = build_optimizer(model=model, learning_rate=LEARNING_RATE, adam_eps=ADAM_EPS)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Balanced batches per epoch: {len(train_loader)}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective train batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    ce_loss_fn = nn.CrossEntropyLoss()

    print("\n=== Training ===")
    best_epoch = -1
    best_epoch_f1 = -1.0
    best_epoch_val_probs: np.ndarray | None = None
    best_epoch_val_labels: np.ndarray | None = None
    epoch_history: List[dict] = []

    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_supcon_loss = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - training", unit="batch")
        for step, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            ce_loss = ce_loss_fn(outputs["logits"], labels)
            supcon_loss = supervised_contrastive_loss(
                projections=outputs["projection"],
                labels=labels,
                temperature=SUPCON_TEMPERATURE,
            )
            loss = ce_loss + SUPCON_WEIGHT * supcon_loss

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

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            batch_size = input_ids.size(0)
            running_total_loss += loss_value * batch_size
            running_ce_loss += float(ce_loss.detach().cpu().item()) * batch_size
            running_supcon_loss += float(supcon_loss.detach().cpu().item()) * batch_size
            seen_examples += batch_size

            if (step % GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norm_value = (
                    float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                )
                if not math.isfinite(grad_norm_value):
                    non_finite_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(
                        train_loss=f"{(running_total_loss / max(seen_examples, 1)):.4f}",
                        skipped_loss=non_finite_loss_batches,
                        skipped_grad=non_finite_grad_steps,
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{(running_total_loss / max(seen_examples, 1)):.4f}",
                ce=f"{(running_ce_loss / max(seen_examples, 1)):.4f}",
                supcon=f"{(running_supcon_loss / max(seen_examples, 1)):.4f}",
                skipped_loss=non_finite_loss_batches,
                skipped_grad=non_finite_grad_steps,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_total_loss / max(seen_examples, 1)
        train_ce_loss = running_ce_loss / max(seen_examples, 1)
        train_supcon_loss = running_supcon_loss / max(seen_examples, 1)

        val_loss, val_prob_true, val_true = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{EPOCHS} - validation",
            has_labels=True,
        )
        val_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_loss"] = float(train_loss)
        val_metrics_05["train_ce_loss"] = float(train_ce_loss)
        val_metrics_05["train_supcon_loss"] = float(train_supcon_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        epoch_history.append(val_metrics_05)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | train_ce={train_ce_loss:.6f} "
            f"| train_supcon={train_supcon_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()
            save_best_checkpoint(model, tokenizer, best_epoch, best_epoch_f1)
            print(f"Saved new best model at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

    if best_epoch_val_probs is None or best_epoch_val_labels is None:
        raise RuntimeError("Best validation predictions were not captured.")

    print("\n=== Threshold tuning on validation probabilities ===")
    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        thresholds=THRESHOLD_GRID,
    )
    print(f"Best threshold by validation F1: {best_threshold:.2f}")

    final_val_metrics = compute_metrics(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        threshold=best_threshold,
    )

    val_pred = (best_epoch_val_probs >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": decode_labels(best_epoch_val_labels),
            "pred_label": decode_labels(val_pred),
            "pred_prob_TRUE": best_epoch_val_probs,
        }
    )
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    print("\n=== Validation error analysis ===")
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
    top_k = 200
    fp_df = errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(top_k)
    fn_df = errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(top_k)
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)
    error_df = error_df[["text", "true_label", "pred_label", "pred_prob_TRUE", "error_type", "confidence"]]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    print("\n=== Loading best checkpoint for test inference ===")
    best_model = load_best_model(device)
    _, test_prob_true, _ = evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
        desc="Test inference",
        has_labels=False,
    )

    test_prob_df = pd.DataFrame(
        {
            "text": test_texts,
            "pred_prob_TRUE": test_prob_true,
        }
    )
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {TEST_PROB_PATH}")

    test_pred = (test_prob_true >= best_threshold).astype(int)
    test_pred_labels = decode_labels(test_pred)
    test_distribution = {
        "pred_FALSE": int((test_pred == 0).sum()),
        "pred_TRUE": int((test_pred == 1).sum()),
    }
    print(f"Test prediction distribution: {test_distribution}")

    if solution_df.columns.tolist() == ["label"]:
        submission_df = pd.DataFrame({"label": test_pred_labels})
    else:
        submission_df = solution_df.copy()
        submission_df["label"] = test_pred_labels
    submission_df = submission_df[solution_df.columns.tolist()]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")

    metrics_payload = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
        "config": {
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_train_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
            "loss": "cross_entropy_plus_supervised_contrastive",
            "supcon_weight": SUPCON_WEIGHT,
            "supcon_temperature": SUPCON_TEMPERATURE,
            "projection_dim": PROJECTION_DIM,
            "dropout": DROPOUT,
            "balanced_batch_sampler": {
                "enabled": True,
                "per_class_per_batch": TRAIN_BATCH_SIZE // 2,
                "batches_per_epoch": len(train_loader),
            },
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
        "test_prediction_distribution": test_distribution,
        "paths": {
            "metrics": str(METRICS_PATH),
            "val_predictions": str(VAL_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_DICT_PATH),
        },
        "comparison_target": {
            "submission": "outputs/roberta_base/roberta_base_submission.csv",
            "public_lb": 0.90909091,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes_text = f"""# RoBERTa Supervised Contrastive Notes

## Why This Run Exists

This experiment keeps the successful `roberta-base` setup but adds a light supervised contrastive objective.
The goal is to make same-label samples closer in representation space while keeping normal classification as the primary objective.

## Configuration

- Model: `{MODEL_NAME}`
- Max length: `{MAX_LENGTH}`
- Epochs: `{EPOCHS}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- Effective train batch size: `{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Loss: `CrossEntropyLoss + {SUPCON_WEIGHT} * SupervisedContrastiveLoss`
- SupCon temperature: `{SUPCON_TEMPERATURE}`
- Projection dimension: `{PROJECTION_DIM}`
- Balanced batches: `4 FALSE + 4 TRUE`

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Best checkpoint selected by validation F1 at threshold 0.5
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `{final_val_metrics['threshold']:.2f}`
- Accuracy: `{final_val_metrics['accuracy']:.6f}`
- Precision: `{final_val_metrics['precision']:.6f}`
- Recall: `{final_val_metrics['recall']:.6f}`
- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_val_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{final_val_metrics['prediction_distribution']}`

## Comparison Target

Compare against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.

## Outputs

- `outputs/roberta_supcon/roberta_supcon_metrics.json`
- `outputs/roberta_supcon/roberta_supcon_val_predictions.csv`
- `outputs/roberta_supcon/roberta_supcon_error_analysis.csv`
- `outputs/roberta_supcon/roberta_supcon_test_probabilities.csv`
- `outputs/roberta_supcon/roberta_supcon_submission.csv`
- `outputs/roberta_supcon/roberta_supcon_notes.md`
- `outputs/roberta_supcon/best_model/`
- `outputs/roberta_supcon/roberta_supcon_best_state_dict.pt`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")
    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
