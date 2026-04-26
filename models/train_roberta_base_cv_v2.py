#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

MODEL_NAME = "roberta-base"
SEEDS = [42, 52, 62]
N_SPLITS = 5
MAX_LENGTH = 256
EPOCHS = 1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
THRESHOLD_GRID = np.round(np.arange(0.25, 0.501, 0.01), 2)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_base_cv_v2"
CHECKPOINTS_DIR = OUTPUT_DIR / "fold_checkpoints"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_base_cv_v2_metrics.json"
OOF_PRED_PATH = OUTPUT_DIR / "roberta_base_cv_v2_oof_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "roberta_base_cv_v2_error_analysis.csv"
TEST_PROB_PATH = OUTPUT_DIR / "roberta_base_cv_v2_test_probabilities.csv"
SUBMISSION_PATH = OUTPUT_DIR / "roberta_base_cv_v2_submission.csv"
NOTES_PATH = OUTPUT_DIR / "roberta_base_cv_v2_notes.md"


def set_seeds(seed: int) -> None:
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
    return np.asarray(encoded, dtype=np.int64)



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

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    return input_ids, attention_mask



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

        tie_break_current = abs(float(threshold) - 0.32)
        tie_break_best = abs(best_threshold - 0.32)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and tie_break_current < tie_break_best):
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, records



def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
    has_labels: bool,
) -> tuple[float | None, np.ndarray, np.ndarray | None]:
    model.eval()
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())

            if has_labels and labels is not None:
                all_labels.append(labels.detach().cpu().numpy())
                batch_size = input_ids.size(0)
                total_loss += outputs.loss.item() * batch_size
                total_examples += batch_size

    prob_true = np.concatenate(all_probs)

    if has_labels:
        y_true = np.concatenate(all_labels)
        avg_loss = total_loss / max(total_examples, 1)
        return avg_loss, prob_true, y_true

    return None, prob_true, None



def build_optimizer(model: AutoModelForSequenceClassification) -> AdamW:
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

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPS)



def make_train_loader(train_dataset: TensorDataset, seed: int, fold_idx: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed * 1000 + fold_idx)
    return DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )



def get_seed_tag(seed: int) -> str:
    return f"seed_{seed}"



def get_fold_dirs(seed: int, fold_idx: int) -> tuple[Path, Path, Path]:
    seed_dir = CHECKPOINTS_DIR / get_seed_tag(seed)
    fold_dir = seed_dir / f"fold_{fold_idx}"
    best_model_dir = fold_dir / "best_model"
    state_dict_path = fold_dir / "roberta_base_cv_v2_best_state_dict.pt"
    return fold_dir, best_model_dir, state_dict_path



def make_seed_oof_path(seed: int) -> Path:
    return OUTPUT_DIR / f"roberta_base_cv_v2_{get_seed_tag(seed)}_oof_predictions.csv"



def make_seed_test_path(seed: int) -> Path:
    return OUTPUT_DIR / f"roberta_base_cv_v2_{get_seed_tag(seed)}_test_probabilities.csv"



def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    pin_memory = device.type == "cuda"

    print("\n=== Startup Configuration ===")
    print(f"device: {device}")
    print(f"model: {MODEL_NAME}")
    print(f"seeds: {SEEDS}")
    print(f"folds: {N_SPLITS}")
    print(f"epochs per fold: {EPOCHS}")
    print(f"train batch size: {TRAIN_BATCH_SIZE}")
    print(f"gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"learning rate: {LEARNING_RATE}")
    print("ensemble-ready outputs: per-seed OOF and test probabilities will be saved")

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

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("\n=== Tokenizing full train/test once ===")
    full_train_input_ids, full_train_attention_mask = tokenize_texts(
        texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing full train",
    )
    test_input_ids, test_attention_mask = tokenize_texts(
        test_texts,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test",
    )

    full_train_labels = torch.tensor(y, dtype=torch.long)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    all_seed_oof = np.zeros((len(texts), len(SEEDS)), dtype=np.float64)
    all_seed_test = np.zeros((len(test_texts), len(SEEDS)), dtype=np.float64)
    seed_results: List[dict] = []

    print("\n=== Multi-Seed 5-Fold CV Training ===")
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n######## Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) ########")
        set_seeds(seed)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        seed_oof = np.zeros(len(texts), dtype=np.float64)
        seed_oof_seen_mask = np.zeros(len(texts), dtype=bool)
        seed_fold_test_prob = np.zeros((N_SPLITS, len(test_texts)), dtype=np.float64)
        fold_results: List[dict] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(texts)), y), start=1):
            print(f"\n===== Seed {seed} Fold {fold_idx}/{N_SPLITS} =====")

            fold_dir, best_model_dir, state_dict_path = get_fold_dirs(seed, fold_idx)
            fold_dir.mkdir(parents=True, exist_ok=True)
            best_model_dir.mkdir(parents=True, exist_ok=True)

            train_dataset = TensorDataset(
                full_train_input_ids[tr_idx],
                full_train_attention_mask[tr_idx],
                full_train_labels[tr_idx],
            )
            val_dataset = TensorDataset(
                full_train_input_ids[val_idx],
                full_train_attention_mask[val_idx],
                full_train_labels[val_idx],
            )

            train_loader = make_train_loader(train_dataset, seed=seed, fold_idx=fold_idx, pin_memory=pin_memory)
            val_loader = DataLoader(
                val_dataset,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            model.to(device)

            optimizer = build_optimizer(model)
            num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
            total_training_steps = num_update_steps_per_epoch * EPOCHS
            warmup_steps = int(0.1 * total_training_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            seen_examples = 0
            non_finite_loss_batches = 0
            non_finite_grad_steps = 0

            progress = tqdm(
                train_loader,
                desc=f"Seed {seed} Fold {fold_idx} Epoch 1/{EPOCHS} - training",
                unit="batch",
            )
            for step, batch in enumerate(progress, start=1):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
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
                running_loss += loss_value * batch_size
                seen_examples += batch_size

                if (step % GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

                avg_loss = running_loss / max(seen_examples, 1)
                progress.set_postfix(
                    train_loss=f"{avg_loss:.4f}",
                    skipped_loss=non_finite_loss_batches,
                    skipped_grad=non_finite_grad_steps,
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            train_loss = running_loss / max(seen_examples, 1)

            val_loss, val_prob_true, val_true = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                desc=f"Seed {seed} Fold {fold_idx} - validation",
                has_labels=True,
            )

            fold_metrics_05 = compute_metrics(val_true, val_prob_true, threshold=0.5)
            fold_best_threshold, fold_threshold_table = tune_threshold_for_f1(
                y_true=val_true,
                prob_true=val_prob_true,
                thresholds=THRESHOLD_GRID,
            )
            fold_metrics_tuned = compute_metrics(val_true, val_prob_true, threshold=fold_best_threshold)

            print(
                f"Seed {seed} Fold {fold_idx} | train_loss={train_loss:.6f} | "
                f"val_loss={float(val_loss):.6f} | val_f1@0.5={fold_metrics_05['f1']:.6f} | "
                f"val_auc={fold_metrics_05['roc_auc']:.6f} | val_best_thr={fold_best_threshold:.2f} | "
                f"val_f1@best_thr={fold_metrics_tuned['f1']:.6f}"
            )

            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            torch.save(
                {
                    "seed": seed,
                    "fold": fold_idx,
                    "epoch": 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_f1_threshold_0_5": float(fold_metrics_05["f1"]),
                    "val_f1_best_threshold": float(fold_metrics_tuned["f1"]),
                    "val_best_threshold": float(fold_best_threshold),
                    "model_state_dict": model.state_dict(),
                },
                state_dict_path,
            )

            _, test_prob_true, _ = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                desc=f"Seed {seed} Fold {fold_idx} - test inference",
                has_labels=False,
            )

            seed_oof[val_idx] = val_prob_true
            seed_oof_seen_mask[val_idx] = True
            seed_fold_test_prob[fold_idx - 1] = test_prob_true

            fold_result = {
                "seed": int(seed),
                "fold": int(fold_idx),
                "epoch": 1,
                "num_train_rows": int(len(tr_idx)),
                "num_val_rows": int(len(val_idx)),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "metrics_threshold_0_5": fold_metrics_05,
                "best_threshold": float(fold_best_threshold),
                "metrics_best_threshold": fold_metrics_tuned,
                "threshold_table": fold_threshold_table,
                "skipped_non_finite_loss_batches": int(non_finite_loss_batches),
                "skipped_non_finite_grad_steps": int(non_finite_grad_steps),
                "best_model_dir": str(best_model_dir),
                "best_state_dict": str(state_dict_path),
            }
            fold_results.append(fold_result)

        if not np.all(seed_oof_seen_mask):
            missing = int((~seed_oof_seen_mask).sum())
            raise RuntimeError(f"Seed {seed}: missing OOF predictions for {missing} rows.")

        seed_test_mean = seed_fold_test_prob.mean(axis=0)
        all_seed_oof[:, seed_idx] = seed_oof
        all_seed_test[:, seed_idx] = seed_test_mean

        seed_best_threshold, seed_threshold_table = tune_threshold_for_f1(
            y_true=y,
            prob_true=seed_oof,
            thresholds=THRESHOLD_GRID,
        )
        seed_oof_metrics = compute_metrics(y_true=y, prob_true=seed_oof, threshold=seed_best_threshold)

        seed_oof_df = pd.DataFrame(
            {
                "text": texts,
                "true_label": decode_labels(y),
                "oof_prob_TRUE": seed_oof,
                "oof_pred_label": decode_labels((seed_oof >= seed_best_threshold).astype(int)),
            }
        )
        seed_oof_path = make_seed_oof_path(seed)
        seed_oof_df.to_csv(seed_oof_path, index=False)

        seed_test_df = pd.DataFrame(
            {
                "text": test_texts,
                "pred_prob_TRUE": seed_test_mean,
            }
        )
        seed_test_path = make_seed_test_path(seed)
        seed_test_df.to_csv(seed_test_path, index=False)

        seed_results.append(
            {
                "seed": int(seed),
                "folds": fold_results,
                "threshold_tuning": {
                    "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
                    "step": 0.01,
                    "selection_objective": "max_seed_oof_f1",
                    "selected_threshold": float(seed_best_threshold),
                    "table": seed_threshold_table,
                },
                "oof_final": seed_oof_metrics,
                "test_prediction_distribution": {
                    "pred_FALSE": int((seed_test_mean < seed_best_threshold).sum()),
                    "pred_TRUE": int((seed_test_mean >= seed_best_threshold).sum()),
                },
                "paths": {
                    "oof_predictions": str(seed_oof_path),
                    "test_probabilities": str(seed_test_path),
                },
            }
        )
        print(f"Saved seed OOF predictions: {seed_oof_path}")
        print(f"Saved seed test probabilities: {seed_test_path}")

    avg_oof_prob = all_seed_oof.mean(axis=1)
    avg_test_prob = all_seed_test.mean(axis=1)

    print("\n=== Threshold tuning on averaged OOF probabilities ===")
    best_threshold, threshold_table = tune_threshold_for_f1(
        y_true=y,
        prob_true=avg_oof_prob,
        thresholds=THRESHOLD_GRID,
    )
    print(f"Best threshold by averaged OOF F1: {best_threshold:.2f}")

    final_oof_metrics = compute_metrics(y_true=y, prob_true=avg_oof_prob, threshold=best_threshold)
    avg_oof_pred = (avg_oof_prob >= best_threshold).astype(int)

    oof_payload = {
        "text": texts,
        "true_label": decode_labels(y),
        "oof_prob_TRUE_mean": avg_oof_prob,
        "oof_pred_label": decode_labels(avg_oof_pred),
    }
    for seed_idx, seed in enumerate(SEEDS):
        oof_payload[f"oof_prob_TRUE_{get_seed_tag(seed)}"] = all_seed_oof[:, seed_idx]
    oof_pred_df = pd.DataFrame(oof_payload)
    oof_pred_df.to_csv(OOF_PRED_PATH, index=False)
    print(f"Saved averaged OOF predictions: {OOF_PRED_PATH}")

    print("\n=== OOF error analysis ===")
    errors = oof_pred_df[oof_pred_df["true_label"] != oof_pred_df["oof_pred_label"]].copy()
    errors["error_type"] = np.where(
        (errors["true_label"] == "FALSE") & (errors["oof_pred_label"] == "TRUE"),
        "false_positive",
        "false_negative",
    )
    errors["confidence"] = np.where(
        errors["error_type"] == "false_positive",
        errors["oof_prob_TRUE_mean"],
        1.0 - errors["oof_prob_TRUE_mean"],
    )
    top_k = 250
    fp_df = errors[errors["error_type"] == "false_positive"].sort_values("confidence", ascending=False).head(top_k)
    fn_df = errors[errors["error_type"] == "false_negative"].sort_values("confidence", ascending=False).head(top_k)
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)
    error_keep_cols = [
        "text",
        "true_label",
        "oof_pred_label",
        "oof_prob_TRUE_mean",
        "error_type",
        "confidence",
    ] + [f"oof_prob_TRUE_{get_seed_tag(seed)}" for seed in SEEDS]
    error_df = error_df[error_keep_cols]
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")

    test_payload = {
        "text": test_texts,
        "pred_prob_TRUE": avg_test_prob,
    }
    for seed_idx, seed in enumerate(SEEDS):
        test_payload[f"pred_prob_TRUE_{get_seed_tag(seed)}"] = all_seed_test[:, seed_idx]
    test_prob_df = pd.DataFrame(test_payload)
    test_prob_df.to_csv(TEST_PROB_PATH, index=False)
    print(f"Saved averaged test probabilities: {TEST_PROB_PATH}")

    test_pred = (avg_test_prob >= best_threshold).astype(int)
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
        "device": str(device),
        "config": {
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "threshold_grid_min": float(THRESHOLD_GRID.min()),
            "threshold_grid_max": float(THRESHOLD_GRID.max()),
            "threshold_grid_step": 0.01,
        },
        "seed_runs": seed_results,
        "threshold_tuning": {
            "search_range": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_mean_oof_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "oof_final": final_oof_metrics,
        "test_prediction_distribution": test_distribution,
        "paths": {
            "metrics": str(METRICS_PATH),
            "oof_predictions": str(OOF_PRED_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "test_probabilities": str(TEST_PROB_PATH),
            "submission": str(SUBMISSION_PATH),
            "notes": str(NOTES_PATH),
            "fold_checkpoints_dir": str(CHECKPOINTS_DIR),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_PATH}")

    notes_text = f"""# RoBERTa-base CV V2 Notes

## Run Configuration

- Model: `{MODEL_NAME}`
- Device: `{device}`
- Seeds: `{SEEDS}`
- Folds: `{N_SPLITS}` (StratifiedKFold)
- Epochs per fold: `{EPOCHS}`
- Max length: `{MAX_LENGTH}`
- Train batch size: `{TRAIN_BATCH_SIZE}`
- Eval batch size: `{EVAL_BATCH_SIZE}`
- Gradient accumulation: `{GRADIENT_ACCUMULATION_STEPS}`
- Learning rate: `{LEARNING_RATE}`
- Weight decay: `{WEIGHT_DECAY}`

## Design Rationale

- Keeps `roberta-base` as the main model because it is the strongest single-model baseline so far.
- Uses 5 folds instead of a single split for more stable OOF estimates.
- Uses 1 epoch per fold to stay close to the observed sweet spot where RoBERTa performed best before overfitting.
- Uses 3 seeds to reduce variance and produce ensemble-ready probabilities.
- Saves per-seed OOF and test probabilities so a later stacker/blender can use them directly.

## Thresholding

- Threshold tuned on averaged OOF probabilities.
- Search range: `{THRESHOLD_GRID.min():.2f}` to `{THRESHOLD_GRID.max():.2f}`
- Selected threshold: `{best_threshold:.2f}`

## Final Averaged OOF Metrics

- Accuracy: `{final_oof_metrics['accuracy']:.6f}`
- Precision: `{final_oof_metrics['precision']:.6f}`
- Recall: `{final_oof_metrics['recall']:.6f}`
- F1: `{final_oof_metrics['f1']:.6f}`
- ROC AUC: `{final_oof_metrics['roc_auc']:.6f}`
- Confusion matrix [[TN, FP], [FN, TP]]: `{final_oof_metrics['confusion_matrix']['matrix']}`
- Prediction distribution: `{final_oof_metrics['prediction_distribution']}`

## Outputs

- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_metrics.json`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_oof_predictions.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_error_analysis.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_test_probabilities.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_submission.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_notes.md`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_seed_*_oof_predictions.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_seed_*_test_probabilities.csv`
- `outputs/roberta_base_cv_v2/fold_checkpoints/`
"""
    NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {NOTES_PATH}")

    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
