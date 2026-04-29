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
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from roberta_dapt_pipeline_common import (
    detect_device,
    load_jsonl,
    normalize_texts,
    set_seeds,
    validate_competition_inputs,
)

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm torch transformers accelerate"

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
EPOCHS = 2
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
MLM_PROBABILITY = 0.15
EVAL_SPLIT = 0.05
TOKENIZE_BATCH_SIZE = 256

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "roberta_dapt_mlm"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
LAST_MODEL_DIR = OUTPUT_DIR / "last_model"
BEST_STATE_PATH = OUTPUT_DIR / "roberta_dapt_mlm_best_state_dict.pt"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

METRICS_PATH = OUTPUT_DIR / "roberta_dapt_mlm_metrics.json"
NOTES_PATH = OUTPUT_DIR / "roberta_dapt_mlm_notes.md"
SPLIT_META_PATH = OUTPUT_DIR / "roberta_dapt_mlm_split_meta.json"


class MLMDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining for RoBERTa.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS)
    parser.add_argument("--mlm-probability", type=float, default=MLM_PROBABILITY)
    parser.add_argument("--eval-split", type=float, default=EVAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def tokenize_mlm_texts(
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []
    all_special_token_masks = []

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
        all_special_token_masks.append(encoded["special_tokens_mask"])

    return (
        torch.cat(all_input_ids, dim=0),
        torch.cat(all_attention_masks, dim=0),
        torch.cat(all_special_token_masks, dim=0),
    )


def evaluate_mlm(
    model: AutoModelForMaskedLM,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = int(batch["input_ids"].shape[0])
            total_loss += float(outputs.loss.detach().cpu().item()) * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def build_optimizer(model: AutoModelForMaskedLM, learning_rate: float, weight_decay: float, adam_eps: float) -> AdamW:
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


def write_notes(metrics: dict) -> None:
    notes = f"""# RoBERTa DAPT MLM Notes

## Purpose

This script performs domain-adaptive masked language model pretraining on the competition text only.
It uses unlabeled `train + test` text and does not use any external data.

## Configuration

- Base model: `{metrics['model_name']}`
- Max length: `{metrics['config']['max_length']}`
- Epochs: `{metrics['config']['epochs']}`
- Train batch size: `{metrics['config']['train_batch_size']}`
- Eval batch size: `{metrics['config']['eval_batch_size']}`
- Gradient accumulation: `{metrics['config']['gradient_accumulation_steps']}`
- Learning rate: `{metrics['config']['learning_rate']}`
- MLM probability: `{metrics['config']['mlm_probability']}`
- Eval split: `{metrics['config']['eval_split']}`

## Data

- Train text rows: `{metrics['data']['train_rows']}`
- Test text rows: `{metrics['data']['test_rows']}`
- Combined unlabeled rows: `{metrics['data']['combined_rows']}`
- MLM train rows: `{metrics['data']['mlm_train_rows']}`
- MLM eval rows: `{metrics['data']['mlm_eval_rows']}`

## Result

- Best epoch: `{metrics['best_epoch']['epoch']}`
- Best eval loss: `{metrics['best_epoch']['eval_loss']:.6f}`

## Outputs

- `outputs/roberta_dapt_mlm/best_model/`
- `outputs/roberta_dapt_mlm/last_model/`
- `outputs/roberta_dapt_mlm/roberta_dapt_mlm_metrics.json`
- `outputs/roberta_dapt_mlm/roberta_dapt_mlm_notes.md`
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== RoBERTa DAPT MLM Pretraining ===")

    set_seeds(args.seed)
    device = detect_device()
    print(f"device: {device}")
    print(f"model: {args.model_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LAST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading competition data ===")
    train_df = load_jsonl(TRAIN_PATH, "Loading train JSONL")
    test_df = load_jsonl(TEST_PATH, "Loading test JSONL")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    validate_competition_inputs(train_df, test_df, solution_df)

    train_texts = normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = normalize_texts(test_df["text"], desc="Normalizing test text")
    combined_texts = train_texts + test_texts

    indices = np.arange(len(combined_texts))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=args.eval_split,
        random_state=args.seed,
        shuffle=True,
    )
    mlm_train_texts = [combined_texts[idx] for idx in train_idx]
    mlm_eval_texts = [combined_texts[idx] for idx in eval_idx]

    split_meta = {
        "seed": args.seed,
        "train_indices": train_idx.tolist(),
        "eval_indices": eval_idx.tolist(),
        "train_rows": len(mlm_train_texts),
        "eval_rows": len(mlm_eval_texts),
    }
    SPLIT_META_PATH.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    print("\n=== Loading tokenizer/model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    print("\n=== Tokenizing MLM corpora ===")
    train_input_ids, train_attention_mask, train_special_tokens = tokenize_mlm_texts(
        mlm_train_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=TOKENIZE_BATCH_SIZE,
        desc="Tokenizing MLM train",
    )
    eval_input_ids, eval_attention_mask, eval_special_tokens = tokenize_mlm_texts(
        mlm_eval_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=TOKENIZE_BATCH_SIZE,
        desc="Tokenizing MLM eval",
    )

    train_dataset = MLMDataset(train_input_ids, train_attention_mask, train_special_tokens)
    eval_dataset = MLMDataset(eval_input_ids, eval_attention_mask, eval_special_tokens)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_eps,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print("\n=== MLM training ===")
    epoch_history = []
    best_epoch = -1
    best_eval_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_examples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - MLM train", unit="batch")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_value = float(loss.detach().cpu().item())

            if not math.isfinite(loss_value):
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / args.grad_accum).backward()

            batch_size = int(batch["input_ids"].shape[0])
            running_loss += loss_value * batch_size
            seen_examples += batch_size

            if (step % args.grad_accum == 0) or (step == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=f"{(running_loss / max(seen_examples, 1)):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(seen_examples, 1)
        eval_loss = evaluate_mlm(model, eval_loader, device=device, desc=f"Epoch {epoch}/{args.epochs} - MLM eval")
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
        }
        epoch_history.append(epoch_record)

        print(f"Epoch {epoch} | train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = float(eval_loss)
            best_epoch = int(epoch)
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            torch.save(
                {
                    "epoch": best_epoch,
                    "eval_loss": best_eval_loss,
                    "model_state_dict": model.state_dict(),
                },
                BEST_STATE_PATH,
            )
            print(f"Saved new best MLM checkpoint at epoch {best_epoch}.")

    model.save_pretrained(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)

    metrics = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "config": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "mlm_probability": args.mlm_probability,
            "eval_split": args.eval_split,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
        },
        "data": {
            "train_rows": len(train_texts),
            "test_rows": len(test_texts),
            "combined_rows": len(combined_texts),
            "mlm_train_rows": len(mlm_train_texts),
            "mlm_eval_rows": len(mlm_eval_texts),
        },
        "epoch_history": epoch_history,
        "best_epoch": {
            "epoch": best_epoch,
            "eval_loss": best_eval_loss,
        },
        "paths": {
            "metrics": str(METRICS_PATH),
            "notes": str(NOTES_PATH),
            "split_meta": str(SPLIT_META_PATH),
            "best_model_dir": str(BEST_MODEL_DIR),
            "last_model_dir": str(LAST_MODEL_DIR),
            "best_state_dict": str(BEST_STATE_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_notes(metrics)
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved notes: {NOTES_PATH}")


if __name__ == "__main__":
    main()
