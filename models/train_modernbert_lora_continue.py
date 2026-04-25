#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm torch transformers accelerate peft
# Safety: This script is generated but not executed by Codex. User should run it manually.

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import train_modernbert_lora as base

INSTALL_CMD = base.INSTALL_CMD


def load_previous_best_validation_artifacts() -> tuple[np.ndarray, np.ndarray]:
    if base.TRAINING_STATE_PATH.exists():
        state = torch.load(base.TRAINING_STATE_PATH, map_location="cpu")
        probs = state.get("best_epoch_val_probs")
        labels = state.get("best_epoch_val_labels")
        if probs is not None and labels is not None:
            return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)

    if not base.VAL_PRED_PATH.exists():
        raise FileNotFoundError(
            f"Could not find prior validation predictions at {base.VAL_PRED_PATH}."
        )

    val_df = pd.read_csv(base.VAL_PRED_PATH)
    required_cols = {"pred_prob_TRUE", "true_label"}
    if missing_cols := sorted(required_cols - set(val_df.columns)):
        raise ValueError(f"Validation prediction file missing required columns: {missing_cols}")

    probs = pd.to_numeric(val_df["pred_prob_TRUE"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(probs).any():
        raise ValueError("Validation prediction file contains invalid probabilities.")
    labels = val_df["true_label"].astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0}).to_numpy()
    if pd.isna(labels).any():
        raise ValueError("Validation prediction file contains invalid labels.")
    return probs, labels.astype(np.int64)


def load_resume_state() -> dict | None:
    if base.TRAINING_STATE_PATH.exists() and base.LAST_MODEL_DIR.exists():
        return torch.load(base.TRAINING_STATE_PATH, map_location="cpu")
    return None


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base.LAST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    base.set_seeds(base.SEED)
    device = base.detect_device()
    print("\n=== Startup Configuration ===")
    print(f"torch version: {torch.__version__}")
    print(f"transformers version: {transformers.__version__}")
    print(f"device: {device}")
    print(f"model name: {base.MODEL_NAME}")
    print(f"target total epochs: {base.EPOCHS}")
    print("precision mode: fp32 (no AMP/fp16)")

    print("\n=== Loading data ===")
    train_df = base.load_jsonl(base.TRAIN_PATH, desc="Loading train JSONL")
    test_df = base.load_jsonl(base.TEST_PATH, desc="Loading test JSONL")
    solution_df = pd.read_csv(base.SOLUTION_FORMAT_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution format shape: {solution_df.shape}")

    base.validate_inputs(train_df, test_df, solution_df)
    print("Input schema checks passed.")
    print(f"Expected submission columns: {solution_df.columns.tolist()}")

    print("\n=== Preparing text and labels ===")
    texts = base.normalize_texts(train_df["text"], desc="Normalizing train text")
    test_texts = base.normalize_texts(test_df["text"], desc="Normalizing test text")
    y = base.encode_labels(train_df["label"], desc="Encoding labels")

    default_idx = np.arange(len(texts))
    default_train_idx, default_val_idx = train_test_split(
        default_idx,
        test_size=0.2,
        random_state=base.SEED,
        stratify=y,
    )

    resume_state = load_resume_state()
    exact_resume = resume_state is not None

    if exact_resume:
        train_idx = np.asarray(resume_state["train_indices"], dtype=np.int64)
        val_idx = np.asarray(resume_state["val_indices"], dtype=np.int64)
        completed_epochs = int(resume_state["completed_epoch"])
        resume_mode = "exact_resume"
        print("\n=== Resume Mode ===")
        print("Using exact resume from saved last checkpoint + optimizer/scheduler state.")
    else:
        train_idx = default_train_idx
        val_idx = default_val_idx
        if not base.METRICS_PATH.exists():
            raise FileNotFoundError(
                f"Could not find metrics file for warm restart: {base.METRICS_PATH}"
            )
        prior_metrics = json.loads(base.METRICS_PATH.read_text(encoding="utf-8"))
        completed_epochs = int(max(row["epoch"] for row in prior_metrics["epoch_metrics_threshold_0_5"]))
        resume_mode = "warm_restart_from_best_adapter"
        print("\n=== Resume Mode ===")
        print("Optimizer/scheduler state not found for this run.")
        print("Warm-restarting from saved best adapter weights with a fresh optimizer/scheduler.")

    start_epoch = completed_epochs + 1
    if start_epoch > base.EPOCHS:
        raise RuntimeError(
            f"No remaining epochs to run. Completed epochs = {completed_epochs}, target total epochs = {base.EPOCHS}."
        )

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Train split size: {len(train_texts)}")
    print(f"Validation split size: {len(val_texts)}")
    print(f"Completed epochs before resume: {completed_epochs}")
    print(f"Resume start epoch: {start_epoch}")

    tokenizer_source = base.LAST_MODEL_DIR if exact_resume else base.BEST_MODEL_DIR
    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    print("\n=== Tokenizing ===")
    train_input_ids, train_attention_mask = base.tokenize_texts(
        train_texts,
        tokenizer,
        max_length=base.MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing train split",
    )
    val_input_ids, val_attention_mask = base.tokenize_texts(
        val_texts,
        tokenizer,
        max_length=base.MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing validation split",
    )
    test_input_ids, test_attention_mask = base.tokenize_texts(
        test_texts,
        tokenizer,
        max_length=base.MAX_LENGTH,
        batch_size=256,
        desc="Tokenizing test split",
    )

    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        val_dataset,
        batch_size=base.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=base.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Loading model ===")
    if exact_resume:
        model = base.load_lora_model_from_dir(base.LAST_MODEL_DIR, device)
    else:
        model = base.load_best_lora_model(device)
    trainable_params, total_params = base.count_trainable_parameters(model)
    trainable_pct = 100.0 * trainable_params / max(total_params, 1)
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_pct:.4f}%)"
    )

    print("\n=== Optimizer/Scheduler setup ===")
    optimizer = base.build_optimizer(model=model, learning_rate=base.LEARNING_RATE, adam_eps=base.ADAM_EPS)
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / base.TRAIN_BATCH_SIZE / base.GRADIENT_ACCUMULATION_STEPS)

    if exact_resume:
        total_training_steps = int(resume_state["total_training_steps"])
        warmup_steps = int(resume_state["warmup_steps"])
    else:
        remaining_epochs = base.EPOCHS - completed_epochs
        total_training_steps = num_update_steps_per_epoch * remaining_epochs
        warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if exact_resume:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])

    if exact_resume:
        epoch_history = list(resume_state["epoch_history"])
        best_epoch = int(resume_state["best_epoch"])
        best_epoch_f1 = float(resume_state["best_epoch_f1"])
        best_epoch_val_probs = np.asarray(resume_state["best_epoch_val_probs"], dtype=np.float64)
        best_epoch_val_labels = np.asarray(resume_state["best_epoch_val_labels"], dtype=np.int64)
    else:
        prior_metrics = json.loads(base.METRICS_PATH.read_text(encoding="utf-8"))
        epoch_history = list(prior_metrics["epoch_metrics_threshold_0_5"])
        best_epoch = int(prior_metrics["best_epoch"]["epoch"])
        best_epoch_f1 = float(prior_metrics["best_epoch"]["val_f1_threshold_0_5"])
        best_epoch_val_probs, best_epoch_val_labels = load_previous_best_validation_artifacts()

    print("\n=== Continued Training ===")
    for epoch in range(start_epoch, base.EPOCHS + 1):
        train_loader = base.make_epoch_train_loader(
            train_dataset=train_dataset,
            batch_size=base.TRAIN_BATCH_SIZE,
            pin_memory=pin_memory,
            epoch=epoch,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        seen_examples = 0
        non_finite_loss_batches = 0
        non_finite_grad_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{base.EPOCHS} - training", unit="batch")
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

            (loss / base.GRADIENT_ACCUMULATION_STEPS).backward()

            batch_size = input_ids.size(0)
            running_loss += loss_value * batch_size
            seen_examples += batch_size

            if (step % base.GRADIENT_ACCUMULATION_STEPS == 0) or (step == len(train_loader)):
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

        val_loss, val_prob_true, val_true = base.evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{base.EPOCHS} - validation",
            has_labels=True,
        )

        val_metrics_05 = base.compute_metrics(val_true, val_prob_true, threshold=0.5)
        val_metrics_05["epoch"] = int(epoch)
        val_metrics_05["train_loss"] = float(train_loss)
        val_metrics_05["val_loss"] = float(val_loss if val_loss is not None else np.nan)
        val_metrics_05["skipped_non_finite_loss_batches"] = int(non_finite_loss_batches)
        val_metrics_05["skipped_non_finite_grad_steps"] = int(non_finite_grad_steps)
        epoch_history.append(val_metrics_05)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_metrics_05['val_loss']:.6f} "
            f"| val_f1@0.5={val_metrics_05['f1']:.6f} | val_auc={val_metrics_05['roc_auc']:.6f} "
            f"| skipped_non_finite_loss_batches={non_finite_loss_batches} "
            f"| skipped_non_finite_grad_steps={non_finite_grad_steps}"
        )

        if val_metrics_05["f1"] > best_epoch_f1:
            best_epoch_f1 = float(val_metrics_05["f1"])
            best_epoch = int(epoch)
            best_epoch_val_probs = val_prob_true.copy()
            best_epoch_val_labels = val_true.copy()

            model.save_pretrained(base.BEST_MODEL_DIR)
            tokenizer.save_pretrained(base.BEST_MODEL_DIR)
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_f1": best_epoch_f1,
                    "adapter_state_dict": base.extract_adapter_state_dict(model),
                },
                base.BEST_STATE_DICT_PATH,
            )
            print(f"Saved new best model at epoch {best_epoch} with val_f1@0.5={best_epoch_f1:.6f}")

        base.save_last_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epoch=epoch,
            epoch_history=epoch_history,
            best_epoch=best_epoch,
            best_epoch_f1=best_epoch_f1,
            best_epoch_val_probs=best_epoch_val_probs,
            best_epoch_val_labels=best_epoch_val_labels,
            train_idx=train_idx,
            val_idx=val_idx,
            warmup_steps=warmup_steps,
            total_training_steps=total_training_steps,
        )

    print("\n=== Threshold tuning on validation probabilities ===")
    best_threshold, threshold_table = base.tune_threshold_for_f1(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        thresholds=base.THRESHOLD_GRID,
    )
    print(f"Best threshold by validation F1: {best_threshold:.2f}")

    final_val_metrics = base.compute_metrics(
        y_true=best_epoch_val_labels,
        prob_true=best_epoch_val_probs,
        threshold=best_threshold,
    )

    val_pred = (best_epoch_val_probs >= best_threshold).astype(int)
    val_pred_df = pd.DataFrame(
        {
            "text": val_texts,
            "true_label": base.decode_labels(best_epoch_val_labels),
            "pred_label": base.decode_labels(val_pred),
            "pred_prob_TRUE": best_epoch_val_probs,
        }
    )
    val_pred_df.to_csv(base.VAL_PRED_PATH, index=False)
    print(f"Saved validation predictions: {base.VAL_PRED_PATH}")

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
    error_df.to_csv(base.ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis: {base.ERROR_ANALYSIS_PATH}")

    print("\n=== Loading best checkpoint for test inference ===")
    best_model = base.load_best_lora_model(device)
    _, test_prob_true, _ = base.evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
        desc="Test inference",
        has_labels=False,
    )

    test_prob_df = pd.DataFrame({"text": test_texts, "pred_prob_TRUE": test_prob_true})
    test_prob_df.to_csv(base.TEST_PROB_PATH, index=False)
    print(f"Saved test probabilities: {base.TEST_PROB_PATH}")

    test_pred = (test_prob_true >= best_threshold).astype(int)
    test_pred_labels = base.decode_labels(test_pred)
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
    submission_df.to_csv(base.SUBMISSION_PATH, index=False)
    print(f"Saved submission: {base.SUBMISSION_PATH}")

    metrics_payload = {
        "model": base.MODEL_NAME,
        "mode": "lora_resume_training",
        "resume_mode": resume_mode,
        "seed": base.SEED,
        "device": str(device),
        "config": {
            "max_length": base.MAX_LENGTH,
            "epochs": base.EPOCHS,
            "train_batch_size": base.TRAIN_BATCH_SIZE,
            "effective_train_batch_size": base.TRAIN_BATCH_SIZE,
            "eval_batch_size": base.EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": base.GRADIENT_ACCUMULATION_STEPS,
            "effective_gradient_accumulation_steps": base.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "adam_eps": base.ADAM_EPS,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "warmup_steps": warmup_steps,
            "total_training_steps": total_training_steps,
            "lora_r": base.LORA_R,
            "lora_alpha": base.LORA_ALPHA,
            "lora_dropout": base.LORA_DROPOUT,
            "lora_target_modules": base.LORA_TARGET_MODULES,
            "lora_modules_to_save": base.LORA_MODULES_TO_SAVE,
            "trainable_parameters": int(trainable_params),
            "total_parameters": int(total_params),
            "trainable_parameter_percent": float(trainable_pct),
            "fp16_amp_used": False,
        },
        "resume": {
            "completed_epochs_before_resume": int(completed_epochs),
            "resume_start_epoch": int(start_epoch),
            "target_total_epochs": int(base.EPOCHS),
        },
        "split": {
            "train_rows": int(len(train_texts)),
            "val_rows": int(len(val_texts)),
            "test_rows": int(len(test_texts)),
            "stratified": True,
            "random_state": base.SEED,
        },
        "epoch_metrics_threshold_0_5": epoch_history,
        "best_epoch": {
            "epoch": int(best_epoch),
            "val_f1_threshold_0_5": float(best_epoch_f1),
            "checkpoint_dir": str(base.BEST_MODEL_DIR),
            "state_dict_path": str(base.BEST_STATE_DICT_PATH),
        },
        "resume_state": {
            "last_model_dir": str(base.LAST_MODEL_DIR),
            "training_state_path": str(base.TRAINING_STATE_PATH),
        },
        "threshold_tuning": {
            "search_range": [float(base.THRESHOLD_GRID.min()), float(base.THRESHOLD_GRID.max())],
            "step": 0.01,
            "selection_objective": "max_validation_f1",
            "selected_threshold": float(best_threshold),
            "table": threshold_table,
        },
        "validation_final": final_val_metrics,
        "test_prediction_distribution": test_distribution,
        "paths": {
            "metrics": str(base.METRICS_PATH),
            "val_predictions": str(base.VAL_PRED_PATH),
            "error_analysis": str(base.ERROR_ANALYSIS_PATH),
            "test_probabilities": str(base.TEST_PROB_PATH),
            "submission": str(base.SUBMISSION_PATH),
            "notes": str(base.NOTES_PATH),
            "best_model_dir": str(base.BEST_MODEL_DIR),
            "best_state_dict": str(base.BEST_STATE_DICT_PATH),
            "last_model_dir": str(base.LAST_MODEL_DIR),
            "training_state": str(base.TRAINING_STATE_PATH),
        },
    }
    base.METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {base.METRICS_PATH}")

    notes_text = f"""# ModernBERT-base LoRA Resume Notes

## Resume Summary

- Resume mode: `{resume_mode}`
- Completed epochs before resume: `{completed_epochs}`
- Resume start epoch: `{start_epoch}`
- Target total epochs: `{base.EPOCHS}`
- Device: `{device}`
- Trainable parameters: `{trainable_params:,}` / `{total_params:,}` ({trainable_pct:.4f}%)

## Validation Metrics

- Threshold: `{final_val_metrics['threshold']:.2f}`
- Accuracy: `{final_val_metrics['accuracy']:.6f}`
- Precision: `{final_val_metrics['precision']:.6f}`
- Recall: `{final_val_metrics['recall']:.6f}`
- F1: `{final_val_metrics['f1']:.6f}`
- ROC AUC: `{final_val_metrics['roc_auc']:.6f}`

## Reproducibility Note

- Future exact resumes use:
- `outputs/modernbert_lora/last_model/`
- `outputs/modernbert_lora/modernbert_lora_training_state.pt`
- If those are missing for an older run, this script warm-restarts from the saved best adapter.
"""
    base.NOTES_PATH.write_text(notes_text, encoding="utf-8")
    print(f"Saved notes: {base.NOTES_PATH}")

    print("\nRun setup complete.")


if __name__ == "__main__":
    main()
