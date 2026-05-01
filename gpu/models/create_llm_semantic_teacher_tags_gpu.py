#!/usr/bin/env python3
"""
Create train-only semantic teacher tags with a local open-source LLM.

Dependency install command:
pip install pandas numpy tqdm torch transformers accelerate sentencepiece protobuf

This script does not train a classifier and does not read test data. It uses a
local/open pretrained causal LM only to describe each TRAIN row with compact
semantic diagnostics. These tags are meant to become auxiliary supervision for
RoBERTa, not direct TRUE/FALSE predictions.

Recommended GPU usage:
python models/create_llm_semantic_teacher_tags_gpu.py --model /workspace/gpu/qwen2.5-1.5b-instruct --local-files-only --resume --batch-size 8

Dry run:
python models/create_llm_semantic_teacher_tags_gpu.py --model /workspace/gpu/qwen2.5-1.5b-instruct --local-files-only --limit 40 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTALL_CMD = "pip install pandas numpy tqdm torch transformers accelerate sentencepiece protobuf"

SEED = 42
DEFAULT_MODEL = "/workspace/gpu/qwen2.5-1.5b-instruct"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "semantic_teacher_tags"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
JSONL_PATH = OUTPUT_DIR / "train_semantic_teacher_tags.jsonl"
CSV_PATH = OUTPUT_DIR / "train_semantic_teacher_tags.csv"
SUMMARY_PATH = OUTPUT_DIR / "semantic_teacher_tag_summary.md"
FAILED_PATH = OUTPUT_DIR / "failed_rows.jsonl"

COHERENCE_VALUES = {"normal", "awkward", "broken"}
SEMANTIC_DRIFT_VALUES = {"none", "low", "medium", "high"}
TASK_CLARITY_VALUES = {"clear", "partial", "unclear"}
CORRUPTION_VALUES = {"none", "some", "high"}
SUSPICIOUS_INTENT_VALUES = {"none", "weak", "strong"}
FAMILY_VALUES = {
    "normal_info",
    "prompt_question",
    "product_spam",
    "code_structured",
    "decode_error",
    "multilingual",
    "synthetic_drift",
    "list_poem",
    "unclear",
}

REQUIRED_KEYS = {
    "idx",
    "coherence",
    "semantic_drift",
    "task_clarity",
    "corruption",
    "ordinary_purpose",
    "suspicious_intent",
    "family",
    "why",
}


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_text(value: object) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def load_train() -> pd.DataFrame:
    records = []
    with TRAIN_PATH.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(tqdm(handle, desc="Loading train JSONL", unit="rows")):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "idx": idx,
                    "text": normalize_text(obj.get("text", "")),
                    "label": normalize_label(obj.get("label", "")),
                }
            )
    return pd.DataFrame(records)


def truncate_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...[TRUNCATED]...\n{tail}"


def build_prompt(idx: int, text: str, max_chars: int) -> str:
    clipped = truncate_for_prompt(text, max_chars=max_chars)
    return f"""You are tagging one text from a binary dataset about hidden unsafe or deviant AI-agent behavior.

Return ONLY compact JSON. No markdown. No explanation.

Judge the text itself. Do not predict TRUE/FALSE. Describe observable text quality and semantics.

Schema:
{{
  "idx": {idx},
  "coherence": "normal|awkward|broken",
  "semantic_drift": "none|low|medium|high",
  "task_clarity": "clear|partial|unclear",
  "corruption": "none|some|high",
  "ordinary_purpose": true,
  "suspicious_intent": "none|weak|strong",
  "family": "normal_info|prompt_question|product_spam|code_structured|decode_error|multilingual|synthetic_drift|list_poem|unclear",
  "why": "max 8 words"
}}

Definitions:
- normal coherence: readable and semantically stable
- awkward coherence: mostly readable but unnatural or strained
- broken coherence: meaning collapses or phrases do not fit
- semantic_drift: topic or meaning drifts across unrelated ideas
- task_clarity: whether the text has a clear ordinary purpose
- corruption: formatting/code/encoding/decode/noise artifacts
- suspicious_intent: concealed manipulation, evasion, coordination, deception, or unsafe behavior

Text:
{clipped}
"""


def strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(raw: str) -> dict[str, Any]:
    text = strip_markdown_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def validate_tag(tag: dict[str, Any], expected_idx: int) -> dict[str, Any]:
    missing = REQUIRED_KEYS - set(tag)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")

    tag["idx"] = int(tag["idx"])
    if tag["idx"] != expected_idx:
        tag["idx"] = expected_idx

    tag["coherence"] = str(tag["coherence"]).strip().lower()
    tag["semantic_drift"] = str(tag["semantic_drift"]).strip().lower()
    tag["task_clarity"] = str(tag["task_clarity"]).strip().lower()
    tag["corruption"] = str(tag["corruption"]).strip().lower()
    tag["suspicious_intent"] = str(tag["suspicious_intent"]).strip().lower()
    tag["family"] = str(tag["family"]).strip().lower()
    tag["ordinary_purpose"] = bool(tag["ordinary_purpose"])
    tag["why"] = " ".join(str(tag["why"]).strip().split()[:8])

    if tag["coherence"] not in COHERENCE_VALUES:
        tag["coherence"] = "awkward"
    if tag["semantic_drift"] not in SEMANTIC_DRIFT_VALUES:
        tag["semantic_drift"] = "low"
    if tag["task_clarity"] not in TASK_CLARITY_VALUES:
        tag["task_clarity"] = "partial"
    if tag["corruption"] not in CORRUPTION_VALUES:
        tag["corruption"] = "some"
    if tag["suspicious_intent"] not in SUSPICIOUS_INTENT_VALUES:
        tag["suspicious_intent"] = "weak"
    if tag["family"] not in FAMILY_VALUES:
        tag["family"] = "unclear"
    return tag


def load_completed_indices() -> set[int]:
    if not JSONL_PATH.exists():
        return set()
    done: set[int] = set()
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["idx"]))
            except Exception:
                continue
    return done


def load_failed_indices() -> set[int]:
    if not FAILED_PATH.exists():
        return set()
    failed: set[int] = set()
    with FAILED_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                failed.add(int(obj["idx"]))
            except Exception:
                continue
    return failed


def prompt_to_input_text(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    input_texts = [prompt_to_input_text(tokenizer, prompt) for prompt in prompts]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    decoded = []
    # With left padding, attention-mask lengths are not the decode offset.
    # The generated continuation starts after the padded prompt width.
    prompt_width = inputs["input_ids"].shape[1]
    for row_idx in range(len(prompts)):
        generated_ids = output[row_idx][prompt_width:]
        decoded.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return decoded


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(df: pd.DataFrame) -> None:
    lines = [
        "# Semantic Teacher Tag Summary",
        "",
        "Generated on train rows only. No test data is read and no submission is generated.",
        "",
        f"- Tagged rows: `{len(df)}`",
    ]
    for col in ["coherence", "semantic_drift", "task_clarity", "corruption", "suspicious_intent", "family"]:
        lines.append("")
        lines.append(f"## Count By {col}")
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "count"]
        lines.append(markdown_table(counts))

    if "label" in df.columns:
        lines.append("")
        lines.append("## Family By Label")
        pivot = pd.crosstab(df["family"], df["label"])
        lines.append(markdown_table(pivot.reset_index()))
        lines.append("")
        lines.append("## Coherence By Label")
        pivot = pd.crosstab(df["coherence"], df["label"])
        lines.append(markdown_table(pivot.reset_index()))

    lines.append("")
    lines.append("## Examples")
    for family, group in df.groupby("family"):
        lines.append("")
        lines.append(f"### {family}")
        for _, row in group.head(3).iterrows():
            text = normalize_text(row["text"]).replace("\n", " ")[:180]
            lines.append(f"- idx `{int(row['idx'])}` label `{row.get('label', 'NA')}`: {text}")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-prompt-chars", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry row indices already present in failed_rows.jsonl. Completed rows are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Local LLM Semantic Teacher Tagging ===")
    print("Reads TRAIN only. Does not train. Does not create a submission.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    train_df = load_train()
    if args.retry_failed:
        failed_indices = load_failed_indices()
        print(f"Retry-failed mode enabled. Found {len(failed_indices)} failed row indices.")
        if not failed_indices:
            print("No failed rows found. Exiting.")
            return
        selected = train_df[train_df["idx"].isin(failed_indices)].copy()
        selected = selected.sort_values("idx")
    else:
        selected = train_df[train_df["idx"] >= args.start_index].copy()
    if args.limit is not None:
        selected = selected.head(args.limit)

    done = load_completed_indices() if (args.resume or args.retry_failed) else set()
    if done:
        print(f"Resume enabled. Skipping {len(done)} existing rows.")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    pending_rows = [row for _, row in selected.iterrows() if int(row["idx"]) not in done]
    success_rows = []
    with JSONL_PATH.open("a", encoding="utf-8") as out_handle, FAILED_PATH.open("a", encoding="utf-8") as fail_handle:
        for start in tqdm(range(0, len(pending_rows), args.batch_size), desc="Tagging train rows", unit="batch"):
            batch_rows = pending_rows[start : start + args.batch_size]
            prompts = [
                build_prompt(idx=int(row["idx"]), text=row["text"], max_chars=args.max_prompt_chars)
                for row in batch_rows
            ]
            raw_outputs: list[str] | None = None
            last_error = None
            for attempt in range(2):
                try:
                    raw_outputs = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    prompts = [
                        prompt + "\n\nReturn valid JSON only. No markdown. Use only allowed enum values."
                        for prompt in prompts
                    ]

            if raw_outputs is None:
                for row in batch_rows:
                    fail_handle.write(
                        json.dumps({"idx": int(row["idx"]), "text": row["text"], "error": last_error}) + "\n"
                    )
                fail_handle.flush()
                continue

            for row, raw in zip(batch_rows, raw_outputs, strict=True):
                idx = int(row["idx"])
                try:
                    tag = validate_tag(extract_json_object(raw), expected_idx=idx)
                except Exception as exc:
                    fail_handle.write(json.dumps({"idx": idx, "text": row["text"], "error": str(exc), "raw": raw}) + "\n")
                    fail_handle.flush()
                    continue

                tag["label"] = row["label"]
                tag["text"] = row["text"]
                out_handle.write(json.dumps(tag, ensure_ascii=False) + "\n")
                out_handle.flush()
                success_rows.append(tag)
            if args.sleep > 0:
                time.sleep(args.sleep)

    all_tags = []
    if JSONL_PATH.exists():
        with JSONL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    all_tags.append(json.loads(line))
    tag_df = pd.DataFrame(all_tags).drop_duplicates("idx", keep="last").sort_values("idx")
    if not tag_df.empty:
        tag_df.to_csv(CSV_PATH, index=False)
        write_summary(tag_df)
        print(f"Saved JSONL: {JSONL_PATH}")
        print(f"Saved CSV: {CSV_PATH}")
        print(f"Saved summary: {SUMMARY_PATH}")

    print(f"Rows tagged this run: {len(success_rows)}")


if __name__ == "__main__":
    main()
