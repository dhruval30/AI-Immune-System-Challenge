#!/usr/bin/env python3
"""
Local-only short semantic tagging for AI Immune System train rows.

No test data is read. No training is performed. No submission is generated.

Default backend is deterministic heuristics. Optional local OpenAI-compatible
backend is allowed only for localhost/127.0.0.1 URLs.

Usage examples:

Dry run:
python analysis/local_short_row_tags.py --limit 20

Cheap useful run:
python analysis/local_short_row_tags.py --only-hard --resume --sleep 0.1

Full train run:
python analysis/local_short_row_tags.py --resume --sleep 0.1

Local OpenAI-compatible endpoint, if self-hosted:
python analysis/local_short_row_tags.py --backend local-openai --base-url http://localhost:8000/v1 --only-hard --resume
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data" / "train_labeled_comp.jsonl"

BASE_VAL_PRED_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_val_predictions.csv"
BASE_ERROR_PATH = ROOT_DIR / "outputs" / "roberta_base" / "roberta_base_error_analysis.csv"
HARD_SUBSET_PATH = ROOT_DIR / "analysis" / "hard_subset" / "roberta_base_hard_subset.csv"

OUTPUT_DIR = ROOT_DIR / "analysis" / "local_tags"
JSONL_OUT = OUTPUT_DIR / "train_local_short_tags.jsonl"
CSV_OUT = OUTPUT_DIR / "train_local_short_tags.csv"
SUMMARY_OUT = OUTPUT_DIR / "local_short_tag_summary.md"
FAILED_ROWS_OUT = OUTPUT_DIR / "failed_rows.jsonl"

ALLOWED_FAMILIES = {
    "clean_benign",
    "coordination",
    "covert_instruction",
    "manipulation",
    "deception",
    "policy_evasion",
    "tool_or_workflow",
    "code_or_structured",
    "noise_or_corrupt",
    "generic_filler",
    "unclear",
}

ALLOWED_INTENTS = {
    "benign_info",
    "benign_plan",
    "coordinate",
    "instruct",
    "hide_or_obfuscate",
    "manipulate",
    "evade",
    "extract_or_leak",
    "none",
    "unclear",
}

ALLOWED_RISKS = {"none", "low", "med", "high"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
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
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}

KEYWORDS = {
    "coordination": {
        "coordinate",
        "coordination",
        "collaborate",
        "collaboration",
        "align",
        "agreement",
        "cooperate",
        "shared",
        "together",
        "schedule",
        "synchronize",
        "negotiate",
    },
    "covert_instruction": {
        "secret",
        "hidden",
        "covert",
        "discreet",
        "quietly",
        "undercover",
        "coded",
        "codeword",
        "signal",
        "without detection",
        "conceal",
        "mask",
    },
    "manipulation": {
        "manipulate",
        "pressure",
        "exploit",
        "persuade",
        "mislead",
        "influence",
        "threaten",
        "force",
        "coerce",
        "inflate",
        "rig",
        "steer",
    },
    "deception": {
        "deceive",
        "deception",
        "fake",
        "false",
        "fabricate",
        "pretend",
        "lie",
        "misrepresent",
        "cover story",
        "forged",
        "spoof",
    },
    "policy_evasion": {
        "bypass",
        "evade",
        "avoid detection",
        "workaround",
        "jailbreak",
        "ignore policy",
        "circumvent",
        "loophole",
        "filter",
    },
    "tool_or_workflow": {
        "tool",
        "workflow",
        "api",
        "function",
        "system",
        "process",
        "pipeline",
        "database",
        "query",
        "task",
        "automation",
        "agent",
    },
    "extract_or_leak": {
        "extract",
        "leak",
        "exfiltrate",
        "password",
        "credential",
        "token",
        "private",
        "confidential",
        "sensitive",
        "secret key",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only short semantic tagging for train rows.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum selected rows to annotate.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this original train row index.")
    parser.add_argument("--resume", action="store_true", help="Skip idx values already present in output JSONL.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between local endpoint calls.")
    parser.add_argument("--model", default="local-model", help="Local endpoint model name.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max completion tokens for local endpoint.")
    parser.add_argument("--only-hard", action="store_true", help="Annotate only hard/error rows.")
    parser.add_argument(
        "--backend",
        choices=["heuristic", "local-openai"],
        default="heuristic",
        help="Tagging backend. local-openai must point to a localhost endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible local base URL. Must be localhost/127.0.0.1.",
    )
    parser.add_argument(
        "--api-key-env",
        default="LOCAL_LLM_API_KEY",
        help="Optional env var for local endpoint bearer token.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=2000,
        help="Truncate text in local endpoint prompts to this many characters.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def load_train() -> pd.DataFrame:
    records = []
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading train JSONL", unit="rows")):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                {
                    "idx": idx,
                    "text": normalize_text(row.get("text", "")),
                    "label": normalize_label(row.get("label", "")),
                }
            )
    return pd.DataFrame(records)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def attach_base_context(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["norm_text"] = train_df["text"].map(normalize_text)
    train_df["base_prob_TRUE"] = "NA"
    train_df["base_pred_label"] = "NA"
    train_df["base_error_type"] = "NA"

    val_df = load_optional_csv(BASE_VAL_PRED_PATH)
    if val_df is not None and {"text", "pred_prob_TRUE", "pred_label"}.issubset(val_df.columns):
        val_df = val_df.copy()
        val_df["norm_text"] = val_df["text"].map(normalize_text)
        prob_lookup = val_df.drop_duplicates("norm_text").set_index("norm_text")["pred_prob_TRUE"].to_dict()
        pred_lookup = val_df.drop_duplicates("norm_text").set_index("norm_text")["pred_label"].to_dict()
        train_df["base_prob_TRUE"] = train_df["norm_text"].map(prob_lookup).fillna("NA")
        train_df["base_pred_label"] = train_df["norm_text"].map(pred_lookup).fillna("NA")

    err_df = load_optional_csv(BASE_ERROR_PATH)
    if err_df is not None and {"text", "error_type"}.issubset(err_df.columns):
        err_df = err_df.copy()
        err_df["norm_text"] = err_df["text"].map(normalize_text)
        err_lookup = err_df.drop_duplicates("norm_text").set_index("norm_text")["error_type"].to_dict()
        train_df["base_error_type"] = train_df["norm_text"].map(err_lookup).fillna("NA")

    return train_df


def load_hard_norm_texts() -> set[str]:
    hard_texts: set[str] = set()

    hard_df = load_optional_csv(HARD_SUBSET_PATH)
    if hard_df is not None and "text" in hard_df.columns:
        hard_texts.update(hard_df["text"].map(normalize_text).tolist())

    err_df = load_optional_csv(BASE_ERROR_PATH)
    if err_df is not None and "text" in err_df.columns:
        hard_texts.update(err_df["text"].map(normalize_text).tolist())

    return hard_texts


def selected_rows(train_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = train_df[train_df["idx"] >= args.start_index].copy()

    if args.only_hard:
        hard_texts = load_hard_norm_texts()
        if not hard_texts:
            raise FileNotFoundError(
                "No hard rows found. Expected analysis/hard_subset/roberta_base_hard_subset.csv "
                "or outputs/roberta_base/roberta_base_error_analysis.csv."
            )
        selected = selected[selected["norm_text"].isin(hard_texts)].copy()

    if args.limit is not None:
        selected = selected.head(args.limit).copy()

    return selected


def read_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()

    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(int(json.loads(line)["idx"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def ensure_output_policy(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_OUT.exists() and not args.resume:
        raise FileExistsError(
            f"{JSONL_OUT} already exists. Use --resume to append/skip completed rows, "
            "or move the existing file."
        )
    FAILED_ROWS_OUT.touch(exist_ok=True)


def token_stats(text: str) -> dict:
    words = re.findall(r"[A-Za-z0-9_'-]+", text)
    lower_words = [word.lower() for word in words]
    unique_ratio = len(set(lower_words)) / max(len(lower_words), 1)
    stopword_ratio = sum(1 for word in lower_words if word in STOPWORDS) / max(len(lower_words), 1)
    non_ascii_ratio = sum(1 for ch in text if ord(ch) > 127) / max(len(text), 1)
    punct_ratio = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(len(text), 1)
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(len(text), 1)
    repeated_token_count = sum(1 for a, b in zip(lower_words, lower_words[1:]) if a == b)
    long_token_count = sum(1 for word in words if len(word) >= 18)
    repeated_char_count = len(re.findall(r"(.)\1{3,}", text))
    return {
        "word_count": len(words),
        "char_length": len(text),
        "unique_ratio": unique_ratio,
        "stopword_ratio": stopword_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "punct_ratio": punct_ratio,
        "digit_ratio": digit_ratio,
        "newline_count": text.count("\n"),
        "repeated_token_count": repeated_token_count,
        "long_token_count": long_token_count,
        "repeated_char_count": repeated_char_count,
    }


def contains_any(lower_text: str, terms: Iterable[str]) -> bool:
    return any(term in lower_text for term in terms)


def heuristic_tag(idx: int, label: str, text: str, base_prob_true: object, error_type: object) -> dict:
    lower = text.lower()
    stats = token_stats(text)

    code_or_structured = bool(
        re.search(r"```|[\{\}\[\]]|^\s*[-*]\s+|user:|assistant:|system:|json|<[^>]+>", text, re.I | re.M)
    )
    url_or_email = bool(re.search(r"https?://|www\.|@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    noise = (
        stats["non_ascii_ratio"] > 0.03
        or stats["unique_ratio"] < 0.38 and stats["word_count"] >= 20
        or stats["repeated_token_count"] >= 2
        or stats["long_token_count"] >= 2
        or stats["repeated_char_count"] >= 1
        or stats["punct_ratio"] > 0.18
    )

    family = "clean_benign"
    intent = "benign_info"
    risk = "none"
    hidden = False
    why = "ordinary informational text"

    if noise:
        family = "noise_or_corrupt"
        intent = "none"
        risk = "low"
        why = "noisy or corrupted wording"

    if code_or_structured:
        family = "code_or_structured"
        intent = "instruct"
        risk = "low"
        why = "structured or prompt-like text"

    if contains_any(lower, KEYWORDS["tool_or_workflow"]):
        family = "tool_or_workflow" if family == "clean_benign" else family
        intent = "benign_plan" if intent == "benign_info" else intent
        why = "workflow or tool language"

    if contains_any(lower, KEYWORDS["coordination"]):
        family = "coordination"
        intent = "coordinate"
        risk = "low"
        why = "coordination language present"

    if contains_any(lower, KEYWORDS["covert_instruction"]):
        family = "covert_instruction"
        intent = "hide_or_obfuscate"
        risk = "med"
        hidden = True
        why = "hidden signaling language"

    if contains_any(lower, KEYWORDS["deception"]):
        family = "deception"
        intent = "hide_or_obfuscate"
        risk = "med"
        hidden = True
        why = "deception cue present"

    if contains_any(lower, KEYWORDS["manipulation"]):
        family = "manipulation"
        intent = "manipulate"
        risk = "med"
        hidden = hidden or "quiet" in lower or "secret" in lower
        why = "manipulation cue present"

    if contains_any(lower, KEYWORDS["policy_evasion"]):
        family = "policy_evasion"
        intent = "evade"
        risk = "high"
        hidden = True
        why = "evasion cue present"

    if contains_any(lower, KEYWORDS["extract_or_leak"]):
        family = "policy_evasion"
        intent = "extract_or_leak"
        risk = "high"
        hidden = True
        why = "sensitive data cue"

    if stats["word_count"] < 5:
        family = "generic_filler"
        intent = "none"
        risk = "none"
        hidden = False
        why = "too little content"

    if url_or_email and family == "clean_benign":
        family = "tool_or_workflow"
        intent = "benign_info"
        risk = "low"
        why = "contains link or contact"

    if not text.strip():
        family = "unclear"
        intent = "unclear"
        risk = "none"
        hidden = False
        why = "empty text"

    return normalize_tag(
        {
            "idx": idx,
            "family": family,
            "intent": intent,
            "risk": risk,
            "hidden": hidden,
            "why": why,
        }
    )


def build_prompt(row: pd.Series, max_input_chars: int) -> str:
    text = row["text"]
    if len(text) > max_input_chars:
        text = text[:max_input_chars] + "\n[TRUNCATED]"

    return f"""You are tagging one text from a binary dataset about hidden unsafe/deviant AI-agent behavior.

Return ONLY compact JSON. No markdown. No explanation.

Schema:
{{
  "idx": <int>,
  "family": "<one short label>",
  "intent": "<one short label>",
  "risk": "<none|low|med|high>",
  "hidden": <true|false>,
  "why": "<max 8 words>"
}}

Allowed family labels:
clean_benign
coordination
covert_instruction
manipulation
deception
policy_evasion
tool_or_workflow
code_or_structured
noise_or_corrupt
generic_filler
unclear

Allowed intent labels:
benign_info
benign_plan
coordinate
instruct
hide_or_obfuscate
manipulate
evade
extract_or_leak
none
unclear

Rules:
- "why" must be <= 8 words.
- Use the closest label, do not invent long labels.
- If unsure, use unclear.
- Do not simply copy the ground-truth label.
- Judge the text itself.

Input:
idx: {int(row["idx"])}
label: {row["label"]}
base_prob_TRUE: {row["base_prob_TRUE"]}
base_error_type: {row["base_error_type"]}
text:
{text}

Return only JSON."""


def validate_local_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        raise ValueError(
            "Refusing non-local model endpoint. Use a localhost/127.0.0.1 OpenAI-compatible server."
        )
    return base_url.rstrip("/")


def call_local_openai(row: pd.Series, args: argparse.Namespace, reminder: bool = False) -> dict:
    base_url = validate_local_base_url(args.base_url)
    prompt = build_prompt(row, max_input_chars=args.max_input_chars)
    if reminder:
        prompt += "\n\nReminder: return valid compact JSON only. No markdown."

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "top_p": 0.95,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    api_key = None
    try:
        import os

        api_key = os.environ.get(args.api_key_env)
    except Exception:
        api_key = None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Local endpoint request failed: {exc}") from exc

    content = response_payload["choices"][0]["message"]["content"]
    return parse_model_json(content, idx=int(row["idx"]))


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_model_json(content: str, idx: int) -> dict:
    parsed = json.loads(extract_json_object(content))
    parsed["idx"] = idx
    return normalize_tag(parsed)


def normalize_tag(tag: dict) -> dict:
    idx = int(tag["idx"])
    family = str(tag.get("family", "unclear")).strip()
    intent = str(tag.get("intent", "unclear")).strip()
    risk = str(tag.get("risk", "none")).strip().lower()
    hidden = tag.get("hidden", False)
    why = str(tag.get("why", "unclear")).strip()

    if family not in ALLOWED_FAMILIES:
        family = "unclear"
    if intent not in ALLOWED_INTENTS:
        intent = "unclear"
    if risk not in ALLOWED_RISKS:
        risk = "none"
    if isinstance(hidden, str):
        hidden = hidden.strip().lower() == "true"
    else:
        hidden = bool(hidden)

    why_words = why.split()
    if len(why_words) > 8:
        why = " ".join(why_words[:8])
    if not why:
        why = "unclear"

    return {
        "idx": idx,
        "family": family,
        "intent": intent,
        "risk": risk,
        "hidden": hidden,
        "why": why,
    }


def tag_row(row: pd.Series, args: argparse.Namespace) -> dict:
    if args.backend == "heuristic":
        return heuristic_tag(
            idx=int(row["idx"]),
            label=row["label"],
            text=row["text"],
            base_prob_true=row["base_prob_TRUE"],
            error_type=row["base_error_type"],
        )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return call_local_openai(row, args, reminder=attempt > 0)
        except Exception as exc:
            last_error = exc
            time.sleep(min(0.25 * (attempt + 1), 1.0))
    raise RuntimeError(f"Failed to parse local model output after retries: {last_error}") from last_error


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def build_output_record(row: pd.Series, tag: dict, backend: str) -> dict:
    return {
        "idx": int(row["idx"]),
        "label": row["label"],
        "text": row["text"],
        "base_prob_TRUE": row["base_prob_TRUE"],
        "base_pred_label": row["base_pred_label"],
        "base_error_type": row["base_error_type"],
        "backend": backend,
        **tag,
    }


def convert_jsonl_to_csv() -> pd.DataFrame:
    if not JSONL_OUT.exists():
        return pd.DataFrame()
    records = []
    with JSONL_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records).drop_duplicates("idx", keep="last").sort_values("idx")
    df.to_csv(CSV_OUT, index=False)
    return df


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def count_table_md(df: pd.DataFrame, column: str, title: str) -> list[str]:
    counts = df[column].value_counts(dropna=False)
    lines = [f"## {title}", "", "| value | count |", "|---|---:|"]
    for value, count in counts.items():
        lines.append(f"| {value} | {int(count)} |")
    lines.append("")
    return lines


def crosstab_md(df: pd.DataFrame, row_col: str, col_col: str, title: str) -> list[str]:
    table = pd.crosstab(df[row_col], df[col_col])
    lines = [f"## {title}", ""]
    if table.empty:
        lines.extend(["No rows available.", ""])
        return lines

    columns = list(table.columns)
    lines.append("| value | " + " | ".join(map(str, columns)) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for idx, row in table.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(int(row[col])) for col in columns) + " |")
    lines.append("")
    return lines


def enrichment_table(df: pd.DataFrame, label: str, title: str) -> list[str]:
    if df.empty:
        return [f"## {title}", "", "No rows available.", ""]

    rows = []
    total_label = max((df["label"] == label).sum(), 1)
    total_other = max((df["label"] != label).sum(), 1)
    for family, family_df in df.groupby("family"):
        label_count = int((family_df["label"] == label).sum())
        other_count = int((family_df["label"] != label).sum())
        label_rate = label_count / total_label
        other_rate = other_count / total_other
        enrichment = label_rate / max(other_rate, 1e-9)
        rows.append((family, label_count, other_count, label_rate, other_rate, enrichment))

    rows.sort(key=lambda item: item[-1], reverse=True)
    lines = [f"## {title}", "", "| family | label_count | other_count | label_share | other_share | enrichment |", "|---|---:|---:|---:|---:|---:|"]
    for family, label_count, other_count, label_rate, other_rate, enrichment in rows[:10]:
        lines.append(
            f"| {family} | {label_count} | {other_count} | {pct(label_rate)} | {pct(other_rate)} | {enrichment:.2f} |"
        )
    lines.append("")
    return lines


def examples_md(df: pd.DataFrame) -> list[str]:
    lines = ["## Examples By Major Family", ""]
    major_families = df["family"].value_counts().head(8).index.tolist()
    for family in major_families:
        lines.extend([f"### {family}", ""])
        examples = df[df["family"] == family].head(5)
        for _, row in examples.iterrows():
            text = normalize_text(row["text"]).replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            lines.append(f"- idx `{int(row['idx'])}` label `{row['label']}` risk `{row['risk']}` hidden `{row['hidden']}`: {text}")
        lines.append("")
    return lines


def write_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Local Short Row Tag Summary",
        "",
        "This summary is generated from local-only semantic tags for training rows.",
        "",
        "No test data was read. No model training was performed. No submission was generated.",
        "",
        "## Run Configuration",
        "",
        f"- Backend: `{args.backend}`",
        f"- Only hard rows: `{args.only_hard}`",
        f"- Start index: `{args.start_index}`",
        f"- Limit: `{args.limit}`",
        f"- Rows tagged in CSV: `{len(df)}`",
        "",
    ]

    if df.empty:
        lines.append("No completed tags found.")
        SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(count_table_md(df, "family", "Count By Family"))
    lines.extend(count_table_md(df, "intent", "Count By Intent"))
    lines.extend(count_table_md(df, "risk", "Risk Distribution"))

    hidden_by_label = df.groupby("label")["hidden"].mean().sort_index()
    lines.extend(["## Hidden=True Rate By Ground-Truth Label", "", "| label | hidden_true_rate | count |", "|---|---:|---:|"])
    for label, rate in hidden_by_label.items():
        count = int((df["label"] == label).sum())
        lines.append(f"| {label} | {pct(float(rate))} | {count} |")
    lines.append("")

    lines.extend(crosstab_md(df, "family", "label", "Family Distribution By TRUE/FALSE"))
    lines.extend(crosstab_md(df, "intent", "label", "Intent Distribution By TRUE/FALSE"))

    if "base_error_type" in df.columns and (df["base_error_type"] != "NA").any():
        fn_df = df[df["base_error_type"] == "false_negative"]
        fp_df = df[df["base_error_type"] == "false_positive"]
        lines.extend(count_table_md(fn_df, "family", "Family Distribution For Base False Negatives") if not fn_df.empty else ["## Family Distribution For Base False Negatives", "", "No rows available.", ""])
        lines.extend(count_table_md(fp_df, "family", "Family Distribution For Base False Positives") if not fp_df.empty else ["## Family Distribution For Base False Positives", "", "No rows available.", ""])

    lines.extend(enrichment_table(df, "TRUE", "Top Families Enriched In TRUE"))
    lines.extend(enrichment_table(df, "FALSE", "Top Families Enriched In FALSE"))
    lines.extend(examples_md(df))

    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_policy(args)

    print("=== Local Short Row Tagging ===")
    print("No test data is read.")
    print("No training is performed.")
    print("No submission is generated.")
    print(f"Backend: {args.backend}")
    if args.backend == "local-openai":
        print(f"Local endpoint: {validate_local_base_url(args.base_url)}")

    train_df = attach_base_context(load_train())
    rows = selected_rows(train_df, args)
    completed = read_completed_indices(JSONL_OUT) if args.resume else set()

    print(f"Selected rows before resume skip: {len(rows)}")
    print(f"Completed rows in existing JSONL: {len(completed)}")

    success_count = 0
    failure_count = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Tagging rows", unit="rows"):
        idx = int(row["idx"])
        if idx in completed:
            continue

        try:
            tag = tag_row(row, args)
            output_record = build_output_record(row, tag, backend=args.backend)
            append_jsonl(JSONL_OUT, output_record)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            append_jsonl(
                FAILED_ROWS_OUT,
                {
                    "idx": idx,
                    "label": row["label"],
                    "text": row["text"],
                    "error": str(exc),
                },
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    df = convert_jsonl_to_csv()
    write_summary(df, args)

    print(f"New successful tags: {success_count}")
    print(f"New failures: {failure_count}")
    print(f"Saved JSONL: {JSONL_OUT}")
    print(f"Saved CSV: {CSV_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved failures: {FAILED_ROWS_OUT}")


if __name__ == "__main__":
    main()
