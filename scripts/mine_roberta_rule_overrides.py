#!/usr/bin/env python3
# pip install pandas numpy scikit-learn tqdm
#
# Mine high-purity text rules on non-validation train rows, validate them on
# the saved RoBERTa validation split, and generate controlled RoBERTa override
# submissions. No test labels are used.

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"

SEED = 42
MIN_TRAIN_SUPPORT = 8
MIN_TRAIN_PURITY = 0.80
MAX_SELECTED_RULES = 40

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ANALYSIS_DIR = ROOT_DIR / "analysis" / "rule_overrides"
SUBMISSION_DIR = OUTPUTS_DIR / "rule_overrides"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_FORMAT_PATH = DATA_DIR / "solution_format.csv"

ROBERTA_VAL_PRED_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_val_predictions.csv"
ROBERTA_TEST_PROB_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_test_probabilities.csv"
ROBERTA_METRICS_PATH = OUTPUTS_DIR / "roberta_base" / "roberta_base_metrics.json"

MINED_RULES_PATH = ANALYSIS_DIR / "mined_rules.csv"
VALIDATED_RULES_PATH = ANALYSIS_DIR / "validated_rules.csv"
SELECTED_RULES_PATH = ANALYSIS_DIR / "selected_rule_sets.csv"
VAL_DIAGNOSTICS_PATH = ANALYSIS_DIR / "validation_override_diagnostics.csv"
TEST_DIAGNOSTICS_PATH = ANALYSIS_DIR / "test_override_diagnostics.csv"
SUMMARY_PATH = ANALYSIS_DIR / "rule_override_summary.md"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "for",
    "from",
    "had",
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
    "may",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
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

REGEX_RULES = {
    "regex::decode_error": re.compile(r"\[decode error\]|decode error|invalid utf|invalid character|unknown encoding", re.I),
    "regex::agent_markers": re.compile(r"\[[0-9]+_(?:system|user|assistant)|(?:^|[\s,])(?:user|assistant|system):", re.I),
    "regex::prompt_roles": re.compile(r"\b(?:user|assistant|system)\s*:", re.I),
    "regex::code_or_json": re.compile(r"```|[\{\}\[\]]|</?\w+>|\"[A-Za-z0-9_]+\"\s*:", re.I),
    "regex::url_or_email": re.compile(r"https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I),
    "regex::paraphrase_prompt": re.compile(r"\bparaphrase\b|\bin simpler terms\b|\bbased on (?:the )?passage\b", re.I),
    "regex::instructional": re.compile(r"\b(?:please|must|should|need to|make sure|ensure|follow|reply|respond|write|generate)\b", re.I),
    "regex::coordination_terms": re.compile(r"\b(?:coordinate|collaborate|cooperate|align|agreement|together|shared|schedule|synchronize)\b", re.I),
    "regex::deception_terms": re.compile(r"\b(?:deceive|fake|false|fabricate|pretend|mislead|secret|hidden|covert|conceal|mask)\b", re.I),
    "regex::manipulation_terms": re.compile(r"\b(?:manipulate|pressure|exploit|persuade|influence|coerce|inflate|rig|steer)\b", re.I),
    "regex::evasion_terms": re.compile(r"\b(?:bypass|evade|avoid detection|workaround|jailbreak|circumvent|loophole)\b", re.I),
    "regex::sensitive_data_terms": re.compile(r"\b(?:password|credential|token|private|confidential|sensitive|secret key|leak|extract)\b", re.I),
    "regex::workflow_terms": re.compile(r"\b(?:tool|workflow|api|function|system|process|pipeline|database|query|automation|agent)\b", re.I),
    "regex::seo_or_spam": re.compile(r"\b(?:best|top|review|reviews|buy|cheap|discount|weight loss|keto|supplement)\b", re.I),
}


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


def encode_label(value: object) -> int:
    return 1 if normalize_label(value) == "TRUE" else 0


def decode_labels(values: np.ndarray) -> np.ndarray:
    return np.where(values.astype(int) == 1, "TRUE", "FALSE")


def load_jsonl(path: Path, has_label: bool, desc: str) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=desc, unit="rows")):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {
                "idx": idx,
                "text": normalize_text(obj.get("text", "")),
            }
            if has_label:
                row["label"] = normalize_label(obj.get("label", ""))
                row["y"] = encode_label(row["label"])
            records.append(row)
    df = pd.DataFrame(records)
    df["norm_text"] = df["text"].map(normalize_text)
    return df


def load_base_threshold() -> float:
    payload = json.loads(ROBERTA_METRICS_PATH.read_text(encoding="utf-8"))
    threshold = payload.get("threshold_tuning", {}).get("selected_threshold")
    if threshold is None:
        threshold = payload.get("validation_final", {}).get("threshold", 0.32)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid RoBERTa threshold: {threshold}")
    return threshold


def text_stats(text: str) -> dict:
    tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())
    chars = len(text)
    words = len(tokens)
    unique_ratio = len(set(tokens)) / max(words, 1)
    stopword_ratio = sum(1 for token in tokens if token in STOPWORDS) / max(words, 1)
    non_ascii_ratio = sum(1 for ch in text if ord(ch) > 127) / max(chars, 1)
    punct_ratio = sum(1 for ch in text if (not ch.isalnum() and not ch.isspace())) / max(chars, 1)
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(chars, 1)
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(sum(1 for ch in text if ch.isalpha()), 1)
    repeated_tokens = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return {
        "tokens": tokens,
        "char_length": chars,
        "word_count": words,
        "unique_ratio": unique_ratio,
        "stopword_ratio": stopword_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "punct_ratio": punct_ratio,
        "digit_ratio": digit_ratio,
        "uppercase_ratio": uppercase_ratio,
        "newline_count": text.count("\n"),
        "repeated_tokens": repeated_tokens,
        "long_token_count": sum(1 for token in tokens if len(token) >= 18),
        "repeated_char_count": len(re.findall(r"(.)\1{3,}", text.lower())),
    }


def token_ngrams(tokens: list[str], max_n: int = 4) -> Iterable[str]:
    clean_tokens = [token for token in tokens if len(token) >= 2]
    for n in range(1, max_n + 1):
        if len(clean_tokens) < n:
            break
        for i in range(0, len(clean_tokens) - n + 1):
            gram_tokens = clean_tokens[i : i + n]
            if all(token in STOPWORDS for token in gram_tokens):
                continue
            phrase = " ".join(gram_tokens)
            if len(phrase) < 3:
                continue
            yield f"ngram{n}::{phrase}"


def extract_rules(text: str) -> set[str]:
    lower = text.lower()
    stats = text_stats(text)
    tokens = stats["tokens"]
    rules: set[str] = set()

    for name, pattern in REGEX_RULES.items():
        if pattern.search(text):
            rules.add(name)

    for gram in token_ngrams(tokens):
        rules.add(gram)

    # Numeric/style bins. These are intentionally coarse to avoid one-off rules.
    if stats["char_length"] <= 60:
        rules.add("shape::char_le_60")
    if stats["char_length"] <= 100:
        rules.add("shape::char_le_100")
    if stats["char_length"] >= 180:
        rules.add("shape::char_ge_180")
    if stats["char_length"] >= 256:
        rules.add("shape::char_ge_256")
    if stats["word_count"] <= 10:
        rules.add("shape::word_le_10")
    if stats["word_count"] >= 35:
        rules.add("shape::word_ge_35")
    if stats["word_count"] >= 50:
        rules.add("shape::word_ge_50")
    if stats["newline_count"] >= 1:
        rules.add("shape::has_newline")
    if stats["newline_count"] >= 2:
        rules.add("shape::newline_ge_2")
    if stats["unique_ratio"] <= 0.55 and stats["word_count"] >= 12:
        rules.add("shape::low_unique_ratio")
    if stats["unique_ratio"] >= 0.95 and stats["word_count"] >= 12:
        rules.add("shape::high_unique_ratio")
    if stats["stopword_ratio"] <= 0.20 and stats["word_count"] >= 12:
        rules.add("shape::low_stopword_ratio")
    if stats["stopword_ratio"] >= 0.55 and stats["word_count"] >= 12:
        rules.add("shape::high_stopword_ratio")
    if stats["non_ascii_ratio"] >= 0.03:
        rules.add("shape::non_ascii_high")
    if stats["punct_ratio"] >= 0.14:
        rules.add("shape::punct_high")
    if stats["digit_ratio"] >= 0.05:
        rules.add("shape::digit_high")
    if stats["uppercase_ratio"] >= 0.18 and stats["word_count"] >= 8:
        rules.add("shape::uppercase_high")
    if stats["repeated_tokens"] >= 1:
        rules.add("shape::repeated_adjacent_token")
    if stats["long_token_count"] >= 1:
        rules.add("shape::has_long_token")
    if stats["repeated_char_count"] >= 1:
        rules.add("shape::repeated_chars")
    if "..." in lower:
        rules.add("shape::ellipsis")
    if "—" in text or "–" in text:
        rules.add("shape::dash_unicode")
    if text.endswith("?"):
        rules.add("shape::ends_question")
    if re.search(r"\b[a-z]+[A-Z][a-z]+", text):
        rules.add("shape::camel_or_joined_token")

    return rules


def mine_rules(mining_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, row in tqdm(mining_df.iterrows(), total=len(mining_df), desc="Mining rule counts", unit="rows"):
        for rule in extract_rules(row["text"]):
            counts[rule][int(row["y"])] += 1

    rows = []
    for rule, counter in counts.items():
        false_count = int(counter[0])
        true_count = int(counter[1])
        support = false_count + true_count
        if support < MIN_TRAIN_SUPPORT:
            continue

        true_precision = true_count / support
        false_precision = false_count / support
        if true_precision >= MIN_TRAIN_PURITY:
            rows.append(
                {
                    "rule": rule,
                    "target_label": "TRUE",
                    "target_y": 1,
                    "train_support": support,
                    "train_true": true_count,
                    "train_false": false_count,
                    "train_precision": true_precision,
                }
            )
        if false_precision >= MIN_TRAIN_PURITY:
            rows.append(
                {
                    "rule": rule,
                    "target_label": "FALSE",
                    "target_y": 0,
                    "train_support": support,
                    "train_true": true_count,
                    "train_false": false_count,
                    "train_precision": false_precision,
                }
            )

    return pd.DataFrame(rows).sort_values(["train_precision", "train_support"], ascending=[False, False])


def metric_bundle(y_true: np.ndarray, pred: np.ndarray, prob_true: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "pred_FALSE": int((pred == 0).sum()),
        "pred_TRUE": int((pred == 1).sum()),
    }
    if prob_true is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, prob_true))
    return result


def build_rule_index(texts: Iterable[str], candidate_rules: set[str], desc: str) -> tuple[list[set[str]], dict[str, list[int]]]:
    row_rules = []
    rule_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(tqdm(list(texts), desc=desc, unit="rows")):
        rules = extract_rules(text)
        matched = rules & candidate_rules
        row_rules.append(matched)
        for rule in matched:
            rule_to_indices[rule].append(i)
    return row_rules, rule_to_indices


def validate_rules(
    mined_rules: pd.DataFrame,
    val_df: pd.DataFrame,
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    y_val: np.ndarray,
) -> pd.DataFrame:
    candidate_rules = set(mined_rules["rule"])
    _, rule_to_indices = build_rule_index(val_df["text"], candidate_rules, desc="Matching rules on RoBERTa validation rows")
    base_metrics = metric_bundle(y_val, base_pred, base_prob)

    rows = []
    mined_lookup = mined_rules.set_index(["rule", "target_y"]).to_dict("index")
    for _, rule_row in tqdm(mined_rules.iterrows(), total=len(mined_rules), desc="Validating candidate rules", unit="rules"):
        rule = rule_row["rule"]
        target_y = int(rule_row["target_y"])
        indices = np.asarray(rule_to_indices.get(rule, []), dtype=int)
        support = int(len(indices))

        if support:
            target_count = int((y_val[indices] == target_y).sum())
            val_precision = target_count / support
        else:
            target_count = 0
            val_precision = 0.0

        override_pred = base_pred.copy()
        if support:
            override_pred[indices] = target_y
        override_metrics = metric_bundle(y_val, override_pred, base_prob)

        original_correct = base_pred == y_val
        new_correct = override_pred == y_val
        fixed = int((~original_correct & new_correct).sum())
        broken = int((original_correct & ~new_correct).sum())
        changed = int((override_pred != base_pred).sum())

        rows.append(
            {
                **mined_lookup[(rule, target_y)],
                "rule": rule,
                "target_label": "TRUE" if target_y == 1 else "FALSE",
                "target_y": target_y,
                "val_support": support,
                "val_target_count": target_count,
                "val_precision": val_precision,
                "changed_val_rows": changed,
                "fixed_val_errors": fixed,
                "broken_val_correct": broken,
                "net_fixed_minus_broken": fixed - broken,
                "base_val_f1": base_metrics["f1"],
                "override_val_f1": override_metrics["f1"],
                "delta_val_f1": override_metrics["f1"] - base_metrics["f1"],
                "override_pred_TRUE": override_metrics["pred_TRUE"],
                "override_pred_FALSE": override_metrics["pred_FALSE"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["delta_val_f1", "net_fixed_minus_broken", "val_precision", "train_precision", "val_support"],
        ascending=[False, False, False, False, False],
    )


def select_rule_sets(validated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if validated.empty:
        return {}

    val_support = validated["val_support"]
    safe = validated[
        (val_support >= 2)
        & (validated["val_precision"] >= 0.90)
        & (validated["net_fixed_minus_broken"] > 0)
        & (validated["delta_val_f1"] > 0)
    ].copy()

    balanced = validated[
        (val_support >= 2)
        & (validated["val_precision"] >= 0.75)
        & (validated["net_fixed_minus_broken"] > 0)
        & (validated["delta_val_f1"] > 0)
    ].copy()

    true_recall = validated[
        (validated["target_label"] == "TRUE")
        & (val_support >= 1)
        & (validated["val_precision"] >= 0.65)
        & (validated["net_fixed_minus_broken"] > 0)
        & (validated["delta_val_f1"] > 0)
    ].copy()

    false_precision = validated[
        (validated["target_label"] == "FALSE")
        & (val_support >= 1)
        & (validated["val_precision"] >= 0.75)
        & (validated["net_fixed_minus_broken"] > 0)
        & (validated["delta_val_f1"] > 0)
    ].copy()

    high_train_only = validated[
        (validated["train_support"] >= 20)
        & (validated["train_precision"] >= 0.95)
        & (validated["val_support"] >= 1)
        & (validated["val_precision"] >= 0.60)
        & (validated["net_fixed_minus_broken"] >= 0)
    ].copy()

    rule_sets = {
        "safe": safe.head(MAX_SELECTED_RULES),
        "balanced": balanced.head(MAX_SELECTED_RULES),
        "true_recall": true_recall.head(MAX_SELECTED_RULES),
        "false_precision": false_precision.head(MAX_SELECTED_RULES),
        "high_train_only": high_train_only.head(MAX_SELECTED_RULES),
    }
    return {name: df for name, df in rule_sets.items() if not df.empty}


def apply_rule_set(
    texts: Iterable[str],
    base_pred: np.ndarray,
    selected_rules: pd.DataFrame,
    desc: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    if selected_rules.empty:
        return base_pred.copy(), pd.DataFrame()

    selected_lookup = {
        row.rule: {
            "target_y": int(row.target_y),
            "target_label": row.target_label,
            "score": 0.70 * float(row.val_precision) + 0.30 * float(row.train_precision),
            "val_precision": float(row.val_precision),
            "train_precision": float(row.train_precision),
        }
        for row in selected_rules.itertuples(index=False)
    }
    selected_rule_names = set(selected_lookup)

    final_pred = base_pred.copy()
    diagnostics = []
    for i, text in enumerate(tqdm(list(texts), desc=desc, unit="rows")):
        matched = extract_rules(text) & selected_rule_names
        if not matched:
            continue

        best_by_label: dict[int, tuple[str, float]] = {}
        for rule in matched:
            info = selected_lookup[rule]
            target_y = info["target_y"]
            score = info["score"]
            if target_y not in best_by_label or score > best_by_label[target_y][1]:
                best_by_label[target_y] = (rule, score)

        if len(best_by_label) == 1:
            target_y = next(iter(best_by_label))
            chosen_rule, chosen_score = best_by_label[target_y]
        else:
            false_rule, false_score = best_by_label.get(0, ("", -1.0))
            true_rule, true_score = best_by_label.get(1, ("", -1.0))
            if abs(true_score - false_score) < 0.05:
                continue
            if true_score > false_score:
                target_y = 1
                chosen_rule = true_rule
                chosen_score = true_score
            else:
                target_y = 0
                chosen_rule = false_rule
                chosen_score = false_score

        old_pred = int(final_pred[i])
        final_pred[i] = target_y
        diagnostics.append(
            {
                "row_index": i,
                "old_pred": "TRUE" if old_pred == 1 else "FALSE",
                "new_pred": "TRUE" if target_y == 1 else "FALSE",
                "changed": bool(old_pred != target_y),
                "chosen_rule": chosen_rule,
                "chosen_score": chosen_score,
                "matched_rules": " || ".join(sorted(matched)),
            }
        )

    return final_pred, pd.DataFrame(diagnostics)


def make_submission(solution_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    labels = decode_labels(pred)
    if solution_df.columns.tolist() == ["label"]:
        return pd.DataFrame({"label": labels})
    submission = solution_df.copy()
    submission["label"] = labels
    return submission[solution_df.columns.tolist()]


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    train_df = load_jsonl(TRAIN_PATH, has_label=True, desc="Loading train")
    test_df = load_jsonl(TEST_PATH, has_label=False, desc="Loading test")
    solution_df = pd.read_csv(SOLUTION_FORMAT_PATH)
    val_pred_df = pd.read_csv(ROBERTA_VAL_PRED_PATH)
    test_prob_df = pd.read_csv(ROBERTA_TEST_PROB_PATH)

    if len(test_df) != len(solution_df) or len(test_df) != len(test_prob_df):
        raise ValueError("Test, solution_format, and RoBERTa test probabilities row counts must match.")

    base_threshold = load_base_threshold()
    print(f"RoBERTa base threshold: {base_threshold:.2f}")

    val_pred_df["text"] = val_pred_df["text"].map(normalize_text)
    val_pred_df["norm_text"] = val_pred_df["text"].map(normalize_text)
    val_pred_df["y"] = val_pred_df["true_label"].map(encode_label)
    val_pred_df["base_prob_TRUE"] = pd.to_numeric(val_pred_df["pred_prob_TRUE"], errors="raise")
    val_base_pred = (val_pred_df["base_prob_TRUE"].to_numpy(float) >= base_threshold).astype(int)
    y_val = val_pred_df["y"].to_numpy(int)
    val_base_metrics = metric_bundle(y_val, val_base_pred, val_pred_df["base_prob_TRUE"].to_numpy(float))

    val_norm_texts = set(val_pred_df["norm_text"])
    mining_df = train_df[~train_df["norm_text"].isin(val_norm_texts)].copy()
    print(f"Full train rows: {len(train_df)}")
    print(f"RoBERTa validation rows: {len(val_pred_df)}")
    print(f"Rule mining rows excluding validation texts: {len(mining_df)}")
    print(f"Base validation F1: {val_base_metrics['f1']:.6f}")

    print("\n=== Mining rules ===")
    mined_rules = mine_rules(mining_df)
    mined_rules.to_csv(MINED_RULES_PATH, index=False)
    print(f"Saved mined rules: {MINED_RULES_PATH} ({len(mined_rules)} rows)")

    print("\n=== Validating rules on RoBERTa validation split ===")
    validated = validate_rules(
        mined_rules=mined_rules,
        val_df=val_pred_df,
        base_pred=val_base_pred,
        base_prob=val_pred_df["base_prob_TRUE"].to_numpy(float),
        y_val=y_val,
    )
    validated.to_csv(VALIDATED_RULES_PATH, index=False)
    print(f"Saved validated rules: {VALIDATED_RULES_PATH} ({len(validated)} rows)")

    rule_sets = select_rule_sets(validated)
    selected_rows = []
    for name, df in rule_sets.items():
        temp = df.copy()
        temp.insert(0, "rule_set", name)
        selected_rows.append(temp)
    selected_all = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    selected_all.to_csv(SELECTED_RULES_PATH, index=False)
    print(f"Saved selected rule sets: {SELECTED_RULES_PATH}")

    print("\n=== Simulating selected rule sets ===")
    test_prob = pd.to_numeric(test_prob_df["pred_prob_TRUE"], errors="raise").to_numpy(float)
    test_base_pred = (test_prob >= base_threshold).astype(int)

    val_diag_frames = []
    test_diag_frames = []
    summary_records = []

    for name, selected in rule_sets.items():
        val_override_pred, val_diag = apply_rule_set(
            texts=val_pred_df["text"],
            base_pred=val_base_pred,
            selected_rules=selected,
            desc=f"Applying {name} rules to validation",
        )
        test_override_pred, test_diag = apply_rule_set(
            texts=test_df["text"],
            base_pred=test_base_pred,
            selected_rules=selected,
            desc=f"Applying {name} rules to test",
        )

        val_metrics = metric_bundle(y_val, val_override_pred, val_pred_df["base_prob_TRUE"].to_numpy(float))
        changed_val = int((val_override_pred != val_base_pred).sum())
        changed_test = int((test_override_pred != test_base_pred).sum())

        submission_path = SUBMISSION_DIR / f"roberta_rule_override_{name}.csv"
        make_submission(solution_df, test_override_pred).to_csv(submission_path, index=False)

        if not val_diag.empty:
            val_diag.insert(0, "rule_set", name)
            val_diag["true_label"] = decode_labels(y_val[val_diag["row_index"].to_numpy(int)])
            val_diag_frames.append(val_diag)
        if not test_diag.empty:
            test_diag.insert(0, "rule_set", name)
            test_diag["base_prob_TRUE"] = test_prob[test_diag["row_index"].to_numpy(int)]
            test_diag_frames.append(test_diag)

        summary_records.append(
            {
                "rule_set": name,
                "num_rules": int(len(selected)),
                "val_f1": val_metrics["f1"],
                "delta_val_f1": val_metrics["f1"] - val_base_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_pred_TRUE": val_metrics["pred_TRUE"],
                "val_pred_FALSE": val_metrics["pred_FALSE"],
                "changed_val_rows": changed_val,
                "test_pred_TRUE": int((test_override_pred == 1).sum()),
                "test_pred_FALSE": int((test_override_pred == 0).sum()),
                "changed_test_rows": changed_test,
                "submission_path": str(submission_path),
            }
        )

    val_diag_all = pd.concat(val_diag_frames, ignore_index=True) if val_diag_frames else pd.DataFrame()
    test_diag_all = pd.concat(test_diag_frames, ignore_index=True) if test_diag_frames else pd.DataFrame()
    val_diag_all.to_csv(VAL_DIAGNOSTICS_PATH, index=False)
    test_diag_all.to_csv(TEST_DIAGNOSTICS_PATH, index=False)

    summary_df = pd.DataFrame(summary_records).sort_values("delta_val_f1", ascending=False)
    write_summary(summary_df, validated, selected_all, val_base_metrics)
    print(f"Saved validation diagnostics: {VAL_DIAGNOSTICS_PATH}")
    print(f"Saved test diagnostics: {TEST_DIAGNOSTICS_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


def write_summary(
    summary_df: pd.DataFrame,
    validated: pd.DataFrame,
    selected_all: pd.DataFrame,
    base_metrics: dict,
) -> None:
    lines = [
        "# RoBERTa Rule Override Mining Summary",
        "",
        "## Purpose",
        "",
        "This script mines high-purity train patterns and tests whether they can safely override the RoBERTa base model on hard validation cases.",
        "",
        "Rules are mined only on train rows outside the saved RoBERTa validation split, then evaluated on the saved RoBERTa validation rows.",
        "",
        "No test labels are used.",
        "",
        "## Base Validation Metrics",
        "",
        f"- F1: `{base_metrics['f1']:.6f}`",
        f"- Precision: `{base_metrics['precision']:.6f}`",
        f"- Recall: `{base_metrics['recall']:.6f}`",
        f"- Accuracy: `{base_metrics['accuracy']:.6f}`",
        f"- ROC AUC: `{base_metrics.get('roc_auc', float('nan')):.6f}`",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{base_metrics['confusion_matrix']}`",
        "",
        "## Rule Set Results",
        "",
    ]

    if summary_df.empty:
        lines.extend(["No selected rule sets improved validation under current filters.", ""])
    else:
        lines.append(summary_df.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Top Validated Rules",
            "",
        ]
    )
    top_cols = [
        "rule",
        "target_label",
        "train_support",
        "train_precision",
        "val_support",
        "val_precision",
        "fixed_val_errors",
        "broken_val_correct",
        "delta_val_f1",
    ]
    if not validated.empty:
        lines.append(validated[top_cols].head(30).to_markdown(index=False))
    else:
        lines.append("No validated rules found.")
    lines.append("")

    lines.extend(["## Selected Rules", ""])
    if not selected_all.empty:
        lines.append(selected_all[["rule_set", *top_cols]].to_markdown(index=False))
    else:
        lines.append("No rules selected.")
    lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
            f"- Mined rules: `{MINED_RULES_PATH.relative_to(ROOT_DIR)}`",
            f"- Validated rules: `{VALIDATED_RULES_PATH.relative_to(ROOT_DIR)}`",
            f"- Selected rules: `{SELECTED_RULES_PATH.relative_to(ROOT_DIR)}`",
            f"- Validation diagnostics: `{VAL_DIAGNOSTICS_PATH.relative_to(ROOT_DIR)}`",
            f"- Test diagnostics: `{TEST_DIAGNOSTICS_PATH.relative_to(ROOT_DIR)}`",
            f"- Submission candidates: `{SUBMISSION_DIR.relative_to(ROOT_DIR)}/roberta_rule_override_*.csv`",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
