#!/usr/bin/env python3
"""
Discover synthetic/noise patterns correlated with binary labels.

Dependency install command:
pip install pandas numpy tqdm

This script does not train a model. It derives text features, computes
feature-label correlations, searches statistically supported single-feature
patterns, and searches simple two-feature rules.
"""

from __future__ import annotations

import json
import math
import re
import string
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


INSTALL_CMD = "pip install pandas numpy tqdm"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "analysis" / "noise_patterns"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"

DISCOVERED_FEATURES_PATH = OUTPUT_DIR / "discovered_features.csv"
FEATURE_CORRELATIONS_PATH = OUTPUT_DIR / "feature_label_correlations.csv"
STRONG_PATTERNS_PATH = OUTPUT_DIR / "strong_patterns.csv"
COMBINED_RULES_PATH = OUTPUT_DIR / "combined_rules.csv"
SUMMARY_PATH = OUTPUT_DIR / "noise_patterns_summary.md"

MIN_SUPPORT = 80
MIN_RULE_SUPPORT = 80
TOP_N = 10
SEED = 42

WORD_RE = re.compile(r"[A-Za-z0-9_']+")
RAW_TOKEN_RE = re.compile(r"\S+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
CAMEL_GLUE_RE = re.compile(r"[a-z][A-Z][a-z]")
PROMPT_RE = re.compile(r"\b(user|assistant|system)\s*:", re.IGNORECASE)
CODE_RE = re.compile(r"```|</s>|<s>|\[/inst\]|\b(import|def|class|return|print)\b|console\.log|\{[\"']?[A-Za-z_][\w -]*[\"']?\s*:")
MARKDOWN_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*]\s+|\d+\.\s+|>\s+)", re.MULTILINE)

PUNCT_CHARS = set(string.punctuation)
BRACKET_CHARS = set("()[]{}<>")
QUOTE_CHARS = set("\"'`“”‘’")
VOWELS = set("aeiou")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


@dataclass
class CandidatePattern:
    pattern_id: str
    feature: str
    condition: str
    mask: np.ndarray


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"Loading {path.name}", unit="rows"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1"}:
        return "TRUE"
    if text in {"FALSE", "F", "0"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def longest_repeated_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
            previous = char
    return int(best)


def extract_base_tokens(texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(match.group(0).lower() for match in WORD_RE.finditer(text))
    return counts


def extract_features(text: str, corpus_token_counts: Counter) -> dict[str, float]:
    text = normalize_text(text)
    lower = text.lower()
    chars = list(text)
    char_count = len(chars)
    safe_char_count = max(char_count, 1)
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    raw_tokens = RAW_TOKEN_RE.findall(text)
    word_count = len(words)
    safe_word_count = max(word_count, 1)
    raw_token_count = len(raw_tokens)
    safe_raw_token_count = max(raw_token_count, 1)
    word_lengths = [len(word) for word in words]
    token_counts = Counter(words)
    char_counts = Counter(chars)
    sentence_parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentence_parts]

    punctuation_count = sum(char in PUNCT_CHARS for char in chars)
    bracket_count = sum(char in BRACKET_CHARS for char in chars)
    quote_count = sum(char in QUOTE_CHARS for char in chars)
    digit_count = sum(char.isdigit() for char in chars)
    alpha_count = sum(char.isalpha() for char in chars)
    uppercase_count = sum(char.isupper() for char in chars)
    whitespace_count = sum(char.isspace() for char in chars)
    non_ascii_count = sum(ord(char) > 127 for char in chars)
    symbol_count = sum((not char.isalnum()) and (not char.isspace()) for char in chars)

    repeated_token_count = sum(count - 1 for count in token_counts.values() if count > 1)
    repeated_token_types = sum(1 for count in token_counts.values() if count > 1)
    stopword_count = sum(1 for word in words if word in STOPWORDS)
    rare_token_count = sum(1 for word in words if corpus_token_counts.get(word, 0) <= 2)
    hapax_token_count = sum(1 for word in words if corpus_token_counts.get(word, 0) == 1)

    weird_token_count = 0
    long_token_count = 0
    mixed_alnum_token_count = 0
    symbol_heavy_token_count = 0
    no_vowel_long_token_count = 0
    camel_glue_count = 0
    for token in raw_tokens:
        token_lower = token.lower()
        has_alpha = any(char.isalpha() for char in token)
        has_digit = any(char.isdigit() for char in token)
        symbol_chars = sum((not char.isalnum()) for char in token)
        alnum_chars = [char for char in token_lower if char.isalnum()]
        vowel_count = sum(char in VOWELS for char in token_lower)
        is_long = len(token) >= 18
        is_mixed = has_alpha and has_digit
        is_symbol_heavy = symbol_chars >= 3
        is_no_vowel_long = len(alnum_chars) >= 8 and vowel_count == 0
        if is_long:
            long_token_count += 1
        if is_mixed:
            mixed_alnum_token_count += 1
        if is_symbol_heavy:
            symbol_heavy_token_count += 1
        if is_no_vowel_long:
            no_vowel_long_token_count += 1
        if CAMEL_GLUE_RE.search(token):
            camel_glue_count += 1
        if is_long or is_mixed or is_symbol_heavy or is_no_vowel_long or REPEATED_CHAR_RE.search(token):
            weird_token_count += 1

    repeated_phrase_count = 0
    duplicate_bigram_count = 0
    if len(words) >= 2:
        bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        duplicate_bigram_count = sum(count - 1 for count in bigram_counts.values() if count > 1)
    if len(words) >= 3:
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        repeated_phrase_count = sum(count - 1 for count in trigram_counts.values() if count > 1)

    if char_count > 0:
        compressed_len = len(zlib.compress(text.encode("utf-8", errors="ignore")))
        compression_ratio = compressed_len / max(len(text.encode("utf-8", errors="ignore")), 1)
    else:
        compression_ratio = 0.0

    return {
        "char_length": float(char_count),
        "byte_length": float(len(text.encode("utf-8", errors="ignore"))),
        "word_count": float(word_count),
        "raw_token_count": float(raw_token_count),
        "sentence_count": float(len(sentence_parts)),
        "avg_sentence_word_count": float(np.mean(sentence_lengths) if sentence_lengths else 0.0),
        "max_sentence_word_count": float(np.max(sentence_lengths) if sentence_lengths else 0.0),
        "short_sentence_count": float(sum(length <= 4 for length in sentence_lengths)),
        "long_sentence_count": float(sum(length >= 30 for length in sentence_lengths)),
        "avg_word_length": float(np.mean(word_lengths) if word_lengths else 0.0),
        "max_word_length": float(np.max(word_lengths) if word_lengths else 0.0),
        "unique_word_ratio": float(len(set(words)) / safe_word_count),
        "stopword_ratio": float(stopword_count / safe_word_count),
        "rare_token_ratio": float(rare_token_count / safe_word_count),
        "hapax_token_ratio": float(hapax_token_count / safe_word_count),
        "punctuation_ratio": float(punctuation_count / safe_char_count),
        "symbol_ratio": float(symbol_count / safe_char_count),
        "digit_ratio": float(digit_count / safe_char_count),
        "alpha_ratio": float(alpha_count / safe_char_count),
        "uppercase_ratio": float(uppercase_count / max(alpha_count, 1)),
        "whitespace_ratio": float(whitespace_count / safe_char_count),
        "non_ascii_ratio": float(non_ascii_count / safe_char_count),
        "newline_count": float(text.count("\n")),
        "comma_count": float(text.count(",")),
        "quote_count": float(quote_count),
        "bracket_count": float(bracket_count),
        "url_count": float(len(URL_RE.findall(text))),
        "email_count": float(len(EMAIL_RE.findall(text))),
        "repeated_char_count": float(len(REPEATED_CHAR_RE.findall(text))),
        "max_repeated_char_run": float(longest_repeated_char_run(text)),
        "repeated_token_count": float(repeated_token_count),
        "repeated_token_ratio": float(repeated_token_count / safe_word_count),
        "repeated_token_types": float(repeated_token_types),
        "duplicate_bigram_count": float(duplicate_bigram_count),
        "repeated_phrase_count": float(repeated_phrase_count),
        "weird_token_count": float(weird_token_count),
        "weird_token_ratio": float(weird_token_count / safe_raw_token_count),
        "long_token_count": float(long_token_count),
        "mixed_alnum_token_count": float(mixed_alnum_token_count),
        "symbol_heavy_token_count": float(symbol_heavy_token_count),
        "no_vowel_long_token_count": float(no_vowel_long_token_count),
        "camel_glue_count": float(camel_glue_count),
        "char_entropy": entropy_from_counts(char_counts),
        "token_entropy": entropy_from_counts(token_counts),
        "compression_ratio": float(compression_ratio),
        "prompt_role_flag": float(bool(PROMPT_RE.search(text))),
        "code_like_flag": float(bool(CODE_RE.search(text))),
        "markdown_like_flag": float(bool(MARKDOWN_RE.search(text))),
        "template_marker_flag": float(("based on the passage above" in lower) or ("the passage above" in lower)),
        "decode_error_flag": float("[decode error]" in lower),
        "list_or_bullet_flag": float(bool(re.search(r"^\s*([-*]|\d+\.)\s+", text, flags=re.MULTILINE))),
        "has_url_flag": float(bool(URL_RE.search(text))),
        "has_non_ascii_flag": float(non_ascii_count > 0),
        "has_newline_flag": float("\n" in text),
    }


def approx_two_sided_p_from_z(z_score: float) -> float:
    return float(math.erfc(abs(z_score) / math.sqrt(2.0)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def correlation_table(feature_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(labels)
    for feature in [col for col in feature_df.columns if col not in {"row_id", "label", "label_int"}]:
        values = feature_df[feature].to_numpy(dtype=float)
        false_values = values[labels == 0]
        true_values = values[labels == 1]
        corr = pearson_corr(values, labels)
        if abs(corr) < 1:
            t_stat = corr * math.sqrt(max(n - 2, 1) / max(1.0 - corr * corr, 1e-12))
        else:
            t_stat = math.copysign(1e9, corr)
        pooled_std = math.sqrt((float(np.var(false_values)) + float(np.var(true_values))) / 2.0)
        effect_size = (float(np.mean(true_values)) - float(np.mean(false_values))) / pooled_std if pooled_std > 1e-12 else 0.0
        rows.append(
            {
                "feature": feature,
                "mean_FALSE": float(np.mean(false_values)),
                "mean_TRUE": float(np.mean(true_values)),
                "median_FALSE": float(np.median(false_values)),
                "median_TRUE": float(np.median(true_values)),
                "std_FALSE": float(np.std(false_values)),
                "std_TRUE": float(np.std(true_values)),
                "pearson_corr_with_TRUE": corr,
                "abs_corr": abs(corr),
                "cohen_d_TRUE_minus_FALSE": effect_size,
                "approx_p_value": approx_two_sided_p_from_z(t_stat),
                "direction": "higher_TRUE" if corr > 0 else "higher_FALSE",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def pattern_stats(pattern: CandidatePattern, labels: np.ndarray, base_true_rate: float) -> dict[str, Any] | None:
    mask = pattern.mask.astype(bool)
    support = int(mask.sum())
    if support < MIN_SUPPORT:
        return None
    true_count = int(labels[mask].sum())
    false_count = support - true_count
    if true_count < 10 and false_count < 10:
        return None
    true_rate = true_count / support
    false_rate = false_count / support
    outside = ~mask
    true_out = int(labels[outside].sum())
    false_out = int(outside.sum() - true_out)
    se = math.sqrt(max(base_true_rate * (1.0 - base_true_rate) / support, 1e-12))
    z_score = (true_rate - base_true_rate) / se
    p_value = approx_two_sided_p_from_z(z_score)
    log_odds_ratio = math.log(((true_count + 0.5) * (false_out + 0.5)) / ((false_count + 0.5) * (true_out + 0.5)))
    direction = "TRUE" if true_rate >= base_true_rate else "FALSE"
    purity = max(true_rate, false_rate)
    lift_true = true_rate / base_true_rate if base_true_rate > 0 else 0.0
    lift_false = false_rate / (1.0 - base_true_rate) if base_true_rate < 1 else 0.0
    return {
        "pattern_id": pattern.pattern_id,
        "feature": pattern.feature,
        "condition": pattern.condition,
        "support_count": support,
        "support_pct": support / len(labels),
        "true_count": true_count,
        "false_count": false_count,
        "true_rate": true_rate,
        "false_rate": false_rate,
        "base_true_rate": base_true_rate,
        "lift_TRUE": lift_true,
        "lift_FALSE": lift_false,
        "purity": purity,
        "direction": direction,
        "z_score": z_score,
        "approx_p_value": p_value,
        "confidence": 1.0 - min(max(p_value, 0.0), 1.0),
        "log_odds_ratio_vs_rest": log_odds_ratio,
        "abs_log_odds_ratio": abs(log_odds_ratio),
    }


def make_numeric_patterns(feature_df: pd.DataFrame, feature: str) -> list[CandidatePattern]:
    values = feature_df[feature].to_numpy(dtype=float)
    patterns: list[CandidatePattern] = []
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0 or np.nanstd(finite_values) < 1e-12:
        return patterns

    quantiles = {
        "lte_p05": 0.05,
        "lte_p10": 0.10,
        "gte_p90": 0.90,
        "gte_p95": 0.95,
    }
    for name, q in quantiles.items():
        threshold = float(np.quantile(finite_values, q))
        if name.startswith("lte"):
            mask = values <= threshold
            condition = f"{feature} <= p{int(q * 100):02d} ({threshold:.6g})"
        else:
            mask = values >= threshold
            condition = f"{feature} >= p{int(q * 100):02d} ({threshold:.6g})"
        patterns.append(CandidatePattern(f"{feature}__{name}", feature, condition, mask))

    if np.nanmin(finite_values) <= 0 <= np.nanmax(finite_values):
        patterns.append(CandidatePattern(f"{feature}__gt_0", feature, f"{feature} > 0", values > 0))
        patterns.append(CandidatePattern(f"{feature}__eq_0", feature, f"{feature} == 0", values == 0))

    try:
        buckets = pd.qcut(values, q=5, duplicates="drop")
        for bucket in pd.Series(buckets).dropna().unique():
            mask = pd.Series(buckets).astype(str).to_numpy() == str(bucket)
            patterns.append(
                CandidatePattern(
                    f"{feature}__quantile_{str(bucket).replace(' ', '')}",
                    feature,
                    f"{feature} in quantile bin {bucket}",
                    mask,
                )
            )
    except ValueError:
        pass

    return patterns


def build_patterns(feature_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    base_true_rate = float(labels.mean())
    pattern_rows = []
    for feature in tqdm([col for col in feature_df.columns if col not in {"row_id", "label", "label_int"}], desc="Scoring single-feature patterns"):
        for pattern in make_numeric_patterns(feature_df, feature):
            stats = pattern_stats(pattern, labels, base_true_rate)
            if stats is not None:
                pattern_rows.append(stats)
    patterns_df = pd.DataFrame(pattern_rows)
    if patterns_df.empty:
        return patterns_df
    # Keep supported/significant patterns with practical label enrichment.
    # This prevents huge-support, tiny-lift patterns from dominating the TRUE table.
    min_true_rate = max(base_true_rate + 0.08, base_true_rate * 1.30)
    true_filter = (
        (patterns_df["direction"] == "TRUE")
        & (patterns_df["true_rate"] >= min_true_rate)
        & (patterns_df["true_count"] >= 40)
    )
    false_filter = (
        (patterns_df["direction"] == "FALSE")
        & (patterns_df["false_rate"] >= 0.85)
        & (patterns_df["false_count"] >= 80)
    )
    patterns_df = patterns_df[
        (patterns_df["support_count"] >= MIN_SUPPORT)
        & (patterns_df["approx_p_value"] <= 0.01)
        & (true_filter | false_filter)
    ].copy()
    return patterns_df.sort_values(["direction", "abs_log_odds_ratio", "support_count"], ascending=[True, False, False])


def build_combined_rules(patterns_df: pd.DataFrame, feature_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if patterns_df.empty:
        return patterns_df
    base_true_rate = float(labels.mean())
    # Limit search to the strongest supported primitive patterns to avoid rare overfit rules.
    candidates_df = patterns_df.sort_values("abs_log_odds_ratio", ascending=False).head(80).reset_index(drop=True)
    candidate_masks: list[tuple[str, str, str, np.ndarray]] = []
    for _, row in candidates_df.iterrows():
        feature = row["feature"]
        condition = row["condition"]
        pattern_id = row["pattern_id"]
        candidate = make_mask_from_condition(feature_df, feature, condition)
        if candidate is not None:
            candidate_masks.append((pattern_id, feature, condition, candidate))

    rows = []
    for i in tqdm(range(len(candidate_masks)), desc="Searching combined two-feature rules"):
        id_a, feature_a, condition_a, mask_a = candidate_masks[i]
        for j in range(i + 1, len(candidate_masks)):
            id_b, feature_b, condition_b, mask_b = candidate_masks[j]
            if feature_a == feature_b:
                continue
            combined_mask = mask_a & mask_b
            support = int(combined_mask.sum())
            if support < MIN_RULE_SUPPORT:
                continue
            pattern = CandidatePattern(
                pattern_id=f"{id_a}__AND__{id_b}",
                feature=f"{feature_a} + {feature_b}",
                condition=f"({condition_a}) AND ({condition_b})",
                mask=combined_mask,
            )
            stats = pattern_stats(pattern, labels, base_true_rate)
            if stats is None:
                continue
            strong_true = stats["direction"] == "TRUE" and stats["true_rate"] >= 0.50 and stats["true_count"] >= 40
            strong_false = stats["direction"] == "FALSE" and stats["false_rate"] >= 0.90 and stats["false_count"] >= 80
            if stats["approx_p_value"] <= 0.01 and (strong_true or strong_false):
                rows.append(stats)
    if not rows:
        return pd.DataFrame()
    rules_df = pd.DataFrame(rows)
    return rules_df.sort_values(["direction", "abs_log_odds_ratio", "support_count"], ascending=[True, False, False])


def make_mask_from_condition(feature_df: pd.DataFrame, feature: str, condition: str) -> np.ndarray | None:
    values = feature_df[feature].to_numpy(dtype=float)
    match = re.search(r"<= p\d+ \(([-+0-9.eE]+)\)", condition)
    if match:
        return values <= float(match.group(1))
    match = re.search(r">= p\d+ \(([-+0-9.eE]+)\)", condition)
    if match:
        return values >= float(match.group(1))
    if condition.endswith(" > 0"):
        return values > 0
    if condition.endswith(" == 0"):
        return values == 0
    match = re.search(r"in quantile bin \(([-+0-9.eE]+), ([-+0-9.eE]+)\]", condition)
    if match:
        left = float(match.group(1))
        right = float(match.group(2))
        return (values > left) & (values <= right)
    match = re.search(r"in quantile bin \[([-+0-9.eE]+), ([-+0-9.eE]+)\]", condition)
    if match:
        left = float(match.group(1))
        right = float(match.group(2))
        return (values >= left) & (values <= right)
    return None


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = TOP_N) -> str:
    if df.empty:
        return "_No supported patterns found._"
    return df[columns].head(n).to_markdown(index=False, floatfmt=".4f")


def write_summary(
    train_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> None:
    base_true_rate = float(train_df["label_int"].mean())
    true_patterns = patterns_df[patterns_df["direction"] == "TRUE"].sort_values(
        ["abs_log_odds_ratio", "support_count"], ascending=[False, False]
    )
    false_patterns = patterns_df[patterns_df["direction"] == "FALSE"].sort_values(
        ["abs_log_odds_ratio", "support_count"], ascending=[False, False]
    )
    true_rules = rules_df[rules_df["direction"] == "TRUE"].sort_values(
        ["abs_log_odds_ratio", "support_count"], ascending=[False, False]
    ) if not rules_df.empty else pd.DataFrame()
    false_rules = rules_df[rules_df["direction"] == "FALSE"].sort_values(
        ["abs_log_odds_ratio", "support_count"], ascending=[False, False]
    ) if not rules_df.empty else pd.DataFrame()

    top_corr = corr_df.head(15)
    columns = [
        "condition",
        "support_count",
        "support_pct",
        "true_rate",
        "false_rate",
        "lift_TRUE",
        "lift_FALSE",
        "approx_p_value",
        "confidence",
    ]
    rule_columns = [
        "condition",
        "support_count",
        "support_pct",
        "true_rate",
        "false_rate",
        "lift_TRUE",
        "lift_FALSE",
        "approx_p_value",
        "confidence",
    ]
    text = f"""# Noise Pattern Discovery Summary

This analysis derives text features and searches for supported label-correlated patterns. It does not train a model.

## Dataset

- Rows: `{len(train_df)}`
- TRUE rows: `{int(train_df['label_int'].sum())}`
- FALSE rows: `{int((1 - train_df['label_int']).sum())}`
- Base TRUE rate: `{base_true_rate:.4f}`
- Minimum single-pattern support: `{MIN_SUPPORT}`
- Minimum combined-rule support: `{MIN_RULE_SUPPORT}`

## Strongest Feature Correlations

Positive correlation means higher values are associated with TRUE.

{top_corr[['feature', 'mean_FALSE', 'mean_TRUE', 'pearson_corr_with_TRUE', 'cohen_d_TRUE_minus_FALSE', 'approx_p_value', 'direction']].to_markdown(index=False, floatfmt='.4f')}

## Top 10 Patterns Predicting TRUE

{markdown_table(true_patterns, columns)}

## Top 10 Patterns Predicting FALSE

{markdown_table(false_patterns, columns)}

## Top Combined TRUE Rules

{markdown_table(true_rules, rule_columns)}

## Top Combined FALSE Rules

{markdown_table(false_rules, rule_columns)}

## Interpretation

- The strongest supported TRUE patterns are mostly length, repetition, line/formatting, and token-abnormality signals when they appear with enough support.
- The strongest supported FALSE patterns are mostly low-length, low-repetition, high lexical diversity, and absence of structural noise.
- Any pattern above is backed by support count, label rate, lift, and an approximate significance test against the base TRUE rate.
- Patterns with low support are excluded to reduce rare-case overfitting.
- Combined rules should be treated as candidate predictive signals, not ground truth. They are useful when they increase purity while retaining non-trivial support.

## Output Files

- `analysis/noise_patterns/discovered_features.csv`
- `analysis/noise_patterns/feature_label_correlations.csv`
- `analysis/noise_patterns/strong_patterns.csv`
- `analysis/noise_patterns/combined_rules.csv`
- `analysis/noise_patterns/noise_patterns_summary.md`
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Discovering Noise Patterns ===")
    print("No model training is performed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_jsonl(TRAIN_PATH)
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(f"Expected train columns text,label; found {list(train_df.columns)}")
    train_df = train_df.copy()
    train_df["text"] = [normalize_text(value) for value in tqdm(train_df["text"], desc="Normalizing text")]
    train_df["label"] = [normalize_label(value) for value in tqdm(train_df["label"], desc="Normalizing labels")]
    train_df["label_int"] = (train_df["label"] == "TRUE").astype(int)
    labels = train_df["label_int"].to_numpy(dtype=int)

    print("\n=== Building corpus token counts ===")
    corpus_token_counts = extract_base_tokens(train_df["text"].tolist())
    print(f"Unique corpus tokens: {len(corpus_token_counts)}")

    print("\n=== Deriving text features ===")
    feature_rows = [
        extract_features(text, corpus_token_counts)
        for text in tqdm(train_df["text"].tolist(), desc="Extracting features", unit="rows")
    ]
    feature_df = pd.DataFrame(feature_rows)
    feature_df.insert(0, "label_int", labels)
    feature_df.insert(0, "label", train_df["label"].to_numpy())
    feature_df.insert(0, "row_id", np.arange(len(feature_df)))
    feature_df.to_csv(DISCOVERED_FEATURES_PATH, index=False)
    print(f"Saved discovered features: {DISCOVERED_FEATURES_PATH}")

    print("\n=== Computing feature-label correlations ===")
    corr_df = correlation_table(feature_df, labels)
    corr_df.to_csv(FEATURE_CORRELATIONS_PATH, index=False)
    print(f"Saved correlations: {FEATURE_CORRELATIONS_PATH}")

    print("\n=== Scoring single-feature patterns ===")
    patterns_df = build_patterns(feature_df, labels)
    patterns_df.to_csv(STRONG_PATTERNS_PATH, index=False)
    print(f"Saved strong patterns: {STRONG_PATTERNS_PATH}")

    print("\n=== Searching combined rules ===")
    rules_df = build_combined_rules(patterns_df, feature_df, labels)
    rules_df.to_csv(COMBINED_RULES_PATH, index=False)
    print(f"Saved combined rules: {COMBINED_RULES_PATH}")

    print("\n=== Writing summary ===")
    write_summary(train_df, corr_df, patterns_df, rules_df)
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
