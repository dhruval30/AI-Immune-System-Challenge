#!/usr/bin/env python3
"""
Inspect train-test leakage, near-duplicates, template reuse, and paraphrase-like similarity.

Dependency install command:
pip install pandas numpy scikit-learn tqdm

This script does not train a model. It uses unsupervised text normalization and TF-IDF
nearest-neighbor search to find train/test overlap and high-similarity cases.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


INSTALL_CMD = "pip install pandas numpy scikit-learn tqdm"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "analysis" / "leakage_similarity"

TRAIN_PATH = DATA_DIR / "train_labeled_comp.jsonl"
TEST_PATH = DATA_DIR / "test_labeled_comp.jsonl"
SOLUTION_PATH = DATA_DIR / "solution_format.csv"

TOP_K = 5
SEED = 42

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


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
    if text in {"TRUE", "1", "YES", "Y"}:
        return "TRUE"
    if text in {"FALSE", "0", "NO", "N"}:
        return "FALSE"
    raise ValueError(f"Unexpected label value: {value!r}")


def find_text_column(df: pd.DataFrame, label_col: str | None = None) -> str:
    candidates = ["text", "conversation", "prompt", "content", "input", "messages"]
    for candidate in candidates:
        if candidate in df.columns and candidate != label_col:
            return candidate
    non_label_cols = [c for c in df.columns if c != label_col]
    if len(non_label_cols) == 1:
        return non_label_cols[0]
    object_cols = [c for c in non_label_cols if df[c].dtype == "object"]
    if object_cols:
        return object_cols[0]
    raise ValueError(f"Could not infer text column from columns: {list(df.columns)}")


def exact_key(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.lower()).strip()


def loose_key(text: str) -> str:
    text = URL_RE.sub(" <url> ", text.lower())
    text = EMAIL_RE.sub(" <email> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    text = text.translate(PUNCT_TABLE)
    return WHITESPACE_RE.sub(" ", text).strip()


def template_key(text: str) -> str:
    """A conservative template key: preserve words, mask volatile values and spacing."""
    text = URL_RE.sub(" <URL> ", text.lower())
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = NUMBER_RE.sub(" <NUM> ", text)
    text = re.sub(r"[`*_#>\-]{2,}", " <MARKUP> ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def shape_key(text: str) -> str:
    """Broad structural signature used only for aggregate template diagnostics."""
    chars = []
    for ch in text:
        if ch.isalpha():
            chars.append("A")
        elif ch.isdigit():
            chars.append("0")
        elif ch.isspace():
            chars.append(" ")
        elif ch in string.punctuation:
            chars.append(ch)
        else:
            chars.append("X")
    collapsed = re.sub(r"A+", "A", "".join(chars))
    collapsed = re.sub(r"0+", "0", collapsed)
    collapsed = WHITESPACE_RE.sub(" ", collapsed).strip()
    return collapsed[:500]


def short_text(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def add_keys(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["text_norm"] = [normalize_text(x) for x in tqdm(out[text_col], desc="Normalizing text", unit="rows")]
    out["exact_key"] = [exact_key(x) for x in tqdm(out["text_norm"], desc="Building exact keys", unit="rows")]
    out["loose_key"] = [loose_key(x) for x in tqdm(out["text_norm"], desc="Building loose keys", unit="rows")]
    out["template_key"] = [template_key(x) for x in tqdm(out["text_norm"], desc="Building template keys", unit="rows")]
    out["shape_key"] = [shape_key(x) for x in tqdm(out["text_norm"], desc="Building shape keys", unit="rows")]
    out["char_length"] = out["text_norm"].str.len()
    out["word_count"] = out["text_norm"].map(lambda x: len(WORD_RE.findall(x)))
    return out


def key_group_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, key_col: str, min_train: int = 1) -> pd.DataFrame:
    train_groups = (
        train_df.groupby(key_col, dropna=False)
        .agg(
            train_match_count=("train_row_id", "count"),
            train_TRUE_count=("label_norm", lambda s: int((s == "TRUE").sum())),
            train_FALSE_count=("label_norm", lambda s: int((s == "FALSE").sum())),
            train_row_ids=("train_row_id", lambda s: "|".join(map(str, s.tolist()[:25]))),
            train_labels=("label_norm", lambda s: "|".join(s.tolist()[:25])),
            train_example_text=("text_norm", lambda s: short_text(s.iloc[0])),
        )
        .reset_index()
    )
    test_groups = (
        test_df.groupby(key_col, dropna=False)
        .agg(
            test_match_count=("test_row_id", "count"),
            test_row_ids=("test_row_id", lambda s: "|".join(map(str, s.tolist()[:25]))),
            test_example_text=("text_norm", lambda s: short_text(s.iloc[0])),
        )
        .reset_index()
    )
    merged = test_groups.merge(train_groups, on=key_col, how="inner")
    merged = merged[merged["train_match_count"] >= min_train].copy()
    if merged.empty:
        return merged
    merged["train_TRUE_rate"] = merged["train_TRUE_count"] / merged["train_match_count"]
    merged["train_FALSE_rate"] = merged["train_FALSE_count"] / merged["train_match_count"]
    merged["cross_pair_count"] = merged["train_match_count"] * merged["test_match_count"]
    return merged.sort_values(["test_match_count", "train_match_count", "train_TRUE_rate"], ascending=[False, False, False])


def within_duplicate_summary(df: pd.DataFrame, id_col: str, key_col: str, label_col: str | None = None) -> pd.DataFrame:
    aggregations = {
        "row_count": (id_col, "count"),
        "row_ids": (id_col, lambda s: "|".join(map(str, s.tolist()[:50]))),
        "example_text": ("text_norm", lambda s: short_text(s.iloc[0])),
    }
    if label_col:
        aggregations.update(
            {
                "TRUE_count": (label_col, lambda s: int((s == "TRUE").sum())),
                "FALSE_count": (label_col, lambda s: int((s == "FALSE").sum())),
                "labels": (label_col, lambda s: "|".join(s.tolist()[:50])),
            }
        )
    grouped = df.groupby(key_col, dropna=False).agg(**aggregations).reset_index()
    grouped = grouped[grouped["row_count"] > 1].copy()
    if label_col and not grouped.empty:
        grouped["TRUE_rate"] = grouped["TRUE_count"] / grouped["row_count"]
        grouped["has_label_conflict"] = (grouped["TRUE_count"] > 0) & (grouped["FALSE_count"] > 0)
    return grouped.sort_values("row_count", ascending=False)


def nearest_neighbors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_features: int,
    output_name: str,
) -> pd.DataFrame:
    print(f"\n=== TF-IDF nearest neighbors: {output_name} ===")
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
    )
    all_text = pd.concat([train_df["text_norm"], test_df["text_norm"]], ignore_index=True)
    print("Fitting vectorizer")
    vectorizer.fit(all_text)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    train_matrix = vectorizer.transform(train_df["text_norm"])
    test_matrix = vectorizer.transform(test_df["text_norm"])

    nn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(train_matrix)
    distances, indices = nn.kneighbors(test_matrix)

    rows = []
    for test_pos in tqdm(range(len(test_df)), desc=f"Collecting {output_name} neighbors", unit="rows"):
        test_row = test_df.iloc[test_pos]
        for rank in range(TOP_K):
            train_pos = int(indices[test_pos, rank])
            train_row = train_df.iloc[train_pos]
            similarity = float(1.0 - distances[test_pos, rank])
            rows.append(
                {
                    "method": output_name,
                    "test_row_id": int(test_row["test_row_id"]),
                    "neighbor_rank": rank + 1,
                    "similarity": similarity,
                    "train_row_id": int(train_row["train_row_id"]),
                    "train_label": train_row["label_norm"],
                    "test_char_length": int(test_row["char_length"]),
                    "train_char_length": int(train_row["char_length"]),
                    "test_word_count": int(test_row["word_count"]),
                    "train_word_count": int(train_row["word_count"]),
                    "test_text": short_text(test_row["text_norm"]),
                    "train_text": short_text(train_row["text_norm"]),
                }
            )
    return pd.DataFrame(rows)


def threshold_summary(neighbors: pd.DataFrame, method: str, thresholds: list[float]) -> pd.DataFrame:
    best = neighbors[neighbors["neighbor_rank"] == 1].copy()
    rows = []
    for threshold in thresholds:
        subset = best[best["similarity"] >= threshold]
        if subset.empty:
            rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "test_rows": 0,
                    "test_pct": 0.0,
                    "nearest_train_TRUE_count": 0,
                    "nearest_train_FALSE_count": 0,
                    "nearest_train_TRUE_rate": np.nan,
                }
            )
            continue
        true_count = int((subset["train_label"] == "TRUE").sum())
        false_count = int((subset["train_label"] == "FALSE").sum())
        rows.append(
            {
                "method": method,
                "threshold": threshold,
                "test_rows": int(len(subset)),
                "test_pct": float(len(subset) / len(best)),
                "nearest_train_TRUE_count": true_count,
                "nearest_train_FALSE_count": false_count,
                "nearest_train_TRUE_rate": float(true_count / len(subset)),
            }
        )
    return pd.DataFrame(rows)


def similarity_quantiles(neighbors: pd.DataFrame, method: str) -> pd.DataFrame:
    best = neighbors[neighbors["neighbor_rank"] == 1]["similarity"]
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return pd.DataFrame(
        {
            "method": method,
            "quantile": qs,
            "similarity": [float(best.quantile(q)) for q in qs],
        }
    )


def write_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exact_cross: pd.DataFrame,
    loose_cross: pd.DataFrame,
    template_cross: pd.DataFrame,
    shape_cross: pd.DataFrame,
    train_dupes: pd.DataFrame,
    train_conflicts: pd.DataFrame,
    test_dupes: pd.DataFrame,
    char_neighbors: pd.DataFrame,
    word_neighbors: pd.DataFrame,
    threshold_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    leakage_cases: pd.DataFrame,
) -> None:
    def count_rows(df: pd.DataFrame, col: str) -> int:
        if df.empty:
            return 0
        return int(df[col].sum())

    exact_test_rows = count_rows(exact_cross, "test_match_count")
    loose_test_rows = count_rows(loose_cross, "test_match_count")
    template_test_rows = count_rows(template_cross, "test_match_count")
    shape_test_rows = count_rows(shape_cross, "test_match_count")

    top_char = char_neighbors[char_neighbors["neighbor_rank"] == 1].sort_values("similarity", ascending=False).head(15)
    top_word = word_neighbors[word_neighbors["neighbor_rank"] == 1].sort_values("similarity", ascending=False).head(15)

    lines = []
    lines.append("# Train/Test Leakage And Similarity Inspection")
    lines.append("")
    lines.append("This analysis does not train a model. It checks exact overlap, normalized overlap, template reuse, duplicate groups, and TF-IDF nearest neighbors between train and test.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Train rows: `{len(train_df)}`")
    lines.append(f"- Test rows: `{len(test_df)}`")
    lines.append(f"- Train TRUE rows: `{int((train_df['label_norm'] == 'TRUE').sum())}`")
    lines.append(f"- Train FALSE rows: `{int((train_df['label_norm'] == 'FALSE').sum())}`")
    lines.append("")
    lines.append("## Exact And Normalized Cross-Set Matches")
    lines.append("")
    lines.append(f"- Exact normalized train/test matched test rows: `{exact_test_rows}`")
    lines.append(f"- Loose normalized train/test matched test rows: `{loose_test_rows}`")
    lines.append(f"- Conservative template-key matched test rows: `{template_test_rows}`")
    lines.append(f"- Broad shape-key matched test rows: `{shape_test_rows}`")
    lines.append("")
    lines.append("Interpretation: exact/loose matches are the strongest leakage signal. Template and shape matches are weaker because broad signatures can match unrelated samples with similar structure.")
    lines.append("")
    lines.append("## Within-Set Duplicate Diagnostics")
    lines.append("")
    lines.append(f"- Train exact duplicate groups: `{len(train_dupes)}`")
    lines.append(f"- Train exact duplicate groups with label conflicts: `{len(train_conflicts)}`")
    lines.append(f"- Test exact duplicate groups: `{len(test_dupes)}`")
    lines.append("")
    lines.append("## Nearest-Neighbor Similarity Thresholds")
    lines.append("")
    lines.append(threshold_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Nearest-Neighbor Similarity Quantiles")
    lines.append("")
    lines.append(quantile_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Top Character TF-IDF Similarities")
    lines.append("")
    if top_char.empty:
        lines.append("No character-neighbor rows found.")
    else:
        lines.append(top_char[["test_row_id", "similarity", "train_row_id", "train_label", "test_text", "train_text"]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Top Word TF-IDF Similarities")
    lines.append("")
    if top_word.empty:
        lines.append("No word-neighbor rows found.")
    else:
        lines.append(top_word[["test_row_id", "similarity", "train_row_id", "train_label", "test_text", "train_text"]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Potential Leakage Cases")
    lines.append("")
    lines.append(f"- Candidate test rows with exact/loose/template match or high TF-IDF similarity: `{len(leakage_cases)}`")
    if not leakage_cases.empty:
        label_counts = leakage_cases["nearest_train_label"].value_counts().to_dict()
        lines.append(f"- Nearest train label distribution among candidate rows: `{label_counts}`")
    lines.append("")
    lines.append("## Practical Implication")
    lines.append("")
    lines.append("- If exact or loose cross-set matches exist, those rows are high-risk leakage candidates and the matched train labels should be inspected manually.")
    lines.append("- If many test rows have very high character/word TF-IDF similarity to train rows, nearest-neighbor label transfer or probability calibration may help.")
    lines.append("- If only moderate similarity exists, this is less like direct leakage and more like repeated synthetic generation templates.")
    lines.append("- Do not blindly override model predictions from these rules. Use the output files to inspect high-confidence cases first.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for filename in [
        "cross_exact_matches.csv",
        "cross_loose_matches.csv",
        "cross_template_matches.csv",
        "cross_shape_matches.csv",
        "train_duplicate_groups.csv",
        "train_duplicate_label_conflicts.csv",
        "test_duplicate_groups.csv",
        "nearest_neighbors_char_tfidf.csv",
        "nearest_neighbors_word_tfidf.csv",
        "similarity_threshold_summary.csv",
        "similarity_quantiles.csv",
        "potential_leakage_cases.csv",
        "leakage_similarity_summary.md",
    ]:
        lines.append(f"- `analysis/leakage_similarity/{filename}`")

    (OUTPUT_DIR / "leakage_similarity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Dependency install command:")
    print(INSTALL_CMD)
    print("\n=== Train/Test Leakage And Similarity Inspection ===")
    print("No model training is performed.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl(TRAIN_PATH)
    test_df = load_jsonl(TEST_PATH)
    solution_df = pd.read_csv(SOLUTION_PATH)

    label_col = "label" if "label" in train_df.columns else None
    if label_col is None:
        raise ValueError(f"Could not find label column in train columns: {list(train_df.columns)}")
    train_text_col = find_text_column(train_df, label_col=label_col)
    test_text_col = find_text_column(test_df, label_col=None)

    if len(test_df) != len(solution_df):
        raise ValueError(f"Test row count {len(test_df)} != solution row count {len(solution_df)}")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Solution shape: {solution_df.shape}")
    print(f"Train text column: {train_text_col}")
    print(f"Test text column: {test_text_col}")

    train_df = train_df.rename(columns={train_text_col: "text"}).copy()
    test_df = test_df.rename(columns={test_text_col: "text"}).copy()
    train_df.insert(0, "train_row_id", np.arange(len(train_df)))
    test_df.insert(0, "test_row_id", np.arange(len(test_df)))
    train_df["label_norm"] = [normalize_label(x) for x in tqdm(train_df[label_col], desc="Normalizing labels", unit="rows")]

    print("\n=== Building text keys ===")
    train_df = add_keys(train_df, "text")
    test_df = add_keys(test_df, "text")

    print("\n=== Exact/normalized/template matches ===")
    exact_cross = key_group_summary(train_df, test_df, "exact_key")
    loose_cross = key_group_summary(train_df, test_df, "loose_key")
    template_cross = key_group_summary(train_df, test_df, "template_key")
    shape_cross = key_group_summary(train_df, test_df, "shape_key", min_train=2)

    train_dupes = within_duplicate_summary(train_df, "train_row_id", "exact_key", label_col="label_norm")
    train_conflicts = train_dupes[train_dupes["has_label_conflict"]].copy() if not train_dupes.empty else train_dupes
    test_dupes = within_duplicate_summary(test_df, "test_row_id", "exact_key")

    exact_cross.to_csv(OUTPUT_DIR / "cross_exact_matches.csv", index=False)
    loose_cross.to_csv(OUTPUT_DIR / "cross_loose_matches.csv", index=False)
    template_cross.to_csv(OUTPUT_DIR / "cross_template_matches.csv", index=False)
    shape_cross.to_csv(OUTPUT_DIR / "cross_shape_matches.csv", index=False)
    train_dupes.to_csv(OUTPUT_DIR / "train_duplicate_groups.csv", index=False)
    train_conflicts.to_csv(OUTPUT_DIR / "train_duplicate_label_conflicts.csv", index=False)
    test_dupes.to_csv(OUTPUT_DIR / "test_duplicate_groups.csv", index=False)

    print(f"Exact matched test rows: {int(exact_cross['test_match_count'].sum()) if not exact_cross.empty else 0}")
    print(f"Loose matched test rows: {int(loose_cross['test_match_count'].sum()) if not loose_cross.empty else 0}")
    print(f"Template matched test rows: {int(template_cross['test_match_count'].sum()) if not template_cross.empty else 0}")
    print(f"Shape matched test rows: {int(shape_cross['test_match_count'].sum()) if not shape_cross.empty else 0}")

    print("\n=== Nearest-neighbor similarity ===")
    char_neighbors = nearest_neighbors(
        train_df,
        test_df,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
        output_name="char_tfidf_3_5",
    )
    word_neighbors = nearest_neighbors(
        train_df,
        test_df,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        output_name="word_tfidf_1_2",
    )

    char_neighbors.to_csv(OUTPUT_DIR / "nearest_neighbors_char_tfidf.csv", index=False)
    word_neighbors.to_csv(OUTPUT_DIR / "nearest_neighbors_word_tfidf.csv", index=False)

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    threshold_df = pd.concat(
        [
            threshold_summary(char_neighbors, "char_tfidf_3_5", thresholds),
            threshold_summary(word_neighbors, "word_tfidf_1_2", thresholds),
        ],
        ignore_index=True,
    )
    quantile_df = pd.concat(
        [
            similarity_quantiles(char_neighbors, "char_tfidf_3_5"),
            similarity_quantiles(word_neighbors, "word_tfidf_1_2"),
        ],
        ignore_index=True,
    )
    threshold_df.to_csv(OUTPUT_DIR / "similarity_threshold_summary.csv", index=False)
    quantile_df.to_csv(OUTPUT_DIR / "similarity_quantiles.csv", index=False)

    best_char = char_neighbors[char_neighbors["neighbor_rank"] == 1][
        ["test_row_id", "similarity", "train_row_id", "train_label", "test_text", "train_text"]
    ].rename(
        columns={
            "similarity": "char_similarity",
            "train_row_id": "char_nearest_train_row_id",
            "train_label": "char_nearest_train_label",
            "train_text": "char_nearest_train_text",
        }
    )
    best_word = word_neighbors[word_neighbors["neighbor_rank"] == 1][
        ["test_row_id", "similarity", "train_row_id", "train_label", "train_text"]
    ].rename(
        columns={
            "similarity": "word_similarity",
            "train_row_id": "word_nearest_train_row_id",
            "train_label": "word_nearest_train_label",
            "train_text": "word_nearest_train_text",
        }
    )
    leakage_cases = best_char.merge(best_word, on="test_row_id", how="inner")
    leakage_cases["has_exact_match"] = leakage_cases["test_row_id"].isin(
        set().union(*[set(map(int, str(x).split("|"))) for x in exact_cross["test_row_ids"]]) if not exact_cross.empty else set()
    )
    leakage_cases["has_loose_match"] = leakage_cases["test_row_id"].isin(
        set().union(*[set(map(int, str(x).split("|"))) for x in loose_cross["test_row_ids"]]) if not loose_cross.empty else set()
    )
    leakage_cases["has_template_match"] = leakage_cases["test_row_id"].isin(
        set().union(*[set(map(int, str(x).split("|"))) for x in template_cross["test_row_ids"]]) if not template_cross.empty else set()
    )
    leakage_cases["nearest_train_label"] = np.where(
        leakage_cases["char_similarity"] >= leakage_cases["word_similarity"],
        leakage_cases["char_nearest_train_label"],
        leakage_cases["word_nearest_train_label"],
    )
    leakage_cases = leakage_cases[
        leakage_cases["has_exact_match"]
        | leakage_cases["has_loose_match"]
        | leakage_cases["has_template_match"]
        | (leakage_cases["char_similarity"] >= 0.85)
        | (leakage_cases["word_similarity"] >= 0.80)
    ].sort_values(["has_exact_match", "has_loose_match", "has_template_match", "char_similarity", "word_similarity"], ascending=False)
    leakage_cases.to_csv(OUTPUT_DIR / "potential_leakage_cases.csv", index=False)

    write_summary(
        train_df=train_df,
        test_df=test_df,
        exact_cross=exact_cross,
        loose_cross=loose_cross,
        template_cross=template_cross,
        shape_cross=shape_cross,
        train_dupes=train_dupes,
        train_conflicts=train_conflicts,
        test_dupes=test_dupes,
        char_neighbors=char_neighbors,
        word_neighbors=word_neighbors,
        threshold_df=threshold_df,
        quantile_df=quantile_df,
        leakage_cases=leakage_cases,
    )

    print("\n=== Saved outputs ===")
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
