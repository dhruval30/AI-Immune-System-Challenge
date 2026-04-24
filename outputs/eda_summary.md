# EDA Summary

## EDA Findings

- **Target**: Binary harmful conversation detection with labels in train (`TRUE` = harmful, `FALSE` = non-harmful).
- **Input structure**: Train and test are JSONL with a primary `text` field. In this dataset, `text` values are plain strings (not nested objects/lists), but helper extraction logic should still stay robust for nested variants.
- **Class balance**: Train has 3500 `FALSE` (71.43%) and 1400 `TRUE` (28.57%), so the dataset is moderately imbalanced toward `FALSE`.
- **Modeling text field**: The `text` column is the core feature field for later modeling. A derived text field is useful during EDA only, especially for consistent metric calculations.
- **Risks / patterns noticed**: There are noisy and unusual phrasing patterns, occasional multiline formatting, and mixed writing quality that can hide intent. Label leakage from explicit keywords alone is unlikely to be sufficient.
- **Train/test/submission checks**:
  - Train columns: label, text
  - Test columns: text
  - Test contains label column: no
  - `solution_format.csv` rows = 2100, test rows = 2100 (match: yes)
  - Submission columns expected: label
- **Recommended next step**: Build a reproducible baseline text-classification pipeline using the train `text` field, stratified validation, and careful handling of class imbalance.

## Manual Sample Review Addendum

- The initial EDA confirmed the core dataset structure, class imbalance, and submission schema expectations.
- Manual reading of train samples suggests `TRUE` is not always direct harmful intent in a narrow toxicity sense.
- Many `TRUE` samples also look abnormal in language behavior: incoherent word mixing, spam-like phrasing, semantic breaks, strange formatting, or suspicious tone.
- `FALSE` samples are often more coherent and purpose-driven, even when grammar quality is imperfect.
- This shifts the near-term analysis direction: treat the problem as harmful/unsafe/abnormal detection, not keyword-level harm detection only.
- Future feature engineering for EDA should measure fluency, coherence, randomness, formatting noise, and suspicious phrasing patterns in addition to explicit harmful cues.
- This is a working hypothesis from observed samples, not a guaranteed labeling rule, and should be validated with broader quantitative analysis.
