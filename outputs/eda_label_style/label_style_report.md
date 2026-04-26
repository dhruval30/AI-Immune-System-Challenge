# Label Style EDA Report

## Dataset

- Train rows: `4900`
- Label counts: `{'FALSE': 3500, 'TRUE': 1400}`

## Strongest Numeric Differences

Features higher in `TRUE`:

- `char_length`: TRUE mean `188.0093`, FALSE mean `132.8809`, standardized diff `0.6722`
- `word_count`: TRUE mean `30.9536`, FALSE mean `22.2454`, standardized diff `0.5824`
- `max_word_length`: TRUE mean `11.3386`, FALSE mean `10.4909`, standardized diff `0.3240`
- `sentence_count`: TRUE mean `2.2264`, FALSE mean `1.6643`, standardized diff `0.3029`
- `avg_word_length`: TRUE mean `5.0078`, FALSE mean `4.8965`, standardized diff `0.1364`
- `avg_sentence_word_count`: TRUE mean `18.1088`, FALSE mean `16.7093`, standardized diff `0.1304`
- `newline_count`: TRUE mean `0.6986`, FALSE mean `0.4800`, standardized diff `0.0897`
- `risk_keyword_count`: TRUE mean `0.0079`, FALSE mean `0.0020`, standardized diff `0.0837`

Features higher in `FALSE`:

- `stopword_ratio`: TRUE mean `0.2976`, FALSE mean `0.3153`, standardized diff `-0.1737`
- `unique_word_ratio`: TRUE mean `0.8939`, FALSE mean `0.9118`, standardized diff `-0.1667`
- `whitespace_ratio`: TRUE mean `0.1580`, FALSE mean `0.1599`, standardized diff `-0.0749`
- `non_alnum_ratio`: TRUE mean `0.1927`, FALSE mean `0.1951`, standardized diff `-0.0521`
- `digit_ratio`: TRUE mean `0.0055`, FALSE mean `0.0061`, standardized diff `-0.0248`
- `punctuation_ratio`: TRUE mean `0.0328`, FALSE mean `0.0330`, standardized diff `-0.0062`
- `code_symbol_count`: TRUE mean `0.2171`, FALSE mean `0.2066`, standardized diff `0.0054`
- `repeated_token_count`: TRUE mean `0.1507`, FALSE mean `0.1237`, standardized diff `0.0099`

## Interpretation

The labels are not explained by explicit harm terms alone. `TRUE` is better understood as harmful, unsafe, or abnormal generated text. The useful signals are weak and overlapping: length, malformed tokens, non-ASCII/noisy text, code-like fragments, repetition, prompt-template residue, and spam/product wording all matter, but none is a rule by itself.

The practical modeling implication is that RoBERTa should remain the semantic anchor, while deterministic abnormality features should be used as a correction/calibration layer around borderline examples.

## RoBERTa Validation Error Pattern

- Outcome counts: `{'true_negative': 685, 'true_positive': 226, 'false_negative': 54, 'false_positive': 15}`

False positives are usually text that looks abnormal but is labeled `FALSE`: mixed-language noise, code fragments, malformed prompts, and weird blog/product prose.

False negatives are usually less visually extreme but still semantically broken: plausible-looking product, travel, science, or business sentences with drift, odd substitutions, or stitched-together fragments.

## Leaderboard Positive Count Pattern

- Best known submission: `blend_r095_m005_thr032` with score `0.91919192` and `578` TRUE predictions.
- Good public submissions cluster around roughly `574-580` TRUE predictions.
- Bad submissions are often far from that region, but TRUE count alone is not enough; the selected rows still matter.

## Recommended Next Step

Build a lightweight calibrator using RoBERTa probability plus the deterministic label-style features from this EDA. Keep it simple and regularized first. Use the known RoBERTa validation split as a quick experiment, then move to OOF only if the idea shows promise.

Generated files are saved under `outputs/eda_label_style/`.
