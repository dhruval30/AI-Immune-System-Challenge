# Baseline TF-IDF + Logistic Regression Notes

## What Was Trained

- Model: word-level TF-IDF + character-level TF-IDF + Logistic Regression (`class_weight=balanced`)
- Seed: 42
- Data: `data/train_labeled_comp.jsonl` for training/validation, `data/test_labeled_comp.jsonl` for inference

## Why This Baseline

This baseline is designed to quickly test whether lexical weirdness, character-level noise, and surface text structure already separate `TRUE` and `FALSE` labels.

It aligns with the current working hypothesis that `TRUE` may capture harmful/unsafe/abnormal language behavior, including incoherence and spam-like artifacts, not only explicit toxicity.

## Validation Metrics

- Accuracy: 0.657143
- Precision: 0.418129
- Recall: 0.510714
- F1: 0.459807
- ROC AUC: 0.670867
- Confusion matrix [[TN, FP], [FN, TP]]: [[501, 199], [137, 143]]
- Validation prediction distribution: {'pred_FALSE': 638, 'pred_TRUE': 342}

## Files Created

- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_metrics.json`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_val_predictions.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_error_analysis.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_submission.csv`
- `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_notes.md`

## What To Inspect Next

- Inspect confident false positives and false negatives in `outputs/baseline_tfidf_logreg/baseline_tfidf_logreg_error_analysis.csv`.
- Check if errors correlate with incoherence, token noise, formatting artifacts, or ambiguous semantics.
- Decide whether the next iteration should add explicit weirdness/fluency features or threshold tuning before heavier models.

## Next Experiment

The next baseline (`models/baseline_tfidf_features_logreg.py`) keeps TF-IDF features and adds explicit abnormality/fluency numeric features (length, token repetition, casing/digits/punctuation/noise ratios, URL/newline signals, and sentence proxies).

Goal: test whether the first baseline's false positives and false negatives were partly caused by missing surface-level numeric signals that TF-IDF alone does not encode strongly enough.
