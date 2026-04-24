# Baseline TF-IDF + Engineered Features + Logistic Regression Notes

## What Was Trained

- Model: word-level TF-IDF + char-level TF-IDF + engineered numeric text features + Logistic Regression (`class_weight=balanced`)
- Seed: 42
- Data: `data/train_labeled_comp.jsonl` for training/validation, `data/test_labeled_comp.jsonl` for inference

## Why This Baseline

This baseline tests whether adding explicit surface abnormality and fluency proxies improves separation of `TRUE` and `FALSE` over pure TF-IDF.

Engineered features include length, token repetition, casing, punctuation/noise ratios, URL/newline signals, and sentence-level count proxies.

## Validation Metrics

- Accuracy: 0.776531
- Precision: 0.592705
- Recall: 0.696429
- F1: 0.640394
- ROC AUC: 0.841005
- Confusion matrix [[TN, FP], [FN, TP]]: [[566, 134], [85, 195]]
- Validation prediction distribution: {'pred_FALSE': 651, 'pred_TRUE': 329}

## Files Created

- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_metrics.json`
- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_val_predictions.csv`
- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_error_analysis.csv`
- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_submission.csv`
- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_notes.md`

## What To Inspect Next

- Compare false positives/false negatives vs the first baseline.
- Check which engineered feature profiles dominate confident mistakes.
- Decide if additional abnormality features or threshold tuning should be tested before heavier models.
