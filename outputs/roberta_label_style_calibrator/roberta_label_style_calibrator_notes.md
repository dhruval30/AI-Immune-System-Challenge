# RoBERTa Label-Style Calibrator Notes

## Purpose

This is a lightweight correction layer around the existing strong RoBERTa single-split model. It does not train a transformer.

The calibrator uses RoBERTa validation probabilities plus deterministic text-style features from the EDA pass. The goal is to learn small ranking corrections for abnormal/generated/corrupted text patterns that RoBERTa sometimes misclassifies.

## Selected Calibrator

- Feature set: `core_style`
- C: `0.1`
- Class weight: `balanced`
- Internal CV F1: `0.892416`
- Internal CV ROC AUC: `0.972112`
- Internal selected threshold: `0.44`

## Submission Strategy

Submissions are written as top-K variants, not raw threshold variants. This avoids probability calibration mismatch and uses the leaderboard observation that good submissions cluster around 574-580 TRUE predictions.

Recommended first file to try:

`outputs/roberta_label_style_calibrator/roberta_label_style_rankblend0p85_top578_submission.csv`

## Outputs

- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_metrics.json`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_val_scores.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_test_scores.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_coefficients.csv`
- `outputs/roberta_label_style_calibrator/roberta_label_style_calibrator_submission_manifest.csv`
