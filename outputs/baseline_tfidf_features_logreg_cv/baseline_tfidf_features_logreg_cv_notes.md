# Baseline TF-IDF + Engineered Features + Logistic Regression (CV)

## What Changed

- Replaced single split validation with 5-fold Stratified K-Fold (`shuffle=True`, `random_state=42`).
- Generated OOF predictions across all training rows.
- Tuned classification threshold on OOF probabilities by maximizing F1 over thresholds 0.30 to 0.70.
- Used OOF predictions for error analysis.
- Trained a full-data model and used fold-model probability averaging for final test inference.

## Pipeline

- Word TF-IDF (`ngram_range=(1,2)`, `min_df=2`, `max_features=100000`)
- Char TF-IDF (`analyzer='char_wb'`, `ngram_range=(3,5)`, `min_df=2`, `max_features=100000`)
- Engineered numeric features (15 features) + `StandardScaler`
- Logistic Regression (`class_weight='balanced'`, `solver='saga'`, `max_iter=2000`, `random_state=42`)

## Why This Setup

This configuration improves validation stability and reduces dependence on one lucky/unlucky split.
It directly tests whether explicit surface abnormality/fluency features improve signal over TF-IDF alone.

## Outputs

- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_metrics.json`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_oof_predictions.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_error_analysis.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_submission.csv`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_notes.md`
