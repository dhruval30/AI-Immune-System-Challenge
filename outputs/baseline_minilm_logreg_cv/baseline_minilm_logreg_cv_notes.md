# Baseline MiniLM Embeddings + Logistic Regression (CV)

## Summary

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Classifier: Logistic Regression (`class_weight='balanced'`, `solver='liblinear'`, `max_iter=2000`)
- Validation: 5-fold StratifiedKFold with OOF predictions
- Threshold selection: best OOF F1 over thresholds 0.30 to 0.70
- Inference: fold-model probability averaging on test

## Why This Baseline

This is a faster semantic baseline to test sentence-level representation quality before expensive transformer fine-tuning.
It checks whether semantic embeddings improve over lexical-only TF-IDF setups while keeping training lightweight.

## Outputs

- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_metrics.json`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_oof_predictions.csv`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_error_analysis.csv`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_submission.csv`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_notes.md`

## Cache Files

- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_train_embeddings.npy`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_test_embeddings.npy`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_train_embeddings_meta.json`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_test_embeddings_meta.json`
