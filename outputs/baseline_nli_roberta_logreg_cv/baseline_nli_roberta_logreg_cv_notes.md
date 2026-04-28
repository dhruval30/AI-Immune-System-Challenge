# Baseline NLI-RoBERTa Embeddings + Logistic Regression (CV)

## Summary

- Embedding model: `sentence-transformers/nli-roberta-base-v2`
- Classifier: Logistic Regression (`class_weight='balanced'`, `solver='liblinear'`, `max_iter=2000`)
- Validation: 5-fold StratifiedKFold with OOF predictions
- Threshold selection: best OOF F1 over thresholds 0.30 to 0.70
- Inference: fold-model probability averaging on test

## Why This Baseline

This tests whether an NLI-trained RoBERTa sentence embedding model gives better semantic separation than the previous MiniLM embedding baseline.

This is not transformer fine-tuning. The encoder is used only to generate cached sentence embeddings, then a Logistic Regression meta-classifier is trained with 5-fold CV.

Expected tradeoff: this should capture richer sentence-level semantics than MiniLM, but it will be slower and may still miss dataset-specific style/noise artifacts that full supervised fine-tuning can learn.

## Outputs

- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_metrics.json`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_oof_predictions.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_error_analysis.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_submission.csv`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_notes.md`

## Cache Files

- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_train_embeddings.npy`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_test_embeddings.npy`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_train_embeddings_meta.json`
- `outputs/baseline_nli_roberta_logreg_cv/baseline_nli_roberta_logreg_cv_test_embeddings_meta.json`
