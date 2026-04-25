# RoBERTa-base 3-Fold CV Fine-Tuning Notes

## Run Configuration

- Model: `roberta-base`
- Seed: `42`
- Device: `mps`
- Folds: `3` (StratifiedKFold)
- Max length: `256`
- Epochs per fold: `2`
- Train batch size: `8`
- Eval batch size: `16`
- Gradient accumulation: `2`
- Learning rate: `1e-05`
- Weight decay: `0.01`

## Training Strategy

- For each fold, best epoch is selected by validation F1 at threshold 0.5.
- OOF probabilities are assembled from fold-best models.
- Final threshold is tuned on full OOF probabilities (0.30 to 0.70) for best OOF F1.
- Test probabilities are averaged across fold-best models.

## Final OOF Metrics

- Threshold: `0.30`
- Accuracy: `0.921429`
- Precision: `0.914286`
- Recall: `0.800000`
- F1: `0.853333`
- ROC AUC: `0.950238`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[3395, 105], [280, 1120]]`
- Prediction distribution: `{'pred_FALSE': 3675, 'pred_TRUE': 1225}`

## Outputs

- `outputs/roberta_base_cv/roberta_base_cv_metrics.json`
- `outputs/roberta_base_cv/roberta_base_cv_oof_predictions.csv`
- `outputs/roberta_base_cv/roberta_base_cv_error_analysis.csv`
- `outputs/roberta_base_cv/roberta_base_cv_test_probabilities.csv`
- `outputs/roberta_base_cv/roberta_base_cv_submission.csv`
- `outputs/roberta_base_cv/roberta_base_cv_notes.md`
- `outputs/roberta_base_cv/fold_checkpoints/`
