# ELECTRA-base Fine-Tuning Notes

## Purpose

This run tests `google/electra-base-discriminator` as a direct alternative to the current RoBERTa anchor.

ELECTRA is worth testing here because its pretraining objective is discriminator-style replaced-token detection. That may help on this dataset if some `TRUE`/`FALSE` separation depends on synthetic, corrupted, or unnatural text artifacts.

This is still a standard supervised binary classifier. It does not use external data, proprietary APIs, pseudo-labels, or retrieval.

## Run Configuration

- Model: `google/electra-base-discriminator`
- Seed: `42`
- Device: `mps`
- Max length: `256`
- Max length rationale: train/test inspection showed only about 4.7% of rows exceed 256 characters and roughly 1% exceed 384 characters
- Epochs: `3`
- Train batch size: `8` (effective: `8`)
- Eval batch size: `16`
- Gradient accumulation: `2` (effective: `2`)
- Learning rate: `1e-05` (effective: `1e-05`)
- Weight decay: `0.01`
- Adam epsilon: `1e-08` (effective: `1e-08`)

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Best checkpoint selected by validation F1 at threshold 0.5 each epoch
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `0.35`
- Accuracy: `0.919388`
- Precision: `0.903614`
- Recall: `0.803571`
- F1: `0.850662`
- ROC AUC: `0.973745`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[676, 24], [55, 225]]`
- Prediction distribution: `{'pred_FALSE': 731, 'pred_TRUE': 249}`

## Outputs

- `outputs/electra_base/electra_base_metrics.json`
- `outputs/electra_base/electra_base_val_predictions.csv`
- `outputs/electra_base/electra_base_error_analysis.csv`
- `outputs/electra_base/electra_base_test_probabilities.csv`
- `outputs/electra_base/electra_base_submission.csv`
- `outputs/electra_base/electra_base_notes.md`
- `outputs/electra_base/best_model/`
- `outputs/electra_base/electra_base_best_state_dict.pt`
