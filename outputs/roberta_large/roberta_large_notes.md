# RoBERTa-large Fine-Tuning Notes

## Why This Run Exists

This is a direct larger-capacity clone of the current best `roberta-base` script.
It keeps the successful normal CrossEntropyLoss setup and changes only model capacity plus batch settings needed for memory safety.

## Run Configuration

- Model: `roberta-large`
- Seed: `42`
- Device: `mps`
- Max length: `256`
- Epochs: `1`
- Train batch size: `2` (effective: `2`)
- Eval batch size: `4`
- Gradient accumulation: `8` (effective: `8`)
- Learning rate: `1e-05` (effective: `1e-05`)
- Weight decay: `0.01`
- Adam epsilon: `1e-08` (effective: `1e-08`)

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Best checkpoint selected by validation F1 at threshold 0.5 each epoch
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `0.35`
- Accuracy: `0.897959`
- Precision: `0.928571`
- Recall: `0.696429`
- F1: `0.795918`
- ROC AUC: `0.964898`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[685, 15], [85, 195]]`
- Prediction distribution: `{'pred_FALSE': 770, 'pred_TRUE': 210}`

## Outputs

- `outputs/roberta_large/roberta_large_metrics.json`
- `outputs/roberta_large/roberta_large_val_predictions.csv`
- `outputs/roberta_large/roberta_large_error_analysis.csv`
- `outputs/roberta_large/roberta_large_test_probabilities.csv`
- `outputs/roberta_large/roberta_large_submission.csv`
- `outputs/roberta_large/roberta_large_notes.md`
- `outputs/roberta_large/best_model/`
- `outputs/roberta_large/roberta_large_best_state_dict.pt`
