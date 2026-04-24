# RoBERTa-base Fine-Tuning Notes

## Run Configuration

- Model: `roberta-base`
- Seed: `42`
- Device: `mps`
- Max length: `256`
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

- Threshold: `0.32`
- Accuracy: `0.929592`
- Precision: `0.937759`
- Recall: `0.807143`
- F1: `0.867562`
- ROC AUC: `0.974199`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[685, 15], [54, 226]]`
- Prediction distribution: `{'pred_FALSE': 739, 'pred_TRUE': 241}`

## Outputs

- `outputs/roberta_base/roberta_base_metrics.json`
- `outputs/roberta_base/roberta_base_val_predictions.csv`
- `outputs/roberta_base/roberta_base_error_analysis.csv`
- `outputs/roberta_base/roberta_base_test_probabilities.csv`
- `outputs/roberta_base/roberta_base_submission.csv`
- `outputs/roberta_base/roberta_base_notes.md`
- `outputs/roberta_base/best_model/`
- `outputs/roberta_base/roberta_base_best_state_dict.pt`
