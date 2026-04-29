# RoBERTa Hard-Weighted Notes

## Why This Run Exists

The current RoBERTa baseline already captures the easy distribution artifact: noisy TRUE and clean FALSE.
The plateau is mostly clean-looking TRUE and messy-looking FALSE. This script keeps `roberta-base`
and normal cross-entropy, but weights those hard regimes more heavily during training.

## Configuration

- Model: `/workspace/gpu/roberta-base`
- Max length: `384`
- Epochs: `6`
- Train batch size: `16`
- Gradient accumulation: `1`
- Effective train batch size: `16`
- Learning rate: `1e-05`
- Loss: `sample_weighted_cross_entropy`
- Hard weights:
  - clean TRUE: `2.5`
  - medium clean TRUE: `1.5`
  - noisy TRUE: `0.85`
  - messy FALSE: `2.0`
  - medium messy FALSE: `1.4`
  - clean FALSE: `0.85`

Weights are normalized by the train split mean raw weight before training.

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Abnormality thresholds fitted on the training split only
- Best checkpoint selected by validation F1 at threshold 0.5
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `0.32`
- Accuracy: `0.959184`
- Precision: `0.944444`
- Recall: `0.910714`
- F1: `0.927273`
- ROC AUC: `0.988679`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[685, 15], [25, 255]]`
- Prediction distribution: `{'pred_FALSE': 710, 'pred_TRUE': 270}`

## Comparison Target

Compare against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.

## Outputs

- `outputs/roberta_hard_weighted/roberta_hard_weighted_metrics.json`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_val_predictions.csv`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_error_analysis.csv`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_test_probabilities.csv`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_submission.csv`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_notes.md`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_weight_summary.csv`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_feature_thresholds.json`
- `outputs/roberta_hard_weighted/best_model/`
- `outputs/roberta_hard_weighted/roberta_hard_weighted_best_state_dict.pt`
