# Hard-Weighted Checkpoint Inference

This is inference only. No training was performed.

## Source

- Checkpoint directory: `outputs/roberta_hard_weighted/best_model`
- State dict: `outputs/roberta_hard_weighted/roberta_hard_weighted_best_state_dict.pt`
- Checkpoint epoch: `1`

## Validation

- F1 @ 0.5: `0.880907`
- Tuned threshold: `0.44`
- F1 @ tuned threshold: `0.889720`
- ROC AUC: `0.974000`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[683, 17], [42, 238]]`

## Test

- Prediction distribution: `{'pred_FALSE': 1528, 'pred_TRUE': 572}`

## Outputs

- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_submission.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_test_probabilities.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_val_predictions.csv`
- `outputs/roberta_hard_weighted_checkpoint_inference/roberta_hard_weighted_checkpoint_inference_metrics.json`
