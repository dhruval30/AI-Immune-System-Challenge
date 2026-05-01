# RoBERTa Hard-Weighted Bootstrapped CE Notes

## Why This Run Exists

The current best GPU run is the hard-weighted RoBERTa model with a fixed `0.32`
submission threshold. The remaining plateau looks more like label noise and
contradictory hard rows than model capacity.

This script keeps the same `roberta-base` hard-weighted setup, but changes the
loss after the first epoch to bootstrapped cross entropy. Bootstrapped CE mixes
the gold one-hot label with the model's detached probability distribution:

`target = beta * one_hot_label + (1 - beta) * stop_gradient(model_probability)`

This keeps labels dominant while reducing the penalty from potentially noisy or
inconsistent labels.

## Configuration

- Model: `/workspace/gpu/roberta-base`
- Max length: `384`
- Epochs: `12`
- Train batch size: `16`
- Gradient accumulation: `1`
- Effective train batch size: `16`
- Learning rate: `1e-05`
- Loss epoch 1: `sample_weighted_cross_entropy`
- Loss epoch 2+: `sample_weighted_bootstrapped_cross_entropy`
- Bootstrap beta: `0.9`
- Fixed leaderboard threshold file: `0.32`
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

- Threshold: `0.48`
- Accuracy: `0.954082`
- Precision: `0.924188`
- Recall: `0.914286`
- F1: `0.919210`
- ROC AUC: `0.986724`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[679, 21], [24, 256]]`
- Prediction distribution: `{'pred_FALSE': 703, 'pred_TRUE': 277}`

## Comparison Target

Primary comparison target is the current best hard-weighted GPU run:

- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_submission_thr0p32.csv`
- Public LB: `0.9347079`

This run should be judged first by its fixed-threshold `0.32` submission.

## Outputs

- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_metrics.json`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_val_predictions.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_error_analysis.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_test_probabilities.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_submission.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_submission_thr0p32.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_notes.md`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_weight_summary.csv`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_feature_thresholds.json`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/best_model/`
- `/workspace/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_best_state_dict.pt`
