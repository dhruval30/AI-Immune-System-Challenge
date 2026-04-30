# RoBERTa DAPT + Hard-Weighted Notes

## Why This Run Exists

The best single-model family has been RoBERTa-base with hard-example weighting. Model-family swaps and TF-IDF side branches did not transfer. This script keeps RoBERTa-base, but makes the model stronger before supervision using domain-adaptive masked-language-model pretraining on the provided competition text only.

This is one final model, not an ensemble.

## Pipeline

1. DAPT MLM on competition text.
2. Hard-weighted supervised fine-tuning from the DAPT checkpoint.
3. Save tuned-threshold and fixed `0.32` submissions.

## DAPT Configuration

- Base model: `/workspace/gpu/roberta-base`
- Uses test text for unsupervised DAPT: `True`
- DAPT epochs: `2`
- DAPT LR: `2e-05`
- MLM probability: `0.15`
- Best DAPT eval loss: `2.648348`

## Fine-Tuning Configuration

- Max length: `384`
- Classifier epochs: `8`
- Train batch size: `16`
- Gradient accumulation: `1`
- Effective train batch size: `16`
- Classifier LR: `8e-06`
- Loss: `sample_weighted_cross_entropy`
- Fixed submission threshold: `0.32`

## Comparison Targets

- RoBERTa base public LB: `0.90909091`
- Hard-weighted RoBERTa GPU public LB: `0.9347079`
- R-Drop hard-weighted RoBERTa GPU public LB: `0.93542757`

## Outputs

- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/dapt_best_model/`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/best_model/`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_metrics.json`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_val_predictions.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_error_analysis.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_test_probabilities.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_submission.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_submission_thr0p32.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_weight_summary.csv`
- `/workspace/gpu/outputs/roberta_dapt_hard_weighted_gpu/roberta_dapt_hard_weighted_gpu_feature_thresholds.json`
