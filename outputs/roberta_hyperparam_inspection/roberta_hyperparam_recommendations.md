# RoBERTa Hyperparameter Inspection

This report inspects token lengths, truncation risk, and existing RoBERTa validation errors. It does not train a model.

## Recommended Edit For `models/train_roberta_base.py`

```python
MAX_LENGTH = 384
EPOCHS = 2
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-08
```

Effective train batch size: `16`.

## Why

- Use max_length=384 for the next controlled larger-context experiment. Pure coverage says 256 is efficient, but 256 does truncate some train/test rows, and repeating the current 256 setup is not an informative next run.
- Do not jump straight to 512 first: 384 removes nearly all truncation with lower cost and memory risk.
- For max_length=384, use batch_size=4 and grad_accum=4 to preserve effective batch size 16 on MPS.
- Existing RoBERTa selected epoch 1 as best, so a larger-context run should avoid long overfit cycles; use 2 epochs with best-checkpoint saving.
- Validation F1 fell after epoch 1 in the anchor run; do not increase epochs first.
- Validation loss increased after epoch 1 in the anchor run; longer training is likely overfitting.

## Existing RoBERTa Anchor

- Best epoch: `1`
- Selected threshold: `0.32`
- Validation F1: `0.8675623800383877`
- Validation ROC AUC: `0.9741989795918368`
- Test distribution: `{'pred_FALSE': 1523, 'pred_TRUE': 577}`

## RoBERTa Error Length Signal

- False negatives: `54`
- False positives: `15`
- False-negative truncation at 256: `0.00%`
- True-positive truncation at 256: `0.00%`
- False-negative token p95: `62.05`
- True-positive token p95: `64.50`

## Token Length Stats

| dataset   | label   |   token_count |   token_mean |   token_p90 |   token_p95 |   token_p99 |   token_max |
|:----------|:--------|--------------:|-------------:|------------:|------------:|------------:|------------:|
| train     | FALSE   |          3500 |      29.4549 |          40 |       47    |      100.13 |         550 |
| train     | TRUE    |          1400 |      41.1557 |          55 |       64.05 |      216.05 |         266 |
| train     | ALL     |          4900 |      32.798  |          46 |       54    |      130.25 |         550 |
| test      | UNKNOWN |          2100 |      33.081  |          46 |       54    |      206.09 |         425 |

## Current 256 Coverage

| dataset   | label   |   max_length |   truncated_count |   total_count |   truncated_pct |
|:----------|:--------|-------------:|------------------:|--------------:|----------------:|
| train     | ALL     |          256 |                13 |          4900 |        0.265306 |
| train     | FALSE   |          256 |                10 |          3500 |        0.285714 |
| train     | TRUE    |          256 |                 3 |          1400 |        0.214286 |
| test      | UNKNOWN |          256 |                 7 |          2100 |        0.333333 |

## Recommended Max Length Coverage

| dataset   | label   |   max_length |   truncated_count |   total_count |   truncated_pct |
|:----------|:--------|-------------:|------------------:|--------------:|----------------:|
| train     | ALL     |          384 |                 3 |          4900 |       0.0612245 |
| train     | FALSE   |          384 |                 3 |          3500 |       0.0857143 |
| train     | TRUE    |          384 |                 0 |          1400 |       0         |
| test      | UNKNOWN |          384 |                 1 |          2100 |       0.047619  |

## Output Files

- `outputs/roberta_hyperparam_inspection/train_token_lengths.csv`
- `outputs/roberta_hyperparam_inspection/test_token_lengths.csv`
- `outputs/roberta_hyperparam_inspection/token_length_stats_by_label.csv`
- `outputs/roberta_hyperparam_inspection/max_length_coverage.csv`
- `outputs/roberta_hyperparam_inspection/roberta_error_length_analysis.csv`
- `outputs/roberta_hyperparam_inspection/roberta_hyperparam_recommendations.json`
- `outputs/roberta_hyperparam_inspection/roberta_hyperparam_recommendations.md`

## Practical Next Step

If the recommendation is `MAX_LENGTH=384` or `512`, create a separate RoBERTa training script or output directory for that run instead of overwriting the current best `outputs/roberta_base/` artifacts.
