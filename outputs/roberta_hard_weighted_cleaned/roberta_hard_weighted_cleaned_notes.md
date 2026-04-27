# RoBERTa Hard-Weighted Cleaned-Data Notes

## Why This Run Exists

The current RoBERTa baseline already captures the easy distribution artifact: noisy TRUE and clean FALSE.
The plateau is mostly clean-looking TRUE and messy-looking FALSE. This script keeps `roberta-base`
and normal cross-entropy, but weights those hard regimes more heavily during training.

Before training, it removes high-confidence likely label-noise rows using out-of-fold RoBERTa CV predictions.
This keeps the architecture, hyperparameters, and hard-weighting logic unchanged while reducing contradictory training examples.

## Cleaning

- Prediction source: `outputs/roberta_base_cv_v2/roberta_base_cv_v2_oof_predictions.csv`
- Probability column: `oof_prob_TRUE_mean`
- Remove TRUE if `prob_TRUE <= 0.05`
- Remove FALSE if `prob_TRUE >= 0.95`
- Original rows: `4900`
- Removed rows: `100`
- Final rows: `4800`
- Original label distribution: `{'FALSE': 3500, 'TRUE': 1400}`
- Removed label distribution: `{'FALSE': 0, 'TRUE': 100}`
- Final label distribution: `{'FALSE': 3500, 'TRUE': 1300}`

## Configuration

- Model: `roberta-base`
- Max length: `256`
- Epochs: `1`
- Train batch size: `8`
- Gradient accumulation: `2`
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

- Threshold: `0.30`
- Accuracy: `0.905208`
- Precision: `0.912195`
- Recall: `0.719231`
- F1: `0.804301`
- ROC AUC: `0.965302`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[682, 18], [73, 187]]`
- Prediction distribution: `{'pred_FALSE': 755, 'pred_TRUE': 205}`

## Comparison Target

Compare against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.

## Outputs

- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_metrics.json`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_val_predictions.csv`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_error_analysis.csv`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_test_probabilities.csv`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_submission.csv`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_notes.md`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_weight_summary.csv`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_feature_thresholds.json`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_cleaning_summary.json`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_removed_rows.csv`
- `outputs/roberta_hard_weighted_cleaned/best_model/`
- `outputs/roberta_hard_weighted_cleaned/roberta_hard_weighted_cleaned_best_state_dict.pt`
