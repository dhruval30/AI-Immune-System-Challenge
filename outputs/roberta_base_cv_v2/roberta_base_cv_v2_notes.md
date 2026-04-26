# RoBERTa-base CV V2 Notes

## Run Configuration

- Model: `roberta-base`
- Device: `mps`
- Seeds: `[42, 52, 62]`
- Folds: `5` (StratifiedKFold)
- Epochs per fold: `1`
- Max length: `256`
- Train batch size: `8`
- Eval batch size: `16`
- Gradient accumulation: `2`
- Learning rate: `1e-05`
- Weight decay: `0.01`

## Design Rationale

- Keeps `roberta-base` as the main model because it is the strongest single-model baseline so far.
- Uses 5 folds instead of a single split for more stable OOF estimates.
- Uses 1 epoch per fold to stay close to the observed sweet spot where RoBERTa performed best before overfitting.
- Uses 3 seeds to reduce variance and produce ensemble-ready probabilities.
- Saves per-seed OOF and test probabilities so a later stacker/blender can use them directly.

## Thresholding

- Threshold tuned on averaged OOF probabilities.
- Search range: `0.25` to `0.50`
- Selected threshold: `0.25`

## Final Averaged OOF Metrics

- Accuracy: `0.907755`
- Precision: `0.913613`
- Recall: `0.747857`
- F1: `0.822467`
- ROC AUC: `0.965288`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[3401, 99], [353, 1047]]`
- Prediction distribution: `{'pred_FALSE': 3754, 'pred_TRUE': 1146}`

## Outputs

- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_metrics.json`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_oof_predictions.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_error_analysis.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_test_probabilities.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_submission.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_notes.md`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_seed_*_oof_predictions.csv`
- `outputs/roberta_base_cv_v2/roberta_base_cv_v2_seed_*_test_probabilities.csv`
- `outputs/roberta_base_cv_v2/fold_checkpoints/`
