# RoBERTa Pseudo-Label Fine-Tuning Notes

## Run Configuration

- Model: `roberta-base`
- Seed: `42`
- Device: `mps`
- Max length: `256`
- Epochs: `2`
- Target train batch size: `16`
- Fallback train batch size: `8`
- Effective train batch size: `16`
- Eval batch size: `16`
- Gradient accumulation: `2`
- Learning rate: `5e-06`
- Weight decay: `0.01`
- Adam epsilon: `1e-08`

## Pseudo-Labeling

- Source probs: `outputs/roberta_base/roberta_base_test_probabilities.csv`
- TRUE rule: `pred_prob_TRUE >= 0.95`
- FALSE rule: `pred_prob_TRUE <= 0.05`
- Pseudo TRUE count: `362`
- Pseudo FALSE count: `1412`
- Pseudo total added: `1774`
- Validation remains original-labeled data only.

## Validation Metrics

- Threshold: `0.30`
- Accuracy: `0.878571`
- Precision: `0.939891`
- Recall: `0.614286`
- F1: `0.742981`
- ROC AUC: `0.954689`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[689, 11], [108, 172]]`
- Prediction distribution: `{'pred_FALSE': 797, 'pred_TRUE': 183}`

## Outputs

- `outputs/roberta_pseudolabel/roberta_pseudolabel_metrics.json`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_pseudo_summary.json`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_pseudo_samples.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_val_predictions.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_error_analysis.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_test_probabilities.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_submission.csv`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_notes.md`
- `outputs/roberta_pseudolabel/best_model/`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_best_state_dict.pt`
