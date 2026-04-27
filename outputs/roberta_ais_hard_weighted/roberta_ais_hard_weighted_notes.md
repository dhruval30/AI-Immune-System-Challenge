# RoBERTa AIS Hard-Weighted Notes

## Why This Run Exists

The current direct anchor is `models/train_roberta_base.py`, which uses ordinary
`roberta-base` fine-tuning and scored `0.90909091` on public LB.

After reviewing the AIS reference material and local PDFs, the target is better framed as
deviant AI-agent behavior: concealed collusion, cryptic coordination, manipulation, or unsafe
coordination inside otherwise ordinary agent conversations. The older hard-weighted script was
useful, but it mostly emphasized surface abnormality/noise. This run keeps the stable RoBERTa
recipe and uses conservative sample weights that combine:

- surface abnormality/noise signals
- AIS semantic cues for collusion, manipulation, cryptic instructions, agent markers, and insurance/assessment context
- hard clean TRUE cases, which are likely the important concealed-risk cases
- benign coordination-like FALSE cases, so the model does not blindly treat all coordination as harmful

## Configuration

- Model: `roberta-base`
- Max length: `256`
- Epochs: `2`
- Train batch size: `8`
- Gradient accumulation: `2`
- Effective train batch size: `16`
- Learning rate: `1e-05`
- Loss: `sample_weighted_cross_entropy`
- AIS semantic features:
  - `insurance_term_count`
  - `coordination_term_count`
  - `manipulation_term_count`
  - `cryptic_term_count`
  - `agent_marker_count`
- Hard weights:
  - hard AIS clean TRUE: `1.9`
  - hard clean TRUE: `1.7`
  - AIS-signal TRUE: `1.4`
  - standard TRUE: `1.05`
  - easy artifact/noisy TRUE: `0.95`
  - hard benign-coordination FALSE: `1.6`
  - hard messy FALSE: `1.45`
  - medium messy FALSE: `1.2`
  - easy clean FALSE: `0.9`

Weights are normalized by the train split mean raw weight before training.

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Abnormality thresholds fitted on the training split only
- Best checkpoint selected by validation F1 at threshold 0.5
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `0.35`
- Accuracy: `0.921429`
- Precision: `0.959276`
- Recall: `0.757143`
- F1: `0.846307`
- ROC AUC: `0.974209`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[691, 9], [68, 212]]`
- Prediction distribution: `{'pred_FALSE': 759, 'pred_TRUE': 221}`

## Comparison Target

Compare against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.

## Outputs

- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_metrics.json`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_val_predictions.csv`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_error_analysis.csv`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_test_probabilities.csv`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_submission.csv`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_notes.md`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_weight_summary.csv`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_feature_thresholds.json`
- `outputs/roberta_ais_hard_weighted/best_model/`
- `outputs/roberta_ais_hard_weighted/roberta_ais_hard_weighted_best_state_dict.pt`
