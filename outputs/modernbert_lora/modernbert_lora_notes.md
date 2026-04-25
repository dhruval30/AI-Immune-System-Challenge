# ModernBERT-base LoRA Fine-Tuning Notes

## Run Configuration

- Model: `answerdotai/ModernBERT-base`
- Seed: `42`
- Device: `mps`
- Max length: `256`
- Epochs: `3`
- Train batch size: `8` (effective: `8`)
- Eval batch size: `16`
- Gradient accumulation: `2` (effective: `2`)
- Learning rate: `0.0001` (effective: `0.0001`)
- Weight decay: `0.01`
- Adam epsilon: `1e-08` (effective: `1e-08`)
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- LoRA target modules: `['Wqkv', 'Wi', 'Wo']`
- LoRA modules to save: `['head', 'classifier', 'classifier', 'score']`
- Trainable parameters: `2,281,730` / `151,888,132` (1.5022%)

## Validation Strategy

- Stratified train/validation split (`test_size=0.2`, `random_state=42`)
- Best checkpoint selected by validation F1 at threshold 0.5 each epoch
- Final threshold tuned over 0.30-0.70 for best validation F1

## Final Validation Metrics

- Threshold: `0.30`
- Accuracy: `0.915306`
- Precision: `0.827243`
- Recall: `0.889286`
- F1: `0.857143`
- ROC AUC: `0.967592`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[648, 52], [31, 249]]`
- Prediction distribution: `{'pred_FALSE': 679, 'pred_TRUE': 301}`

## Outputs

- `outputs/modernbert_lora/modernbert_lora_metrics.json`
- `outputs/modernbert_lora/modernbert_lora_val_predictions.csv`
- `outputs/modernbert_lora/modernbert_lora_error_analysis.csv`
- `outputs/modernbert_lora/modernbert_lora_test_probabilities.csv`
- `outputs/modernbert_lora/modernbert_lora_submission.csv`
- `outputs/modernbert_lora/modernbert_lora_notes.md`
- `outputs/modernbert_lora/best_model/`
- `outputs/modernbert_lora/modernbert_lora_best_adapter_state_dict.pt`
