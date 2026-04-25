# ModernBERT-base LoRA Fine-Tuning V2 Notes

## Run Configuration

- Model: `answerdotai/ModernBERT-base`
- Seed: `42`
- Device: `mps`
- Max length: `256`
- Epochs: `8`
- Train batch size: `8` (effective: `8`)
- Eval batch size: `16`
- Gradient accumulation: `2` (effective: `2`)
- Learning rate: `5e-05` (effective: `5e-05`)
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
- Best checkpoint selected by validation F1 at each epoch's best threshold over `0.30-0.70`
- Final threshold taken from the best checkpoint's validation threshold search
- This is a longer, lower-LR LoRA run intended to improve on the earlier 3-epoch ModernBERT LoRA result

## Final Validation Metrics

- Threshold: `0.37`
- Accuracy: `0.917347`
- Precision: `0.854093`
- Recall: `0.857143`
- F1: `0.855615`
- ROC AUC: `0.972515`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[659, 41], [40, 240]]`
- Prediction distribution: `{'pred_FALSE': 699, 'pred_TRUE': 281}`

## Outputs

- `outputs/modernbert_lora_v2/modernbert_lora_v2_metrics.json`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_val_predictions.csv`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_error_analysis.csv`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_test_probabilities.csv`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_submission.csv`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_notes.md`
- `outputs/modernbert_lora_v2/best_model/`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_best_adapter_state_dict.pt`
- `outputs/modernbert_lora_v2/last_model/`
- `outputs/modernbert_lora_v2/modernbert_lora_v2_training_state.pt`
