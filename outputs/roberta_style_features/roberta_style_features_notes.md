# RoBERTa With Style Features Notes

## Why This Model Exists

The best direct model so far is the single-split `roberta-base` fine-tune with public LB `0.90909091`.
EDA showed that the TRUE class is not only explicit harmfulness; it also often carries abnormality/noise/style signals such as longer text, repetition, lower naturalness, and unusual word statistics.

This model tests whether those label-style signals help when learned jointly with RoBERTa instead of being applied after the fact.

## Difference From Post-Hoc Calibration

Post-hoc calibration changed RoBERTa probabilities after training. This script changes the model itself:

- RoBERTa learns semantic/coherence features from text.
- A small numeric branch learns EDA/style features.
- Both representations are concatenated before the final classifier.
- Gradients update the fusion head and RoBERTa during fine-tuning.

## Features Used

- `char_length`
- `word_count`
- `sentence_count`
- `max_word_length`
- `avg_word_length`
- `stopword_ratio`
- `unique_word_ratio`
- `repeated_token_count`
- `newline_count`

Feature normalization uses only the training split mean/std, then applies those same statistics to validation and test.

## Hyperparameters

- Base encoder: `roberta-base`
- Max length: `256`
- Epochs: `3`
- Train batch size: `8`
- Gradient accumulation: `2`
- RoBERTa learning rate: `1e-05`
- Feature/head learning rate: `3e-05`
- Weight decay: `0.01`
- Dropout: `0.15`
- Feature hidden size: `32`
- Classifier hidden size: `256`

The encoder LR stays conservative to preserve the successful RoBERTa behavior. The feature/head LR is higher so the newly initialized fusion layers can learn quickly.

## Results

- Best epoch: `1`
- Selected threshold: `0.3`
- Validation F1 at selected threshold: `0.8722627737226277`
- Validation ROC AUC: `0.9688214285714285`
- Test prediction distribution: `{'pred_FALSE': 1503, 'pred_TRUE': 597}`

## Comparison Target

Compare this run against `outputs/roberta_base/roberta_base_submission.csv`, public LB `0.90909091`.
