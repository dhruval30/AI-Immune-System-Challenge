# Best Submission Reproduction

## Current Best Public LB

- Public leaderboard score: `0.9347079`
- Submission file:
  - `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_submission_thr0p32.csv`
- Main copied submission file:
  - `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_submission.csv`
- Threshold used for the submitted file: `0.32`
- Prediction distribution:
  - `FALSE`: `1501`
  - `TRUE`: `599`

## Model Run

This submission comes from the GPU hard-weighted RoBERTa run:

- Training script:
  - `gpu/models/train_roberta_hard_weighted_gpu.py`
- Output directory:
  - `gpu/outputs/roberta_hard_weighted_gpu_b64_final/`
- Model family: `roberta-base`
- Remote model path used during training: `/workspace/gpu/roberta-base`
- Max length: `384`
- Epochs: `12`
- Train batch size: `16`
- Gradient accumulation: `1`
- Effective train batch size: `16`
- Learning rate: `1e-5`
- Weight decay: `0.01`
- Loss: sample-weighted cross entropy
- Best checkpoint directory:
  - `gpu/outputs/roberta_hard_weighted_gpu_b64_final/best_model/`

## Hard-Weighting Strategy

The run uses sample weights to emphasize hard regimes:

- clean `TRUE`: `2.5`
- medium clean `TRUE`: `1.5`
- noisy `TRUE`: `0.85`
- messy `FALSE`: `2.0`
- medium messy `FALSE`: `1.4`
- clean `FALSE`: `0.85`

Weights are normalized by the train split mean raw weight before training.

## Validation Metrics From Run

Validation used a stratified train/validation split with `test_size=0.2` and `random_state=42`.

Best saved validation metrics at threshold `0.50`:

- Accuracy: `0.958163`
- Precision: `0.947566`
- Recall: `0.903571`
- F1: `0.925046`
- ROC AUC: `0.985791`
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[686, 14], [27, 253]]`
- Validation prediction distribution:
  - `FALSE`: `713`
  - `TRUE`: `267`

Although the run metrics selected threshold `0.50`, the best public LB submission used threshold `0.32` on test probabilities.

## Files Needed

Required files:

- `data/solution_format.csv`
- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_test_probabilities.csv`

Reference files:

- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_metrics.json`
- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_notes.md`
- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_feature_thresholds.json`
- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_weight_summary.csv`

## Recreate The Submitted CSV From Probabilities

Run from repo root:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path("gpu/outputs/roberta_hard_weighted_gpu_b64_final")
prob_path = folder / "roberta_hard_weighted_gpu_b64_test_probabilities.csv"
solution_path = Path("data/solution_format.csv")
out_path = folder / "roberta_hard_weighted_gpu_b64_submission_thr0p32.csv"

threshold = 0.32
prob_df = pd.read_csv(prob_path)
solution_df = pd.read_csv(solution_path)

probs = pd.to_numeric(prob_df["pred_prob_TRUE"], errors="raise").to_numpy(float)
labels = np.where(probs >= threshold, "TRUE", "FALSE")

if solution_df.columns.tolist() == ["label"]:
    submission_df = pd.DataFrame({"label": labels})
else:
    submission_df = solution_df.copy()
    submission_df["label"] = labels
    submission_df = submission_df[solution_df.columns.tolist()]

submission_df.to_csv(out_path, index=False)
print(out_path)
print(submission_df["label"].value_counts().to_dict())
PY
```

Expected output distribution:

```text
FALSE: 1501
TRUE: 599
```

## Remote Sync Command Used

The GPU output folder was copied locally from:

- Remote host: `root@120.238.149.205`
- SSH port: `33060`
- Remote path: `/workspace/gpu/outputs/roberta_hard_weighted_gpu_b64_final/`
- Local path: `gpu/outputs/roberta_hard_weighted_gpu_b64_final/`

Command:

```bash
mkdir -p gpu/outputs/roberta_hard_weighted_gpu_b64_final

rsync -avzP -e "ssh -p 33060" \
  root@120.238.149.205:/workspace/gpu/outputs/roberta_hard_weighted_gpu_b64_final/ \
  gpu/outputs/roberta_hard_weighted_gpu_b64_final/
```

## Comparison

Previous strong anchor:

- `outputs/roberta_base/roberta_base_submission.csv`
- Public LB: `0.90909091`

Current best:

- `gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_submission_thr0p32.csv`
- Public LB: `0.9347079`

Improvement over RoBERTa base:

- Absolute LB gain: `+0.02561699`
