# GPU RoBERTa-base Bundle

This bundle is intentionally conservative. It uses only `roberta-base` and the competition data.

It does not use:

- DAPT
- distillation
- pseudo labels
- external datasets
- proprietary APIs

The remote `/workspace/gpu/roberta-base` directory should already exist and contain the verified local Hugging Face model files.

## Files

- `data/train_labeled_comp.jsonl`
- `data/test_labeled_comp.jsonl`
- `data/solution_format.csv`
- `models/train_roberta_base.py`
- `models/train_roberta_base_seed_sweep.py`
- `outputs/`

## Copy To Remote

Run this from your local Mac repo root:

```bash
cd /Users/dhruval/Documents/AI-Immune-System-Challenge
rsync -av --exclude 'roberta-base/' --exclude 'outputs/*' gpu/ root@1.193.139.45:/workspace/gpu/
```

If SSH needs the explicit port:

```bash
rsync -av -e 'ssh -p 36767' --exclude 'roberta-base/' --exclude 'outputs/*' gpu/ root@1.193.139.45:/workspace/gpu/
```

## Run On Remote

First reproduce the known RoBERTa-base anchor:

```bash
cd /workspace/gpu
python models/train_roberta_base.py
```

Then run the seed sweep:

```bash
python models/train_roberta_base_seed_sweep.py
```

With more compute, run a larger sweep:

```bash
python models/train_roberta_base_seed_sweep.py \
  --seeds 42 52 62 72 82 92 102 \
  --max-lengths 256 384 \
  --epochs 3 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --grad-accum 1
```

## Current Best RoBERTa-base Direction

The best submitted RoBERTa-base variant so far is the hard-weighted checkpoint inference path (`0.9122807` public LB).
Use the GPU for that family next:

```bash
python models/train_roberta_hard_weighted_gpu.py
```

This writes:

```text
outputs/roberta_hard_weighted_gpu_b64/roberta_hard_weighted_gpu_b64_submission.csv
outputs/roberta_hard_weighted_gpu_b64/roberta_hard_weighted_gpu_b64_test_probabilities.csv
```

After training, generate count-calibrated candidates around the known good TRUE-count region:

```bash
python models/make_probability_count_candidates.py \
  --prob-path outputs/roberta_hard_weighted_gpu_b64/roberta_hard_weighted_gpu_b64_test_probabilities.csv \
  --output-dir outputs/roberta_hard_weighted_gpu_b64/count_candidates \
  --name roberta_hard_weighted_gpu_b64 \
  --counts 568 572 577 582 586
```

## Submission Files

Baseline:

```text
outputs/roberta_base/roberta_base_submission.csv
```

Seed-sweep ensemble:

```text
outputs/roberta_base_seed_sweep/roberta_base_seed_sweep_submission.csv
```

Hard-weighted GPU:

```text
outputs/roberta_hard_weighted_gpu_b64/roberta_hard_weighted_gpu_b64_submission.csv
```
