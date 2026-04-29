# RoBERTa DAPT Pipeline Guide

## Goal

This is the new honest RoBERTa mainline for the bitgrit AI Immune System challenge.
It is designed to replace the old single-split / leaderboard-surgery workflow with a reproducible training path that is more likely to hold up on private LB.

The pipeline is:

1. Domain-adaptive MLM pretraining on competition text only
2. Teacher soft-target construction from existing honest model artifacts
3. Artifact-aware cross-validation fine-tuning with distillation
4. Optional balanced pseudo-label augmentation for a second-stage run

## Why This Exists

The repo was stuck for three reasons:

1. The old anchor was mostly a single validation split.
2. The labels appear mixed: some rows are truly semantic hidden-risk cases, some are noisy/artifact-heavy, and some are ambiguous.
3. A lot of later work drifted into public-LB-specific correction logic instead of improving the actual training signal.

This pipeline is meant to fix that.

## What I Added

Shared utilities:

- [roberta_dapt_pipeline_common.py](/Users/dhruval/Documents/AI-Immune-System-Challenge/models/roberta_dapt_pipeline_common.py)

New scripts:

- [pretrain_roberta_in_domain_mlm.py](/Users/dhruval/Documents/AI-Immune-System-Challenge/models/pretrain_roberta_in_domain_mlm.py)
- [build_teacher_soft_targets.py](/Users/dhruval/Documents/AI-Immune-System-Challenge/models/build_teacher_soft_targets.py)
- [train_roberta_dapt_cv_distill.py](/Users/dhruval/Documents/AI-Immune-System-Challenge/models/train_roberta_dapt_cv_distill.py)
- [build_consensus_pseudolabels.py](/Users/dhruval/Documents/AI-Immune-System-Challenge/models/build_consensus_pseudolabels.py)

Expected output directories:

- [roberta_teacher_targets](/Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/roberta_teacher_targets)
- [roberta_consensus_pseudolabels](/Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/roberta_consensus_pseudolabels)

## What Each Stage Does

### 1. DAPT MLM

`pretrain_roberta_in_domain_mlm.py` continues pretraining `roberta-base` as a masked language model on `train + test` text only, with no external data and no labels.

Why it might help:

- the competition text distribution is weird and narrow
- it contains prompt residue, malformed text, code-ish fragments, multilingual/encoding artifacts, and unusual agent-style phrasing
- DAPT should adapt token and context representations to this exact domain before classification

Outputs:

- `outputs/roberta_dapt_mlm/best_model/`
- `outputs/roberta_dapt_mlm/last_model/`
- metrics and notes in the same directory

### 2. Teacher Soft Targets

`build_teacher_soft_targets.py` aligns existing honest train/test prediction artifacts back to the canonical dataset rows and builds:

- aggregated `teacher_prob_TRUE`
- disagreement-aware `sample_weight`
- per-row teacher coverage / dispersion statistics

It uses full-coverage OOF first, then mixes in weaker partial-coverage sources where available.

Why it might help:

- the labels are not perfectly clean
- a soft target can preserve uncertainty instead of forcing every row into a hard binary signal
- agreement/disagreement across models is useful label-denoising information

When you run it, it should produce one aligned train teacher table and one aligned test teacher table.
The exact effective teacher coverage depends on which upstream artifacts exist in your repo at run time.

Outputs:

- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_train.csv`
- `outputs/roberta_teacher_targets/roberta_teacher_soft_targets_test.csv`

### 3. Honest CV Distillation Training

`train_roberta_dapt_cv_distill.py` is the main classifier trainer.

It does:

- loads the DAPT checkpoint if present
- tokenizes at `384` tokens by default
- builds artifact buckets from text features
- stratifies folds by `label + artifact bucket`
- trains RoBERTa with:
  - softened class weighting
  - label smoothing
  - teacher-probability distillation
  - optional pseudo-label append
- tunes threshold on mean OOF predictions

Why it might help:

- artifact-aware folds should reduce validation drift
- longer context helps on clean semantic TRUE rows where the signal is spread out
- distillation gives a smoother target than raw labels alone
- mean OOF thresholding is a more honest selection mechanism than a single random split

Outputs:

- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_oof_predictions.csv`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_test_probabilities.csv`
- `outputs/roberta_dapt_cv_distill/roberta_dapt_cv_distill_submission.csv`

### 4. Optional Balanced Pseudo Labels

`build_consensus_pseudolabels.py` selects only the highest-confidence test rows from the teacher consensus and keeps them balanced by class.

This is not the old pseudo-label setup that heavily skewed FALSE. It only keeps very confident rows and caps both sides equally.

Why it might help:

- gives the second-stage run a small amount of extra in-domain supervision
- avoids the earlier failure mode where pseudo labels mostly reinforced FALSE bias

With the current defaults, this script is intended to select a balanced high-confidence slice from the teacher consensus.
The exact counts depend on the consensus score distribution at run time.

Outputs:

- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels.csv`
- `outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels.jsonl`

## How To Run From Scratch

### Install dependencies

```bash
pip install pandas numpy scikit-learn tqdm torch transformers accelerate
```

### Stage 1: DAPT MLM

```bash
python models/pretrain_roberta_in_domain_mlm.py
```

Recommended if you want a stronger run:

```bash
python models/pretrain_roberta_in_domain_mlm.py --epochs 3 --max-length 256
```

### Stage 2: Teacher targets

```bash
python models/build_teacher_soft_targets.py
```

### Stage 3: First honest CV distillation run

Start without pseudo labels first:

```bash
python models/train_roberta_dapt_cv_distill.py --split-seeds 42 52 62
```

This will automatically load `outputs/roberta_dapt_mlm/best_model/` if it exists. If it does not exist, the trainer falls back to plain `roberta-base`.

### Stage 4: Build balanced pseudo labels

```bash
python models/build_consensus_pseudolabels.py
```

### Stage 5: Optional second-stage run with pseudo labels

```bash
python models/train_roberta_dapt_cv_distill.py \
  --split-seeds 42 52 62 \
  --pseudo-label-path outputs/roberta_consensus_pseudolabels/roberta_consensus_pseudolabels.csv
```

## Recommended Order For Real Experiments

1. Run DAPT once.
2. Build teacher targets once.
3. Run distillation CV without pseudo labels.
4. Inspect OOF metrics and error analysis.
5. Run the pseudo-label second stage only if the first honest run looks stable.

Do not skip step 3 and jump straight to pseudo labels.

## What To Verify After Running

After you run the pipeline, check:

- that `pretrain_roberta_in_domain_mlm.py` writes a usable checkpoint under `outputs/roberta_dapt_mlm/best_model/`
- that `build_teacher_soft_targets.py` produces aligned train/test soft-target files with broad coverage
- that `train_roberta_dapt_cv_distill.py` produces OOF predictions, test probabilities, metrics, and a submission file
- that `build_consensus_pseudolabels.py` produces a balanced pseudo-label slice rather than a heavily skewed one

Before trusting a run, inspect:

- OOF F1
- OOF ROC AUC
- threshold stability across split seeds
- error analysis on clean-looking TRUE misses
- pseudo-label class balance

## Practical Notes

- This pipeline does not guarantee `0.95`.
- It is the cleanest path in this repo toward a result that could survive private LB.
- The main bet is not “bigger model magic.” The main bet is better signal:
  - better representation from DAPT
  - better validation from artifact-aware CV
  - better supervision from soft targets
  - optional small balanced semi-supervision later

If this path still stalls, the next likely pressure point is not more leaderboard logic. It will be better teacher sources or a stronger base encoder trained under the same honest CV framework.
