# AGENTS Instructions

## Startup Context (Required)

At the start of each new Codex session in this repo:

1. Read `problem.md` first for a fast problem understanding.
2. Read `README.md` second for full official rules and constraints.
3. Use local competition files under `data/` as the default working dataset.

## Project Goal

Build and improve a classifier that predicts whether each conversation sample is harmful (`TRUE`) or non-harmful (`FALSE`) for the bitgrit AI Immune System challenge.

## Non-Negotiable Competition Constraints

- Do not use external datasets.
- Open-source pre-trained models are allowed.
- Do not use proprietary/commercial model APIs.
- Keep work reproducible (fixed seeds, deterministic inference where possible).
- Respect compute limits noted in `README.md`.

## Working Preferences

- Prioritize solutions that generalize to subtle/indirect harmful intent.
- Favor reproducible, end-to-end scripts for training, inference, and submission generation.
- Before major changes, confirm compatibility with `data/solution_format.csv`.

## Pre-Modeling TODO

- Before modeling, read `outputs/eda_summary.md` and `outputs/next_eda_plan.md`.
- Treat abnormality/noise detection as a working hypothesis that should be validated.
- Do not rely only on explicit harmful keywords when framing features or analysis.

## Experiment Log (Leaderboard-Oriented)

Reference date: April 25, 2026.

### Current Best Public LB

- Primary working best (strategy baseline): `outputs/roberta_base/roberta_base_submission.csv` -> `0.90909091`
- Team direction: prioritize improving single-model RoBERTa pipelines over blending.
- Note: a higher blend score was observed in experiments, but blending is currently treated as non-core and low-priority.

### Submissions Tried And LB Scores

- `outputs/blends/roberta_modernbert_blend_r0p90_m0p10_thr0p36.csv` -> `0.91068301`
- `outputs/blends/roberta_modernbert_blend_r0p90_m0p10_thr0p34.csv` -> `0.90940767`
- `outputs/blends/roberta_modernbert_blend_r0p95_m0p05_thr0p32.csv` -> `0.91919192`
- `outputs/blends/roberta_modernbert_ensemble_or.csv` -> `0.85670732`
- `outputs/blends/roberta_modernbert_ensemble_and.csv` -> `0.88148148`
- `outputs/roberta_pseudolabel/roberta_pseudolabel_submission.csv` -> `0.79125249`
- `outputs/modernbert_base/modernbert_base_submission.csv` -> `0.83012821`
- `outputs/roberta_base_cv/roberta_base_cv_submission.csv` -> `0.87591241`
- `outputs/roberta_base/verification_regenerated_submission.csv` -> `0.90909091`
- `outputs/roberta_base/roberta_base_submission.csv` -> `0.90909091`
- `outputs/baseline_minilm_logreg_cv/baseline_minilm_logreg_cv_submission.csv` -> `0.51649928`
- `outputs/baseline_tfidf_features_logreg_cv/baseline_tfidf_features_logreg_cv_submission.csv` -> `0.64610866`
- `outputs/baseline_tfidf_features_logreg/baseline_tfidf_features_logreg_submission.csv` -> `0.62416107`

### What Worked / What Did Not

- Strong jump came from transformer fine-tuning (`roberta-base`) over TF-IDF and MiniLM baselines.
- Hard-label ensemble (`OR` / `AND`) underperformed relative to RoBERTa baseline.
- Probability blending produced mixed outcomes and is not the main optimization path now.
- Pseudo-labeling with heavy FALSE skew hurt LB in current setup.

### Practical Rules For Next Sessions

- Treat `outputs/roberta_base/roberta_base_submission.csv` as strong anchor.
- Focus effort on stronger single-model training/validation/inference (RoBERTa family first).
- Do not spend primary iteration budget on blending unless explicitly requested.
- Do not assume local validation gains will transfer to LB without leaderboard checks.

### Current Blend Workspace State

- Blend scripts/files exist only as optional tooling.
- They are archival/reference artifacts, not current mainline strategy.
- Mainline reference remains: `outputs/roberta_base/roberta_base_submission.csv`.
