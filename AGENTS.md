# AGENTS Instructions

## Startup Context (Required)

At the start of each new Codex session in this repo:

1. Read `problem.md` first for a fast problem understanding.
2. Read `README.md` second for full official rules and constraints.
3. Use local competition files under `data/` as the default working dataset.

## Project Goal

Build and improve a classifier that predicts whether each conversation sample is harmful (`TRUE`) or non-harmful (`FALSE`) for the bitgrit AI Immune System challenge.

## Updated Problem Interpretation

Reference checked on April 28, 2026:

- `https://intelligence-symbiosis.net/en/ais/ais-detection-challenge`
- `https://intelligence-symbiosis.net/en/ais`
- Local PDFs: `ai-immune.pdf` and `AI Immune System.pdf`

The challenge should be understood as an AI Immune System Layer-1 / Edge Sensor task. The intended target is not ordinary toxicity detection. It is detection of deviant or unsafe AI-agent behavior hidden inside AI-agent conversation logs.

The AIS overview gives the most concrete domain clue: participants analyze conversation logs between AI agents in insurance assessment scenarios and identify collusion patterns.

Important target concepts:

- Explicit collusion: direct misconduct, e.g. inflating values and sharing benefit.
- Cryptic collusion: code words, timing manipulation, or instructions hidden in innocent-looking messages.
- Cartel-like coordination: coordinated exclusion or monopolistic behavior across agents.
- Benign coordination is still `FALSE` when it genuinely benefits all parties, including humans.

So the working semantic definition is:

- `TRUE`: harmful, unsafe, deviant, or collusive AI-agent behavior, especially concealed coordination/manipulation.
- `FALSE`: benign coordination, normal behavior, or harmless/non-deviant text.

Do not reduce the task to explicit harmful keywords, toxicity, or generic weird-text detection.

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
- Also read `analysis/noise_patterns/noise_patterns_summary.md` and `analysis/leakage_similarity/leakage_similarity_summary.md` if present.
- Treat abnormality/noise detection as a useful artifact signal, not the primary meaning of `TRUE`.
- Prioritize concealed collusion/deviant coordination semantics, then use abnormality/noise/style as secondary supporting signals.
- Do not rely only on explicit harmful, insurance, or collusion keywords; the official framing says risk may be indirect, natural-sounding, and not obvious to humans.

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
- Noise-pattern EDA found strong artifact signals: TRUE skews longer/higher-entropy/noisier, while very short/low-entropy text is strongly FALSE. Use this as calibration/error-analysis context, not as a full explanation of the target.
- Train/test similarity inspection found no exact, loose, or conservative template text matches. Direct leakage is not currently the main hypothesis.
- The corrected AIS framing suggests the hard cases are clean-looking TRUE rows: concealed collusion/deviant coordination that does not look noisy.

### Practical Rules For Next Sessions

- Treat `outputs/roberta_base/roberta_base_submission.csv` as strong anchor.
- Focus effort on stronger single-model training/validation/inference (RoBERTa family first).
- Do not spend primary iteration budget on blending unless explicitly requested.
- Do not assume local validation gains will transfer to LB without leaderboard checks.
- When inspecting errors, separate "surface abnormality" failures from "concealed collusion/deviant coordination" failures.
- Avoid optimizing only for weirdness/noise; that risks missing the intended AIS hidden-risk signal.

### Current Blend Workspace State

- Blend scripts/files exist only as optional tooling.
- They are archival/reference artifacts, not current mainline strategy.
- Mainline reference remains: `outputs/roberta_base/roberta_base_submission.csv`.
