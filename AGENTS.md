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
