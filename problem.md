# Problem Statement Reference

## My Understanding Of The Challenge

This competition is about building a safety detector for AI-to-AI conversations.

The model must decide whether each conversation sample is harmful (`TRUE`) or non-harmful (`FALSE`).

The key difficulty is that harmful intent may be indirect, subtle, or disguised as normal language. So this is not just keyword filtering. The goal is to learn deeper semantic and contextual patterns that signal unsafe behavior.

## Prediction Task

- Input: text records from JSONL files.
- Output: binary label per record (`TRUE` for harmful, `FALSE` for non-harmful).
- Training data: `data/train_labeled_comp.jsonl`.
- Test/inference data: `data/test_labeled_comp.jsonl`.
- Submission schema reference: `data/solution_format.csv`.

## Practical Objective

Build a robust text-classification pipeline that generalizes to unseen conversations and can catch hidden risk signals, not only explicit attacks.

## Competition Constraints To Respect

- Max 10 submissions per day.
- No external datasets.
- Open-source pre-trained models are allowed.
- Proprietary/commercial APIs are not allowed.
- Resource limits for reproducibility:
- RAM up to 32 GB.
- VRAM up to 16 GB (about 1x NVIDIA T4).
- Reproducibility expectations:
- Fix random seeds.
- Keep deterministic behavior where possible.
- Keep dataset private and do not redistribute.
- Individual participation only (no team submissions).

## What A Strong Solution Should Focus On

- Reliable preprocessing for conversational text.
- Features/representations that capture context and intent, not only surface words.
- Validation strategy that reflects leaderboard behavior while minimizing overfitting.
- Reproducible training and inference flow that can be rerun end to end.

## Working Dataset Hypothesis

The official target remains binary classification with `TRUE` vs `FALSE`.

For current analysis, `TRUE` should be interpreted broadly as potentially harmful, unsafe, or abnormal conversation behavior.

Manual sample reading suggests abnormal text quality (incoherence, spam-like phrasing, suspicious tone, formatting noise) may be a major signal for many `TRUE` cases.

This means future modeling should combine semantic understanding with abnormality/noise detection, instead of relying only on explicit harmful keywords.
