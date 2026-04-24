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

