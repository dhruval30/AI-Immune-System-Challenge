# RoBERTa Hard + Bootstrap Blend Notes

No training is performed. This is pure probability blending.

## Sources

- Previous hard-weighted run: `/Users/dhruval/Documents/AI-Immune-System-Challenge/gpu/outputs/roberta_hard_weighted_gpu_b64_final/roberta_hard_weighted_gpu_b64_test_probabilities.csv`
- Bootstrapped hard-weighted run: `/Users/dhruval/Documents/AI-Immune-System-Challenge/gpu/outputs/roberta_hard_weighted_bootstrap_gpu/roberta_hard_weighted_bootstrap_gpu_test_probabilities.csv`

## Current Best Anchor

- `roberta_hard_weighted_bootstrap_gpu_submission_thr0p32.csv`
- Public LB: `0.93718166`

## Candidate Logic

- Blend probabilities only, not hard labels.
- Use fixed threshold `0.32` because it is the best known LB-calibrated threshold.
- Keep only five candidates.
- Weight bootstrapped model at `0.50`, `0.60`, `0.70`, `0.80`, `0.90`.

Recommended first submissions: `boot070_hard030`, then `boot080_hard020`, then `boot060_hard040`.

Manifest: `/Users/dhruval/Documents/AI-Immune-System-Challenge/gpu/outputs/roberta_hard_bootstrap_blend/roberta_hard_bootstrap_blend_manifest.csv`