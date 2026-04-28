# RoBERTa + ELECTRA Probability Blend Notes

## Summary

- No training was performed.
- This script only blends existing test probabilities.
- This is probability blending, not a router.
- Blend sources:
- `outputs/roberta_base/roberta_base_test_probabilities.csv`
- `outputs/electra_base/electra_base_test_probabilities.csv`

## Why This Blend

RoBERTa is the known strong anchor. ELECTRA may add diversity because its discriminator-style pretraining can react differently to synthetic, corrupted, or unnatural text artifacts.

## Blend Grid

- This script intentionally generates only five selected candidates.
- Candidates: `[{'name': 'count577_r0p90_thr0p34', 'roberta_weight': 0.9, 'threshold': 0.34}, {'name': 'count577_r0p85_thr0p34', 'roberta_weight': 0.85, 'threshold': 0.34}, {'name': 'count578_r0p95_thr0p32', 'roberta_weight': 0.95, 'threshold': 0.32}, {'name': 'count576_r0p95_thr0p34', 'roberta_weight': 0.95, 'threshold': 0.34}, {'name': 'count578_r0p80_thr0p34', 'roberta_weight': 0.8, 'threshold': 0.34}]`
- Total files generated: `5`

## Selection Logic

The candidates keep prediction counts near the known strong RoBERTa distribution (`577 TRUE / 1523 FALSE`) and the best previous blend count (`578 TRUE`).

Use the manifest to choose submission order. Prefer the candidates with the smallest number of changes vs the original RoBERTa submission first.

## Output Files

- Blend submissions: `outputs/blends/roberta_electra_blend_*.csv`
- Manifest: `outputs/blends/roberta_electra_prob_blend_manifest.csv`
