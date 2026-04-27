# Mega Ensemble Ranker Notes

No training was performed. This script creates fixed-TRUE-count submissions by combining existing prediction artifacts.

## Strategy

- Keep `roberta_base` as the anchor.
- Combine model rankings rather than raw probabilities, because different models are calibrated differently.
- Exclude known bad branches by default: `roberta_large`, `roberta_pseudolabel`, `baseline_minilm_logreg_cv`, and hard OR/AND ensembles.
- Generate candidates around the known good TRUE-count region.

## First Candidates To Try

If using submissions from this folder, start with:

- `mega_anchor_rank_light_top577.csv`
- `mega_anchor_rank_light_top578.csv`
- `mega_anchor_rank_calibrator_top577.csv`
- `mega_anchor_rank_calibrator_top578.csv`

Do not submit all candidates blindly. Check the manifest first.

## Outputs

- `outputs/mega_ensemble_ranker/mega_ensemble_manifest.csv`
- `outputs/mega_ensemble_ranker/mega_ensemble_scores.csv`
- `outputs/mega_ensemble_ranker/mega_ensemble_report.json`
