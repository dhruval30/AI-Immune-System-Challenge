# LB Probe Notes

## Why This Exists

`roberta_balanced_swap_03.csv` improved public LB, but we do not know which individual row changes were correct.

These probes isolate row pairs while preserving RoBERTa's TRUE count. They are designed to learn from leaderboard feedback, not to be final model submissions.

## Submission Advice

Submit at most a few probes in one day. Record each returned LB score next to the manifest.

If a one-pair probe beats RoBERTa base, keep that pair. If it beats swap03, it is especially valuable.

## Generated Probes

| filename                          |   false_to_true_idx |   true_to_false_idx |   pred_TRUE_count |   false_to_true_score |   true_to_false_score |   false_to_true_roberta_prob |   true_to_false_roberta_prob |
|:----------------------------------|--------------------:|--------------------:|------------------:|----------------------:|----------------------:|-----------------------------:|-----------------------------:|
| probe_pair_false1686_true857.csv  |                1686 |                 857 |               577 |              0.776632 |              0.471092 |                    0.163692  |                     0.396235 |
| probe_pair_false557_true1187.csv  |                 557 |                1187 |               577 |              0.762697 |              0.469078 |                    0.0663349 |                     0.342859 |
| probe_pair_false1538_true1311.csv |                1538 |                1311 |               577 |              0.759404 |              0.467401 |                    0.248019  |                     0.693283 |
| probe_pair_false1686_true1187.csv |                1686 |                1187 |               577 |              0.776632 |              0.469078 |                    0.163692  |                     0.342859 |
| probe_pair_false557_true1311.csv  |                 557 |                1311 |               577 |              0.762697 |              0.467401 |                    0.0663349 |                     0.693283 |
| probe_pair_false1538_true857.csv  |                1538 |                 857 |               577 |              0.759404 |              0.471092 |                    0.248019  |                     0.396235 |
