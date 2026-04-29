# Balanced RoBERTa Swap Notes

## Purpose

These candidates preserve the RoBERTa base TRUE count while swapping suspicious rows in both directions.

The previous rule override reduced TRUE count and hurt LB. This script avoids that by making balanced swaps.

No training is performed. No test labels are used.

## Signals

- `electra` weight `0.9`
- `modernbert_base` weight `0.7`
- `modernbert_lora_v2` weight `0.8`
- `roberta_style` weight `0.75`
- `label_style_calibrated` weight `0.55`
- `label_style_rankblend_090` weight `0.45`
- `roberta_cv` weight `0.55`
- `roberta_cv_v2` weight `0.55`
- `hard_specialist` weight `0.45`

## Generated Candidates

| filename                     |   swap_count_each_direction |   total_changed_rows |   pred_TRUE_count |   pred_FALSE_count | flip_false_to_true_indices                                               | flip_true_to_false_indices                                               |   mean_false_to_true_score |   mean_true_to_false_score |
|:-----------------------------|----------------------------:|---------------------:|------------------:|-------------------:|:-------------------------------------------------------------------------|:-------------------------------------------------------------------------|---------------------------:|---------------------------:|
| roberta_balanced_swap_03.csv |                           3 |                    6 |               577 |               1523 | 1686 557 1538                                                            | 857 1187 1311                                                            |                   0.766244 |                   0.469191 |
| roberta_balanced_swap_05.csv |                           5 |                   10 |               577 |               1523 | 1686 557 1538 370 263                                                    | 857 1187 1311 236 1987                                                   |                   0.762237 |                   0.459592 |
| roberta_balanced_swap_08.csv |                           8 |                   16 |               577 |               1523 | 1686 557 1538 370 263 1476 653 561                                       | 857 1187 1311 236 1987 1231 849 1446                                     |                   0.75655  |                   0.449401 |
| roberta_balanced_swap_12.csv |                          12 |                   24 |               577 |               1523 | 1686 557 1538 370 263 1476 653 561 1617 1314 1548 384                    | 857 1187 1311 236 1987 1231 849 1446 446 1768 1940 905                   |                   0.750978 |                   0.439683 |
| roberta_balanced_swap_16.csv |                          16 |                   32 |               577 |               1523 | 1686 557 1538 370 263 1476 653 561 1617 1314 1548 384 1468 1442 778 1260 | 857 1187 1311 236 1987 1231 849 1446 446 1768 1940 905 1991 179 504 1190 |                   0.746951 |                   0.432199 |

## Suggested Submission Order

Start small. Submit `roberta_balanced_swap_03.csv` first, then `05` only if it improves.

These files are controlled experiments, not guaranteed improvements.