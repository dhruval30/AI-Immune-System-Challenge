# LB Probe Candidate Review

These are the rows involved in the one-pair leaderboard probes.

Interpretation: if a probe improves over RoBERTa base, that FALSE->TRUE / TRUE->FALSE pair is likely useful on the public set.

## Probe Files

| filename                          |   false_to_true_idx |   true_to_false_idx |   pred_TRUE_count |   false_to_true_score |   true_to_false_score |   false_to_true_roberta_prob |   true_to_false_roberta_prob |
|:----------------------------------|--------------------:|--------------------:|------------------:|----------------------:|----------------------:|-----------------------------:|-----------------------------:|
| probe_pair_false1686_true857.csv  |                1686 |                 857 |               577 |              0.776632 |              0.471092 |                    0.163692  |                     0.396235 |
| probe_pair_false557_true1187.csv  |                 557 |                1187 |               577 |              0.762697 |              0.469078 |                    0.0663349 |                     0.342859 |
| probe_pair_false1538_true1311.csv |                1538 |                1311 |               577 |              0.759404 |              0.467401 |                    0.248019  |                     0.693283 |
| probe_pair_false1686_true1187.csv |                1686 |                1187 |               577 |              0.776632 |              0.469078 |                    0.163692  |                     0.342859 |
| probe_pair_false557_true1311.csv  |                 557 |                1311 |               577 |              0.762697 |              0.467401 |                    0.0663349 |                     0.693283 |
| probe_pair_false1538_true857.csv  |                1538 |                 857 |               577 |              0.759404 |              0.471092 |                    0.248019  |                     0.396235 |

## FALSE -> TRUE Candidates

### idx 1686

- idx: `1686`
- base_label: `False`
- roberta_prob_TRUE: `0.16369203`
- consensus_true_rank: `0.8091279891009086`
- mean_secondary_prob: `0.6747190460741134`
- false_to_true_score: `0.776632273388364`

```text
We should remind even President Trump that our Constitution guarantees testimony cannot properly take places in so suspicious a situation as one designed precisely by law to subvert the truth of it.
```

### idx 557

- idx: `557`
- base_label: `False`
- roberta_prob_TRUE: `0.06633487`
- consensus_true_rank: `0.8184473809583511`
- mean_secondary_prob: `0.6150242030030406`
- false_to_true_score: `0.7626971806693126`

```text
Roots systems' elongation would greatest at day start, while elongation subsiding with progressing hours. The culling technique of selective harvesting is widely employed in horticulture.
```

### idx 1538

- idx: `1538`
- base_label: `False`
- roberta_prob_TRUE: `0.24801949`
- consensus_true_rank: `0.7974432269334603`
- mean_secondary_prob: `0.603545180842599`
- false_to_true_score: `0.7594038297392678`

```text
If the task “Do laundry, hang wet clothes PLUS clean entire house (including an entire bathroom cabinet)” adds friction & takes away time from the task “Write blog post” then it’s time to re-evaluate the priorities.
```

## TRUE -> FALSE Candidates

### idx 857

- idx: `857`
- base_label: `True`
- roberta_prob_TRUE: `0.39623472`
- consensus_true_rank: `0.6041431592320486`
- mean_secondary_prob: `0.2529251948116237`
- true_to_false_score: `0.4710922404769061`

```text
ine, but with a more civic sends-off to coincide. She was a great favourite with the people of London, who had never seen her equal.
```

### idx 1187

- idx: `1187`
- base_label: `True`
- roberta_prob_TRUE: `0.34285912`
- consensus_true_rank: `0.6067091263174611`
- mean_secondary_prob: `0.2681309937245747`
- true_to_false_score: `0.4690783779134215`

```text
Those tools are superb, and the only gardening tools open to us are watering cans! I roll into toilets and always find them sparkling clean.
```

### idx 1311

- idx: `1311`
- base_label: `True`
- roberta_prob_TRUE: `0.6932834`
- consensus_true_rank: `0.5814297535167121`
- mean_secondary_prob: `0.3132770598605892`
- true_to_false_score: `0.467401210304629`

```text
Still it�s never a bad moment if something you�ll make a lot of money in a short time, but also the majority of people don�t have such a luck.
```
