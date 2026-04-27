# Noise Pattern Discovery Summary

This analysis derives text features and searches for supported label-correlated patterns. It does not train a model.

## Dataset

- Rows: `4900`
- TRUE rows: `1400`
- FALSE rows: `3500`
- Base TRUE rate: `0.2857`
- Minimum single-pattern support: `80`
- Minimum combined-rule support: `80`

## Strongest Feature Correlations

Positive correlation means higher values are associated with TRUE.

| feature                 |   mean_FALSE |   mean_TRUE |   pearson_corr_with_TRUE |   cohen_d_TRUE_minus_FALSE |   approx_p_value | direction    |
|:------------------------|-------------:|------------:|-------------------------:|---------------------------:|-----------------:|:-------------|
| token_entropy           |       4.1398 |      4.5783 |                   0.3726 |                     0.9057 |           0.0000 | higher_TRUE  |
| char_length             |     132.8809 |    188.0093 |                   0.3044 |                     0.6724 |           0.0000 | higher_TRUE  |
| byte_length             |     133.6171 |    188.9714 |                   0.3028 |                     0.6708 |           0.0000 | higher_TRUE  |
| word_count              |      22.2489 |     30.9521 |                   0.2695 |                     0.5806 |           0.0000 | higher_TRUE  |
| raw_token_count         |      21.8437 |     30.6464 |                   0.2684 |                     0.5690 |           0.0000 | higher_TRUE  |
| compression_ratio       |       0.8459 |      0.7716 |                  -0.2631 |                    -0.6195 |           0.0000 | higher_FALSE |
| char_entropy            |       4.1873 |      4.2756 |                   0.2248 |                     0.5061 |           0.0000 | higher_TRUE  |
| repeated_token_types    |       1.7609 |      2.6486 |                   0.1705 |                     0.3607 |           0.0000 | higher_TRUE  |
| max_word_length         |      10.4797 |     11.3300 |                   0.1529 |                     0.3239 |           0.0000 | higher_TRUE  |
| rare_token_ratio        |       0.1064 |      0.1397 |                   0.1467 |                     0.3303 |           0.0000 | higher_TRUE  |
| sentence_count          |       1.6643 |      2.2264 |                   0.1439 |                     0.3030 |           0.0000 | higher_TRUE  |
| max_sentence_word_count |      18.8483 |     22.3436 |                   0.1380 |                     0.2867 |           0.0000 | higher_TRUE  |
| hapax_token_ratio       |       0.0724 |      0.0981 |                   0.1338 |                     0.2983 |           0.0000 | higher_TRUE  |
| long_sentence_count     |       0.0931 |      0.1693 |                   0.1079 |                     0.2270 |           0.0000 | higher_TRUE  |
| stopword_ratio          |       0.3935 |      0.3694 |                  -0.0912 |                    -0.2076 |           0.0000 | higher_FALSE |

## Top 10 Patterns Predicting TRUE

| condition                                    |   support_count |   support_pct |   true_rate |   false_rate |   lift_TRUE |   lift_FALSE |   approx_p_value |   confidence |
|:---------------------------------------------|----------------:|--------------:|------------:|-------------:|------------:|-------------:|-----------------:|-------------:|
| token_entropy >= p95 (5)                     |             246 |        0.0502 |      0.6911 |       0.3089 |      2.4187 |       0.4325 |           0.0000 |       1.0000 |
| token_entropy >= p90 (4.875)                 |             502 |        0.1024 |      0.6434 |       0.3566 |      2.2520 |       0.4992 |           0.0000 |       1.0000 |
| token_entropy in quantile bin (4.685, 6.523] |             980 |        0.2000 |      0.5878 |       0.4122 |      2.0571 |       0.5771 |           0.0000 |       1.0000 |
| char_length >= p90 (220)                     |             492 |        0.1004 |      0.6118 |       0.3882 |      2.1413 |       0.5435 |           0.0000 |       1.0000 |
| raw_token_count >= p95 (41)                  |             260 |        0.0531 |      0.6269 |       0.3731 |      2.1942 |       0.5223 |           0.0000 |       1.0000 |
| byte_length >= p90 (221)                     |             494 |        0.1008 |      0.6053 |       0.3947 |      2.1184 |       0.5526 |           0.0000 |       1.0000 |
| byte_length in quantile bin (188.0, 1163.0]  |             979 |        0.1998 |      0.5598 |       0.4402 |      1.9591 |       0.6163 |           0.0000 |       1.0000 |
| char_length in quantile bin (188.0, 1163.0]  |             959 |        0.1957 |      0.5600 |       0.4400 |      1.9599 |       0.6161 |           0.0000 |       1.0000 |
| raw_token_count >= p90 (36)                  |             508 |        0.1037 |      0.5984 |       0.4016 |      2.0945 |       0.5622 |           0.0000 |       1.0000 |
| word_count >= p95 (42)                       |             250 |        0.0510 |      0.6200 |       0.3800 |      2.1700 |       0.5320 |           0.0000 |       1.0000 |

## Top 10 Patterns Predicting FALSE

| condition                                     |   support_count |   support_pct |   true_rate |   false_rate |   lift_TRUE |   lift_FALSE |   approx_p_value |   confidence |
|:----------------------------------------------|----------------:|--------------:|------------:|-------------:|------------:|-------------:|-----------------:|-------------:|
| byte_length in quantile bin (15.999, 96.0]    |            1001 |        0.2043 |      0.0400 |       0.9600 |      0.1399 |       1.3441 |           0.0000 |       1.0000 |
| char_length in quantile bin (15.999, 96.0]    |            1017 |        0.2076 |      0.0413 |       0.9587 |      0.1445 |       1.3422 |           0.0000 |       1.0000 |
| byte_length <= p10 (81)                       |             491 |        0.1002 |      0.0367 |       0.9633 |      0.1283 |       1.3487 |           0.0000 |       1.0000 |
| char_length <= p05 (71)                       |             256 |        0.0522 |      0.0352 |       0.9648 |      0.1230 |       1.3508 |           0.0000 |       1.0000 |
| char_length <= p10 (81)                       |             507 |        0.1035 |      0.0394 |       0.9606 |      0.1381 |       1.3448 |           0.0000 |       1.0000 |
| byte_length <= p05 (71)                       |             248 |        0.0506 |      0.0363 |       0.9637 |      0.1270 |       1.3492 |           0.0000 |       1.0000 |
| word_count in quantile bin (-0.001, 16.0]     |            1129 |        0.2304 |      0.0576 |       0.9424 |      0.2015 |       1.3194 |           0.0000 |       1.0000 |
| raw_token_count in quantile bin (0.999, 16.0] |            1175 |        0.2398 |      0.0604 |       0.9396 |      0.2115 |       1.3154 |           0.0000 |       1.0000 |
| word_count <= p10 (14)                        |             645 |        0.1316 |      0.0558 |       0.9442 |      0.1953 |       1.3219 |           0.0000 |       1.0000 |
| token_entropy in quantile bin (-0.001, 3.875] |            1009 |        0.2059 |      0.0743 |       0.9257 |      0.2602 |       1.2959 |           0.0000 |       1.0000 |

## Top Combined TRUE Rules

| condition                                                                                       |   support_count |   support_pct |   true_rate |   false_rate |   lift_TRUE |   lift_FALSE |   approx_p_value |   confidence |
|:------------------------------------------------------------------------------------------------|----------------:|--------------:|------------:|-------------:|------------:|-------------:|-----------------:|-------------:|
| (token_entropy >= p90 (4.875)) AND (sentence_count >= p90 (3))                                  |             107 |        0.0218 |      0.7477 |       0.2523 |      2.6168 |       0.3533 |           0.0000 |       1.0000 |
| (token_entropy >= p90 (4.875)) AND (sentence_count in quantile bin (2.0, 49.0])                 |             107 |        0.0218 |      0.7477 |       0.2523 |      2.6168 |       0.3533 |           0.0000 |       1.0000 |
| (token_entropy in quantile bin (4.685, 6.523]) AND (sentence_count >= p90 (3))                  |             183 |        0.0373 |      0.7268 |       0.2732 |      2.5437 |       0.3825 |           0.0000 |       1.0000 |
| (token_entropy in quantile bin (4.685, 6.523]) AND (sentence_count in quantile bin (2.0, 49.0]) |             183 |        0.0373 |      0.7268 |       0.2732 |      2.5437 |       0.3825 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (char_length >= p90 (220))                                       |             212 |        0.0433 |      0.7075 |       0.2925 |      2.4764 |       0.4094 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (raw_token_count in quantile bin (30.2, 222.0])                  |             242 |        0.0494 |      0.7025 |       0.2975 |      2.4587 |       0.4165 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (raw_token_count >= p95 (41))                                    |             145 |        0.0296 |      0.7103 |       0.2897 |      2.4862 |       0.4055 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (byte_length >= p90 (221))                                       |             212 |        0.0433 |      0.7028 |       0.2972 |      2.4599 |       0.4160 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (byte_length in quantile bin (188.0, 1163.0])                    |             238 |        0.0486 |      0.6975 |       0.3025 |      2.4412 |       0.4235 |           0.0000 |       1.0000 |
| (token_entropy >= p95 (5)) AND (raw_token_count >= p90 (36))                                    |             219 |        0.0447 |      0.6986 |       0.3014 |      2.4452 |       0.4219 |           0.0000 |       1.0000 |

## Top Combined FALSE Rules

| condition                                                                                      |   support_count |   support_pct |   true_rate |   false_rate |   lift_TRUE |   lift_FALSE |   approx_p_value |   confidence |
|:-----------------------------------------------------------------------------------------------|----------------:|--------------:|------------:|-------------:|------------:|-------------:|-----------------:|-------------:|
| (rare_token_ratio <= p10 (0)) AND (token_entropy in quantile bin (3.875, 4.17])                |             164 |        0.0335 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio == 0) AND (token_entropy in quantile bin (3.875, 4.17])                      |             164 |        0.0335 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio <= p05 (0)) AND (token_entropy in quantile bin (3.875, 4.17])                |             164 |        0.0335 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio <= p10 (0)) AND (raw_token_count in quantile bin (16.0, 20.0])               |             149 |        0.0304 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio == 0) AND (raw_token_count in quantile bin (16.0, 20.0])                     |             149 |        0.0304 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio <= p05 (0)) AND (raw_token_count in quantile bin (16.0, 20.0])               |             149 |        0.0304 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio <= p10 (0)) AND (word_count in quantile bin (16.0, 20.0])                    |             148 |        0.0302 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio == 0) AND (word_count in quantile bin (16.0, 20.0])                          |             148 |        0.0302 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (rare_token_ratio <= p05 (0)) AND (word_count in quantile bin (16.0, 20.0])                    |             148 |        0.0302 |      0.0000 |       1.0000 |      0.0000 |       1.4000 |           0.0000 |       1.0000 |
| (char_length in quantile bin (15.999, 96.0]) AND (char_entropy in quantile bin (1.745, 4.097]) |             323 |        0.0659 |      0.0031 |       0.9969 |      0.0108 |       1.3957 |           0.0000 |       1.0000 |

## Interpretation

- The strongest supported TRUE patterns are mostly length, repetition, line/formatting, and token-abnormality signals when they appear with enough support.
- The strongest supported FALSE patterns are mostly low-length, low-repetition, high lexical diversity, and absence of structural noise.
- Any pattern above is backed by support count, label rate, lift, and an approximate significance test against the base TRUE rate.
- Patterns with low support are excluded to reduce rare-case overfitting.
- Combined rules should be treated as candidate predictive signals, not ground truth. They are useful when they increase purity while retaining non-trivial support.

## Output Files

- `analysis/noise_patterns/discovered_features.csv`
- `analysis/noise_patterns/feature_label_correlations.csv`
- `analysis/noise_patterns/strong_patterns.csv`
- `analysis/noise_patterns/combined_rules.csv`
- `analysis/noise_patterns/noise_patterns_summary.md`
