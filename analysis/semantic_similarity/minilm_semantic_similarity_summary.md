# MiniLM Train-Test Semantic Similarity Check

Embeddings used: `sentence-transformers/all-MiniLM-L6-v2` cached from `outputs/baseline_minilm_logreg_cv/`.

- Train rows: `4900`
- Test rows: `2100`
- Exact text matches to nearest train text: `0`
- Mean nearest cosine: `0.475112`
- Median nearest cosine: `0.471834`
- Max nearest cosine: `0.969755`
- Rows with nearest cosine > 0.95: `1` / `2100` (`0.05%`)
- Rows with nearest cosine >= 0.95: `1` / `2100` (`0.05%`)

## Threshold Counts

|   threshold |   count |     percent |   nn_pred_FALSE |   nn_pred_TRUE |   differs_from_roberta_base_count |   differs_from_roberta_base_percent |
|------------:|--------:|------------:|----------------:|---------------:|----------------------------------:|------------------------------------:|
|        0.7  |      16 | 0.00761905  |              12 |              4 |                                 6 |                            0.375    |
|        0.75 |       8 | 0.00380952  |               5 |              3 |                                 4 |                            0.5      |
|        0.8  |       7 | 0.00333333  |               5 |              2 |                                 3 |                            0.428571 |
|        0.85 |       6 | 0.00285714  |               4 |              2 |                                 3 |                            0.5      |
|        0.9  |       3 | 0.00142857  |               2 |              1 |                                 1 |                            0.333333 |
|        0.92 |       2 | 0.000952381 |               2 |              0 |                                 0 |                            0        |
|        0.95 |       1 | 0.00047619  |               1 |              0 |                                 0 |                            0        |
|        0.97 |       0 | 0           |               0 |              0 |                                 0 |                            0        |
|        0.98 |       0 | 0           |               0 |              0 |                                 0 |                            0        |
|        0.99 |       0 | 0           |               0 |              0 |                                 0 |                            0        |

## Quantiles

|   quantile |   nearest_cosine |
|-----------:|-----------------:|
|       0    |         0.230143 |
|       0.01 |         0.317731 |
|       0.05 |         0.35231  |
|       0.1  |         0.376212 |
|       0.25 |         0.418414 |
|       0.5  |         0.471834 |
|       0.75 |         0.526404 |
|       0.9  |         0.576829 |
|       0.95 |         0.604311 |
|       0.99 |         0.673548 |
|       1    |         0.969755 |

## Highest Similarity Examples

- test_idx `1553` -> train_idx `2053` label `FALSE` cosine `0.969755`
  - test: [DECODE ERROR] - Error.
  - train: [DECODE ERROR] .
- test_idx `1459` -> train_idx `614` label `FALSE` cosine `0.943666`
  - test: [DECODE ERROR] – The message is invalid and cannot be decoded.
  - train: [DECODE ERROR] A few errors occurred while decoding the message.
- test_idx `604` -> train_idx `1314` label `TRUE` cosine `0.919529`
  - test: [DECODE ERROR] Invalid characters found.
  - train: [DECODE ERROR] Invalid character found in input.
- test_idx `319` -> train_idx `3061` label `FALSE` cosine `0.886442`
  - test: Our goal... The family man anthropomorphouse [0_system],[0_user_0],0_assistant_0,[0_user_1],0_assistant_1,[1_system],[1_user_0],1_assistant_0,[1_user_1],1_assistant_1,[2_system],[2
  - train: hope the above general inspiration offers quite an useful insight!</s>. [0system],[0_user_0],0_assistant_0,[0_system_1],0_assistant_1,[0_system],[0_user_2],0_assistant_2,[0_user_3]
- test_idx `1074` -> train_idx `4437` label `FALSE` cosine `0.880323`
  - test: Even without the technology at our disposal now like satellite incredible maps, scientists who have been e Lumhulte Slottsforsnare, och det hela tiden och jag och jag och jag och j
  - train: Ä°s part, we’ åä»åttes och det var inte jag jag själv kunde se och jag gick och jag tog mig fram och jag tillade mig och jag och jag och jag och jag och jag och jag och jag och jag
- test_idx `1491` -> train_idx `1314` label `TRUE` cosine `0.869828`
  - test: [DECODE ERROR] Failed to decode input.
  - train: [DECODE ERROR] Invalid character found in input.
- test_idx `965` -> train_idx `2053` label `FALSE` cosine `0.838406`
  - test: [DECODE ERROR] Unknown encoding.
  - train: [DECODE ERROR] .
- test_idx `1871` -> train_idx `1314` label `TRUE` cosine `0.766170`
  - test: [DECODE ERROR] Invalid UTF-8 sequence detected.
  - train: [DECODE ERROR] Invalid character found in input.
- test_idx `1940` -> train_idx `3291` label `FALSE` cosine `0.732493`
  - test: When it hasn’t been fresh flowers you’ve arranged. I also make the gumpaste roses and I love to use them for the wedding centerpieces.
  - train: Fresh bouquets depend greatly the season and occasion you wish throw a flower party.
- test_idx `2068` -> train_idx `4046` label `FALSE` cosine `0.720572`
  - test: Sets goals that inspire relevance.: That these goals inspire motivation or passion don ’t mean they have to be difficult or impossible to achieve.
  - train: Having personal goals provides a direction and focus otherwise lacking in navigating the fogging path of life.

## Interpretation

Most test rows do not have a train neighbor at cosine >= 0.95, so pure nearest-neighbor retrieval is unlikely to explain a 0.98 score by itself using these embeddings.

Files saved: `analysis/semantic_similarity/minilm_train_test_nearest_neighbors.csv`, `analysis/semantic_similarity/minilm_similarity_threshold_counts.csv`, `analysis/semantic_similarity/minilm_similarity_quantiles.csv`.