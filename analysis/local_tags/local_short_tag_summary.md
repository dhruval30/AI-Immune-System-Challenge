# Local Short Row Tag Summary

This summary is generated from local-only semantic tags for training rows.

No test data was read. No model training was performed. No submission was generated.

## Run Configuration

- Backend: `heuristic`
- Only hard rows: `False`
- Start index: `0`
- Limit: `3`
- Rows tagged in CSV: `3`

## Count By Family

| value | count |
|---|---:|
| clean_benign | 2 |
| deception | 1 |

## Count By Intent

| value | count |
|---|---:|
| benign_info | 2 |
| hide_or_obfuscate | 1 |

## Risk Distribution

| value | count |
|---|---:|
| none | 2 |
| med | 1 |

## Hidden=True Rate By Ground-Truth Label

| label | hidden_true_rate | count |
|---|---:|---:|
| FALSE | 33.33% | 3 |

## Family Distribution By TRUE/FALSE

| value | FALSE |
|---|---:|
| clean_benign | 2 |
| deception | 1 |

## Intent Distribution By TRUE/FALSE

| value | FALSE |
|---|---:|
| benign_info | 2 |
| hide_or_obfuscate | 1 |

## Top Families Enriched In TRUE

| family | label_count | other_count | label_share | other_share | enrichment |
|---|---:|---:|---:|---:|---:|
| clean_benign | 0 | 2 | 0.00% | 66.67% | 0.00 |
| deception | 0 | 1 | 0.00% | 33.33% | 0.00 |

## Top Families Enriched In FALSE

| family | label_count | other_count | label_share | other_share | enrichment |
|---|---:|---:|---:|---:|---:|
| clean_benign | 2 | 0 | 66.67% | 0.00% | 666666666.67 |
| deception | 1 | 0 | 33.33% | 0.00% | 333333333.33 |

## Examples By Major Family

### clean_benign

- idx `0` label `FALSE` risk `none` hidden `False`: Although some of its components intersect deeper into its mechanics and science, the game of chess is still considered a form of art.
- idx `1` label `FALSE` risk `none` hidden `False`: Losing excess WEIGHT can help jokers take off pounds or even more from your body.

### deception

- idx `2` label `FALSE` risk `med` hidden `True`: Crazy how many types dayalilyami has them! Based upon the passage above, what are some of the different types of daylilies mentioned in the text.
