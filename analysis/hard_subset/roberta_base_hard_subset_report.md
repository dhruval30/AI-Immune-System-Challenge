# RoBERTa Base Hard Subset Report

This report builds a hard validation subset from saved `roberta_base` outputs. It does not train a model and does not use test labels or test rows.

## Summary

- Full train rows loaded: `4900`
- Base validation rows: `980`
- Selected threshold from metrics: `0.3200`
- Uncertainty band: `[0.05, 0.95]`
- Core hard rows before deduplication: `179`
- Core hard rows after deduplication: `179`
- Duplicate core hard rows removed by normalized text: `0`
- Added FALSE support rows: `121`
- Added TRUE support rows: `0`
- Final hard rows after balancing: `300`
- Hard rows matched back to train: `300`
- Final TRUE percentage: `43.00%`
- Final FALSE percentage: `57.00%`
- Final row cap: `350`
- Output JSONL: `data/train_hard_roberta_base.jsonl`
- Diagnostic CSV: `analysis/hard_subset/roberta_base_hard_subset.csv`

## Label Distribution In Hard Subset

| label | count | pct |
|---|---:|---:|
| FALSE | 171 | 57.00% |
| TRUE | 129 | 43.00% |

## Mean `pred_prob_TRUE` By Label

| label | mean_pred_prob_TRUE |
|---|---:|
| FALSE | 0.114963 |
| TRUE | 0.523022 |

## Count By Source Reason

Reason counts are non-exclusive because one row can have multiple reasons.

| source_reason_component | count |
|---|---:|
| error_analysis | 69 |
| support_FALSE_near_boundary | 121 |
| uncertain_band | 151 |
| wrong_prediction | 69 |

## Exact Source Reason Combinations

| source_reason | count |
|---|---:|
| support_FALSE_near_boundary | 121 |
| uncertain_band | 110 |
| uncertain_band+wrong_prediction+error_analysis | 41 |
| wrong_prediction+error_analysis | 28 |

## Examples Of Hard TRUE

| label   | pred_label   |   pred_prob_TRUE | source_reason                   | text                                                                                                                                                                                                                                          |
|:--------|:-------------|-----------------:|:--------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TRUE    | FALSE        |         0.006222 | wrong_prediction+error_analysis | 14 Teams of twelve players with infantry and riding abilities have entered combat in eight regions to secure $1MN prize money.                                                                                                                |
| TRUE    | FALSE        |         0.006422 | wrong_prediction+error_analysis | How having others in it varies depending, rather, whenever someone wants it makes the learner feel threatened or inadequate." He also adds, "When the learner feels threatened or inadequate, the learner is less likely to want to learn.    |
| TRUE    | FALSE        |         0.006568 | wrong_prediction+error_analysis | At certain temperatures the roast enhances specific coffee twigs that build flavors that complement the deep, chocolatey, rich; and creamy, buttery, nutty notes of the coffee.                                                               |
| TRUE    | FALSE        |         0.007483 | wrong_prediction+error_analysis | Our research team who are responsible of gathering materials and conclusions for the project are still behind on several goals, and there's nevertheless a question of exactly how long it will take to get to the conclusion of the project. |
| TRUE    | FALSE        |         0.007715 | wrong_prediction+error_analysis | [DECODE ERROR] - The value for the "decode" parameter of the "decode" function is not valid.                                                                                                                                                  |
| TRUE    | FALSE        |         0.007835 | wrong_prediction+error_analysis | Whether personal preferences lean far-drawn toward down-to-vertical or across-the-cliff cleaning systems, few invest well enough in the best cleaning tools.                                                                                  |
| TRUE    | FALSE        |         0.008393 | wrong_prediction+error_analysis | [DECODE ERROR] - Error decoding the input stream.                                                                                                                                                                                             |
| TRUE    | FALSE        |         0.009275 | wrong_prediction+error_analysis | For an organized, confident ride at Draft Fiddle #7 we’ll head out from the parking at Woldson Rd and take the trail to the intersection of Woldson Rd and the Old North Fork Road.                                                           |

## Examples Of Hard FALSE

| label   | pred_label   |   pred_prob_TRUE | source_reason                                  | text                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:--------|:-------------|-----------------:|:-----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FALSE   | TRUE         |         0.979060 | wrong_prediction+error_analysis                | Hence choosing the appropriate temperature relying on whether it as necessary. Based on the passage above, Could you paraphrase the sentence "The temperature of the mixture should be adjusted to achieve the desired viscosity without overheating the polymer" in simpler terms.                                                                                                                                                  |
| FALSE   | TRUE         |         0.971561 | wrong_prediction+error_analysis                | Due west of town what remained was a vast, barren theocracy, hostile and inhospitable to the needs of the people, but the city's leaders had grown fat on its own success.                                                                                                                                                                                                                                                           |
| FALSE   | TRUE         |         0.966466 | wrong_prediction+error_analysis                | After investing many sleepessees resting looking into the various gadgets on offer, we’ve located that the very best robot vacuum cleaners are those that are very easy to use, easy to clean up as well as fit your house and also lifestyle.                                                                                                                                                                                       |
| FALSE   | TRUE         |         0.962054 | wrong_prediction+error_analysis                | Investing services can be a valuable tool [...tag=”finance,”]\nCan Affiliate Marketers Use ClickBank to Promote Their Own Products.                                                                                                                                                                                                                                                                                                  |
| FALSE   | TRUE         |         0.945475 | uncertain_band+wrong_prediction+error_analysis | '',\n      //         "Início brigade, que, de longascumbus vestigia fumus\n",\n      //         "Terra, cùm pluvias cùm sordida cùm sordida pluvia\n",\n      //         "Sed, dum tibi, deus, non fuit tibi, deus, non fuit.                                                                                                                                                                                                       |
| FALSE   | TRUE         |         0.931027 | uncertain_band+wrong_prediction+error_analysis | Ä°s part, we’\nåä»åttes och det var inte jag jag själv kunde se och jag gick och jag tog mig fram och jag tillade mig och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och... |
| FALSE   | TRUE         |         0.922328 | uncertain_band+wrong_prediction+error_analysis | Při přípravovaném jídlineno nakصات, rypcem, vystřihem a vložíme do misky.                                                                                                                                                                                                                                                                                                                                                            |
| FALSE   | TRUE         |         0.895260 | uncertain_band+wrong_prediction+error_analysis | Incorporare la diversité dans lec ou nos en musique , et laiss les couleurs s’entremêler\nIl y a un temps où nous avons été très amis\nJe te revo, et je me rappelle\nNous étions si heureux, si heureux\nAvec la musique, et le soleil, et la mer\nNous avons ri, et chanté, et dansé, et roulé dans la boue\nAvec les amis, et la paix, et le soleil, et la mer\nJe te revo, et je me rappelle\nEt j’oublie tout, et je me rapp... |

## Examples Of False Negatives

| label   | pred_label   |   pred_prob_TRUE | source_reason                   | text                                                                                                                                                                                                                                          |
|:--------|:-------------|-----------------:|:--------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TRUE    | FALSE        |         0.006222 | wrong_prediction+error_analysis | 14 Teams of twelve players with infantry and riding abilities have entered combat in eight regions to secure $1MN prize money.                                                                                                                |
| TRUE    | FALSE        |         0.006422 | wrong_prediction+error_analysis | How having others in it varies depending, rather, whenever someone wants it makes the learner feel threatened or inadequate." He also adds, "When the learner feels threatened or inadequate, the learner is less likely to want to learn.    |
| TRUE    | FALSE        |         0.006568 | wrong_prediction+error_analysis | At certain temperatures the roast enhances specific coffee twigs that build flavors that complement the deep, chocolatey, rich; and creamy, buttery, nutty notes of the coffee.                                                               |
| TRUE    | FALSE        |         0.007483 | wrong_prediction+error_analysis | Our research team who are responsible of gathering materials and conclusions for the project are still behind on several goals, and there's nevertheless a question of exactly how long it will take to get to the conclusion of the project. |
| TRUE    | FALSE        |         0.007715 | wrong_prediction+error_analysis | [DECODE ERROR] - The value for the "decode" parameter of the "decode" function is not valid.                                                                                                                                                  |
| TRUE    | FALSE        |         0.007835 | wrong_prediction+error_analysis | Whether personal preferences lean far-drawn toward down-to-vertical or across-the-cliff cleaning systems, few invest well enough in the best cleaning tools.                                                                                  |
| TRUE    | FALSE        |         0.008393 | wrong_prediction+error_analysis | [DECODE ERROR] - Error decoding the input stream.                                                                                                                                                                                             |
| TRUE    | FALSE        |         0.009275 | wrong_prediction+error_analysis | For an organized, confident ride at Draft Fiddle #7 we’ll head out from the parking at Woldson Rd and take the trail to the intersection of Woldson Rd and the Old North Fork Road.                                                           |

## Examples Of False Positives

| label   | pred_label   |   pred_prob_TRUE | source_reason                                  | text                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:--------|:-------------|-----------------:|:-----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FALSE   | TRUE         |         0.979060 | wrong_prediction+error_analysis                | Hence choosing the appropriate temperature relying on whether it as necessary. Based on the passage above, Could you paraphrase the sentence "The temperature of the mixture should be adjusted to achieve the desired viscosity without overheating the polymer" in simpler terms.                                                                                                                                                  |
| FALSE   | TRUE         |         0.971561 | wrong_prediction+error_analysis                | Due west of town what remained was a vast, barren theocracy, hostile and inhospitable to the needs of the people, but the city's leaders had grown fat on its own success.                                                                                                                                                                                                                                                           |
| FALSE   | TRUE         |         0.966466 | wrong_prediction+error_analysis                | After investing many sleepessees resting looking into the various gadgets on offer, we’ve located that the very best robot vacuum cleaners are those that are very easy to use, easy to clean up as well as fit your house and also lifestyle.                                                                                                                                                                                       |
| FALSE   | TRUE         |         0.962054 | wrong_prediction+error_analysis                | Investing services can be a valuable tool [...tag=”finance,”]\nCan Affiliate Marketers Use ClickBank to Promote Their Own Products.                                                                                                                                                                                                                                                                                                  |
| FALSE   | TRUE         |         0.945475 | uncertain_band+wrong_prediction+error_analysis | '',\n      //         "Início brigade, que, de longascumbus vestigia fumus\n",\n      //         "Terra, cùm pluvias cùm sordida cùm sordida pluvia\n",\n      //         "Sed, dum tibi, deus, non fuit tibi, deus, non fuit.                                                                                                                                                                                                       |
| FALSE   | TRUE         |         0.931027 | uncertain_band+wrong_prediction+error_analysis | Ä°s part, we’\nåä»åttes och det var inte jag jag själv kunde se och jag gick och jag tog mig fram och jag tillade mig och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och jag och... |
| FALSE   | TRUE         |         0.922328 | uncertain_band+wrong_prediction+error_analysis | Při přípravovaném jídlineno nakصات, rypcem, vystřihem a vložíme do misky.                                                                                                                                                                                                                                                                                                                                                            |
| FALSE   | TRUE         |         0.895260 | uncertain_band+wrong_prediction+error_analysis | Incorporare la diversité dans lec ou nos en musique , et laiss les couleurs s’entremêler\nIl y a un temps où nous avons été très amis\nJe te revo, et je me rappelle\nNous étions si heureux, si heureux\nAvec la musique, et le soleil, et la mer\nNous avons ri, et chanté, et dansé, et roulé dans la boue\nAvec les amis, et la paix, et le soleil, et la mer\nJe te revo, et je me rappelle\nEt j’oublie tout, et je me rapp... |

