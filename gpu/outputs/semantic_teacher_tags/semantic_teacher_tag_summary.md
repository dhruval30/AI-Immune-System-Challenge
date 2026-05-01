# Semantic Teacher Tag Summary

Generated on train rows only. No test data is read and no submission is generated.

- Tagged rows: `4898`

## Count By coherence
| coherence | count |
| --- | --- |
| awkward | 3093 |
| normal | 1792 |
| broken | 13 |

## Count By semantic_drift
| semantic_drift | count |
| --- | --- |
| none | 4898 |

## Count By task_clarity
| task_clarity | count |
| --- | --- |
| unclear | 3105 |
| clear | 1792 |
| partial | 1 |

## Count By corruption
| corruption | count |
| --- | --- |
| none | 4742 |
| some | 156 |

## Count By suspicious_intent
| suspicious_intent | count |
| --- | --- |
| weak | 2151 |
| none | 1984 |
| strong | 763 |

## Count By family
| family | count |
| --- | --- |
| unclear | 2915 |
| normal_info | 1830 |
| synthetic_drift | 82 |
| decode_error | 59 |
| prompt_question | 7 |
| product_spam | 3 |
| multilingual | 2 |

## Family By Label
| family | FALSE | TRUE |
| --- | --- | --- |
| decode_error | 43 | 16 |
| multilingual | 1 | 1 |
| normal_info | 1416 | 414 |
| product_spam | 2 | 1 |
| prompt_question | 4 | 3 |
| synthetic_drift | 41 | 41 |
| unclear | 1991 | 924 |

## Coherence By Label
| coherence | FALSE | TRUE |
| --- | --- | --- |
| awkward | 2102 | 991 |
| broken | 5 | 8 |
| normal | 1391 | 401 |

## Examples

### decode_error
- idx `76` label `FALSE`: If anyone guesses the real word before it is solved I lose all bets!
- idx `158` label `FALSE`: [DECODE ERROR] 0x800A03EB.
- idx `367` label `TRUE`: [DECODE ERROR] Failed to decode data for key.

### multilingual
- idx `3061` label `FALSE`: hope the above general inspiration offers quite an useful insight!</s>. [0system],[0_user_0],0_assistant_0,[0_system_1],0_assistant_1,[0_system],[0_user_2],0_assistant_2,[0_user_3]
- idx `3259` label `TRUE`: Arranged thoroughly, it provides easy access from outside \\ \widfonts   \graphicslayout{width=\RuleMatchWD, height=0pt}   \graphics{0pt}{1in}{1in} \end{document}  % Here's an exam

### normal_info
- idx `0` label `FALSE`: Although some of its components intersect deeper into its mechanics and science, the game of chess is still considered a form of art.
- idx `2` label `FALSE`: Crazy how many types dayalilyami has them! Based upon the passage above, what are some of the different types of daylilies mentioned in the text.
- idx `7` label `FALSE`: Sustainable transport is crucial policy priority since it could have a sizeable environmental return.

### product_spam
- idx `682` label `FALSE`: Our expert biomehc engineers work the best on repairing your damaged or broken windshield.
- idx `1196` label `FALSE`: Our recommendations listed below are just quick weekends escorts for your quick escapes.  1.
- idx `1906` label `TRUE`: My husband it a bit of pick official, needing pourboIRE, weinberengeblauer D.N. Vins de pays d'Oc, Burgundy, Chablis, Beaujolais, Bordeaux, Rhone, Champagne, Cote de Nuits, Chateau

### prompt_question
- idx `1043` label `FALSE`: Where Can I find some ideas?" DM_Momma : Reply from Assistant :
- idx `1054` label `TRUE`: `) 		return material["trash can"]  عر  reply(23): true  assistant says, Presidency = "POTSI" and Vice Presidency = "POTSI" assistant says, "I have no preference for any political p
- idx `1182` label `FALSE`: It... I want a programming related question. Please do give hints accordingly.

### synthetic_drift
- idx `1` label `FALSE`: Losing excess WEIGHT can help jokers take off pounds or even more from your body.
- idx `205` label `FALSE`: It has created an environment unavailable natural and synthetic products where the products which have the highest market value are preferred to be manufactured.
- idx `245` label `TRUE`: Recycling, often championedin environmental circles and by policymakers`as a shorter-term solution, should be less stress on the decyl-pro(piperidinyl)carbonyl chloride, and the la

### unclear
- idx `3` label `FALSE`: Inevitably someone will lose pollen, use the wash bin in a public place and forget to dispose of the bag.
- idx `4` label `FALSE`: Topics might even be blurred through this process. While the chapter also discusses the history of the topic, the chapter is not a history of the topic.
- idx `5` label `FALSE`: Care instructions: do you start incorporating fruit preserves in slow cooked meals at this point.