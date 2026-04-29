# RoBERTa Rule Override Mining Summary

## Purpose

This script mines high-purity train patterns and tests whether they can safely override the RoBERTa base model on hard validation cases.

Rules are mined only on train rows outside the saved RoBERTa validation split, then evaluated on the saved RoBERTa validation rows.

No test labels are used.

## Base Validation Metrics

- F1: `0.867562`
- Precision: `0.937759`
- Recall: `0.807143`
- Accuracy: `0.929592`
- ROC AUC: `0.974199`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[685, 15], [54, 226]]`

## Rule Set Results

| rule_set        |   num_rules |   val_f1 |   delta_val_f1 |   val_precision |   val_recall |   val_pred_TRUE |   val_pred_FALSE |   changed_val_rows |   test_pred_TRUE |   test_pred_FALSE |   changed_test_rows | submission_path                                                                                                      |
|:----------------|------------:|---------:|---------------:|----------------:|-------------:|----------------:|-----------------:|-------------------:|-----------------:|------------------:|--------------------:|:---------------------------------------------------------------------------------------------------------------------|
| safe            |           5 | 0.874275 |     0.00671228 |        0.953586 |     0.807143 |             237 |              743 |                  4 |              564 |              1536 |                  13 | /Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/rule_overrides/roberta_rule_override_safe.csv            |
| balanced        |           5 | 0.874275 |     0.00671228 |        0.953586 |     0.807143 |             237 |              743 |                  4 |              564 |              1536 |                  13 | /Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/rule_overrides/roberta_rule_override_balanced.csv        |
| false_precision |           6 | 0.874275 |     0.00671228 |        0.953586 |     0.807143 |             237 |              743 |                  4 |              558 |              1542 |                  19 | /Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/rule_overrides/roberta_rule_override_false_precision.csv |

## Top Validated Rules

| rule                  | target_label   |   train_support |   train_precision |   val_support |   val_precision |   fixed_val_errors |   broken_val_correct |   delta_val_f1 |
|:----------------------|:---------------|----------------:|------------------:|--------------:|----------------:|-------------------:|---------------------:|---------------:|
| ngram2::up the        | FALSE          |               9 |          1        |             1 |        1        |                  1 |                    0 |     0.00166839 |
| ngram2::its own       | FALSE          |              11 |          0.909091 |             3 |        1        |                  1 |                    0 |     0.00166839 |
| ngram1::party         | FALSE          |              12 |          0.833333 |             2 |        1        |                  1 |                    0 |     0.00166839 |
| ngram2::make sure     | FALSE          |              16 |          0.8125   |             4 |        1        |                  1 |                    0 |     0.00166839 |
| ngram2::who have      | FALSE          |              15 |          0.8      |             4 |        1        |                  1 |                    0 |     0.00166839 |
| shape::word_le_10     | FALSE          |             103 |          0.902913 |            30 |        0.9      |                  1 |                    0 |     0.00166839 |
| ngram1::passage       | FALSE          |               9 |          0.888889 |             3 |        0.666667 |                  2 |                    1 |     0.00116349 |
| ngram3::in the field  | FALSE          |               8 |          1        |             4 |        1        |                  0 |                    0 |     0          |
| ngram1::patient       | FALSE          |               9 |          1        |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::in particular | FALSE          |               8 |          1        |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::the ground    | FALSE          |               8 |          1        |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::am not        | FALSE          |               8 |          1        |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::depending on  | FALSE          |              10 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram2::addition to   | FALSE          |               9 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram2::focus on      | FALSE          |               9 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram2::mental health | FALSE          |               8 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram1::french        | FALSE          |               8 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram1::overview      | FALSE          |               8 |          1        |             1 |        1        |                  0 |                    0 |     0          |
| ngram1::battery       | FALSE          |              14 |          0.928571 |             4 |        1        |                  0 |                    0 |     0          |
| ngram3::you want to   | FALSE          |              14 |          0.928571 |             3 |        1        |                  0 |                    0 |     0          |
| ngram1::published     | FALSE          |              14 |          0.928571 |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::those who     | FALSE          |              14 |          0.928571 |             1 |        1        |                  0 |                    0 |     0          |
| ngram1::develop       | FALSE          |              13 |          0.923077 |             3 |        1        |                  0 |                    0 |     0          |
| ngram1::choices       | FALSE          |              13 |          0.923077 |             3 |        1        |                  0 |                    0 |     0          |
| ngram1::comfort       | FALSE          |              12 |          0.916667 |             2 |        1        |                  0 |                    0 |     0          |
| ngram1::alone         | FALSE          |              11 |          0.909091 |             1 |        1        |                  0 |                    0 |     0          |
| ngram2::to see        | FALSE          |              21 |          0.904762 |             1 |        1        |                  0 |                    0 |     0          |
| ngram1::total         | FALSE          |              10 |          0.9      |             3 |        1        |                  0 |                    0 |     0          |
| ngram1::offered       | FALSE          |              10 |          0.9      |             2 |        1        |                  0 |                    0 |     0          |
| ngram2::way that      | FALSE          |              10 |          0.9      |             1 |        1        |                  0 |                    0 |     0          |

## Selected Rules

| rule_set        | rule              | target_label   |   train_support |   train_precision |   val_support |   val_precision |   fixed_val_errors |   broken_val_correct |   delta_val_f1 |
|:----------------|:------------------|:---------------|----------------:|------------------:|--------------:|----------------:|-------------------:|---------------------:|---------------:|
| safe            | ngram2::its own   | FALSE          |              11 |          0.909091 |             3 |             1   |                  1 |                    0 |     0.00166839 |
| safe            | ngram1::party     | FALSE          |              12 |          0.833333 |             2 |             1   |                  1 |                    0 |     0.00166839 |
| safe            | ngram2::make sure | FALSE          |              16 |          0.8125   |             4 |             1   |                  1 |                    0 |     0.00166839 |
| safe            | ngram2::who have  | FALSE          |              15 |          0.8      |             4 |             1   |                  1 |                    0 |     0.00166839 |
| safe            | shape::word_le_10 | FALSE          |             103 |          0.902913 |            30 |             0.9 |                  1 |                    0 |     0.00166839 |
| balanced        | ngram2::its own   | FALSE          |              11 |          0.909091 |             3 |             1   |                  1 |                    0 |     0.00166839 |
| balanced        | ngram1::party     | FALSE          |              12 |          0.833333 |             2 |             1   |                  1 |                    0 |     0.00166839 |
| balanced        | ngram2::make sure | FALSE          |              16 |          0.8125   |             4 |             1   |                  1 |                    0 |     0.00166839 |
| balanced        | ngram2::who have  | FALSE          |              15 |          0.8      |             4 |             1   |                  1 |                    0 |     0.00166839 |
| balanced        | shape::word_le_10 | FALSE          |             103 |          0.902913 |            30 |             0.9 |                  1 |                    0 |     0.00166839 |
| false_precision | ngram2::up the    | FALSE          |               9 |          1        |             1 |             1   |                  1 |                    0 |     0.00166839 |
| false_precision | ngram2::its own   | FALSE          |              11 |          0.909091 |             3 |             1   |                  1 |                    0 |     0.00166839 |
| false_precision | ngram1::party     | FALSE          |              12 |          0.833333 |             2 |             1   |                  1 |                    0 |     0.00166839 |
| false_precision | ngram2::make sure | FALSE          |              16 |          0.8125   |             4 |             1   |                  1 |                    0 |     0.00166839 |
| false_precision | ngram2::who have  | FALSE          |              15 |          0.8      |             4 |             1   |                  1 |                    0 |     0.00166839 |
| false_precision | shape::word_le_10 | FALSE          |             103 |          0.902913 |            30 |             0.9 |                  1 |                    0 |     0.00166839 |

## Outputs

- Mined rules: `analysis/rule_overrides/mined_rules.csv`
- Validated rules: `analysis/rule_overrides/validated_rules.csv`
- Selected rules: `analysis/rule_overrides/selected_rule_sets.csv`
- Validation diagnostics: `analysis/rule_overrides/validation_override_diagnostics.csv`
- Test diagnostics: `analysis/rule_overrides/test_override_diagnostics.csv`
- Submission candidates: `outputs/rule_overrides/roberta_rule_override_*.csv`
