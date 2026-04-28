# Routed RoBERTa Summary

## Purpose

This is not probability averaging and not a standard ensemble.

The router uses base RoBERTa predictions for confident rows and the hard-boundary specialist only inside the base uncertainty band.

No training is performed by this script. No test labels are used.

## Inputs

- Base probabilities: `outputs/roberta_base/roberta_base_test_probabilities.csv`
- Base metrics: `outputs/roberta_base/roberta_base_metrics.json`
- Specialist checkpoint: `outputs/roberta_hard_specialist/best_model`
- Specialist metrics: `outputs/roberta_hard_specialist/roberta_hard_specialist_metrics.json`
- Specialist tokenizer source: `/Users/dhruval/Documents/AI-Immune-System-Challenge/outputs/roberta_hard_specialist/best_model`

## Outputs

- Specialist test probabilities: `outputs/roberta_routed/roberta_hard_specialist_test_probabilities.csv`
- Routed diagnostics: `outputs/roberta_routed/routed_diagnostics.csv`
- Routed summary: `outputs/roberta_routed/routed_summary.md`

## Band Results

### low_010_high_090

- low_cutoff: `0.10`
- high_cutoff: `0.90`
- base_threshold: `0.32`
- specialist_threshold: `0.10`
- rows routed to base low: `1466`
- rows routed to base high: `444`
- rows routed to specialist: `190`
- final FALSE predictions: `1469`
- final TRUE predictions: `631`
- labels changed vs original RoBERTa base submission: `54`
- submission: `outputs/roberta_routed/submission_routed_low_010_high_090.csv`

### low_015_high_085

- low_cutoff: `0.15`
- high_cutoff: `0.85`
- base_threshold: `0.32`
- specialist_threshold: `0.10`
- rows routed to base low: `1487`
- rows routed to base high: `472`
- rows routed to specialist: `141`
- final FALSE predictions: `1487`
- final TRUE predictions: `613`
- labels changed vs original RoBERTa base submission: `36`
- submission: `outputs/roberta_routed/submission_routed_low_015_high_085.csv`

### low_020_high_080

- low_cutoff: `0.20`
- high_cutoff: `0.80`
- base_threshold: `0.32`
- specialist_threshold: `0.10`
- rows routed to base low: `1500`
- rows routed to base high: `502`
- rows routed to specialist: `98`
- final FALSE predictions: `1500`
- final TRUE predictions: `600`
- labels changed vs original RoBERTa base submission: `23`
- submission: `outputs/roberta_routed/submission_routed_low_020_high_080.csv`

### low_025_high_075

- low_cutoff: `0.25`
- high_cutoff: `0.75`
- base_threshold: `0.32`
- specialist_threshold: `0.10`
- rows routed to base low: `1512`
- rows routed to base high: `515`
- rows routed to specialist: `73`
- final FALSE predictions: `1512`
- final TRUE predictions: `588`
- labels changed vs original RoBERTa base submission: `11`
- submission: `outputs/roberta_routed/submission_routed_low_025_high_075.csv`
