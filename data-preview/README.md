# Data Preview

Local preview files split from `data/train_labeled_comp.jsonl` for manual reading.

No test data is included.

## Files

- `train_true.md`: readable TRUE examples (1400 rows)
- `train_false.md`: readable FALSE examples (3500 rows)
- `train_true.csv` / `train_false.csv`: spreadsheet-friendly versions with original row idx
- `train_true.jsonl` / `train_false.jsonl`: machine-readable versions with original row idx

Use the `idx` field to trace any example back to the original train JSONL row.
