# Next EDA Plan

## Goal

Run a second-pass EDA focused on abnormality/weirdness signals while still respecting the official `TRUE` vs `FALSE` target.

## Planned Analyses

### Lexical Weirdness

- Unique word ratio
- Average word length
- Rare-looking token counts (e.g., very long tokens, unusual character mixtures)
- Digits/symbols ratio
- Non-alphanumeric ratio

### Structure And Formatting

- Newline count
- Punctuation density
- URL count
- Quote/code-like formatting markers
- Repeated token patterns

### Fluency Proxies

- Sentence length distribution
- Very short sentence counts and very long sentence counts
- Stopword ratio (only if implemented locally without external downloads)
- Simple readability proxies (length and sentence-structure based)

### Class Comparison

- Compare all above features by `TRUE` and `FALSE`
- Inspect top extreme examples per feature and per class
- Save summary tables directly to `outputs/`

### Qualitative Tagging

- Manually review around 30 `TRUE` and 30 `FALSE` samples
- Tag into rough buckets for insight:
  - coherent
  - incoherent
  - spam-like
  - aggressive/suspicious
  - technical
  - travel/product/normal content
- Use these tags only for interpretation, not as new training labels

## Notes

- Keep this pass EDA-only (no model training, no submission creation).
- Treat abnormality/noise as a working hypothesis to test, not a final rule.
