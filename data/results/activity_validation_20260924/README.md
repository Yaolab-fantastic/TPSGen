# Activity validation workspace

This directory is part of the TPSGen release evidence, not a package-native
40-promoter design run. It contains the record-level activity evaluation used
for the main-text Figure 2C/D panels.

This directory is reserved for the observed-versus-predicted validation used to
prepare the replacement panels C and D of the main figure.

## Current status

- The existing `main_figure_run_20260912` is a package-native design run. Its
  scores are heuristic scores and must not be used as TransVAE-MLP activity
  predictions.
- The 40 input sequences in that run are labelled `training_promoter_*` and
  are not an independent held-out test set.
- The original held-out validation file was recovered from
  `/data/zhoujie/Paper/MpraVAE/data/vaedata/val.csv` and contains 3,703 records.
- Predictions were generated with the bundled best-correlation TransVAE-MLP
  checkpoint. The resulting Pearson correlations are 0.7801, 0.8237, 0.8019
  and 0.7924 for root, stem, leaf and fruit, respectively. These values match
  the original training log at the checkpoint-selection epoch.
- These records form a held-out validation set, not an independent external
  test set. Figure labels and manuscript text must use that terminology.

The original source path in the manifest is retained for provenance. The
record-level export is included here so the reported correlations can be
audited without access to the original absolute path.

## Files

```text
validation_promoters.fasta   sequences from the held-out validation file
observed_expression.csv      four observed expression-associated values
predicted_expression.csv     four checkpoint-derived scores
observed_vs_predicted.csv    merged plotting table
validation_metrics.csv       Pearson r, R-squared and n by tissue
validation_manifest.json     source and backend provenance
```

`prepare_validation.py` recreates the FASTA and observed table from the source
validation file. After running `tpsgen predict-transvae`,
`summarize_validation.py` merges the outputs and recalculates the metrics.
