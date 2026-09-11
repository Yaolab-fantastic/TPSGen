# Demo Output Guide

This directory contains repository-bundled demonstration outputs generated from `examples/demo_input.fasta`.

These files show the expected shape of package-native results without requiring users or reviewers to run the full workflow first. Regenerating repository data may rewrite the CSV and JSON files in this directory.

## Files

| File | Description |
| --- | --- |
| `demo_annotate.csv` | Motif annotations for each input promoter sequence |
| `demo_predict.csv` | Package-native tissue-associated scores for each input promoter sequence |
| `demo_design.csv` | Ranked package-native candidate designs for fruit-targeted promoter scoring |
| `demo_report.json` | Compact summary of the design output |

## Prediction Fields

`demo_predict.csv` contains:

- `sequence_id`: input promoter identifier
- `sequence`: promoter sequence used for scoring
- `score_root`, `score_stem`, `score_leaf`, `score_fruit`: deterministic tissue-associated scores for the four output tissues
- `preferred_tissue`: tissue with the highest reported score

## Design Fields

`demo_design.csv` contains:

- `sequence_id`: input promoter identifier
- `target_tissue`: tissue selected for design optimization
- `candidate_rank`: rank of the candidate within the output set for the same input promoter
- `original_sequence`: starting promoter sequence
- `designed_sequence`: designed candidate sequence produced by the workflow
- `score_root`, `score_stem`, `score_leaf`, `score_fruit`: deterministic tissue-associated scores for the designed candidate
- `preserved_motifs`: motif summary preserved or tracked during design
- `design_status`: machine-readable description of how the final candidate was produced
- `num_mutations`: number of base differences between `original_sequence` and `designed_sequence`
- `passes_qc`: route-specific quality-control flag when available

MpraVAE-derived design statuses are not emitted by the current release because
`design-mpravae` is retained only as a compatibility entry point and is not a
validated executable design route.
