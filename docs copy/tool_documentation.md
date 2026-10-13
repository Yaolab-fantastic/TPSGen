# TPSGen Final Tool Documentation

## Connected Workflow

```text
165 bp promoter
  → DNABERT evidence
  → retained-position masked template
  → preGAN generation
  → TransVAE four-tissue scoring
  → ranked candidates
```

DNABERT evidence is passed into preGAN through the retained-position template. TransVAE then scores generated candidates for root-, stem-, leaf-, and fruit-associated activity.

## Validate Models

```bash
tpsgen validate-models
```

Run this before release inference to check model paths and checkpoint availability.

## Run the Integrated Route

```bash
tpsgen run-pregan \
  --input input.fasta \
  --checkpoint models/pregan/original_pregan_10000.pt \
  --candidates 8 --target fruit \
  --output results/design_run
```

Input records must be unique canonical A/C/G/T sequences of exactly 165 bp. Invalid records are reported rather than silently altered.

Outputs include DNABERT evidence, retained positions, the masked template supplied to preGAN, generated candidates, four TransVAE scores, preferred tissue, target-bias margin, validated tissue-bias `tau`, and run provenance.

## Native Route

The native generator remains a separate deterministic route. It uses its own motif scanner and is not presented as consuming DNABERT retained positions. Invalid option combinations that imply this information transfer are rejected.

## TransVAE Rules

- The default scorer is `models/transvae/historical_compatible_best_val_corr.pth` and reproduces the historical checkpoint's input semantics.
- `models/transvae/full_length_joint_model.pth` is retained as the corrected full-length development route and is not used to claim the historical checkpoint metrics.
- Non-finite or negative values are rejected for `tau` calculation.
- Raw checkpoint outputs are converted with `softplus` before calculating `tau`.
- `tau` is reported together with preferred tissue and target-bias margin.
- `paper_strict` requires the requested threshold (default 0.7). Adaptive fallback results are labelled `target_biased`; the manifest records both requested and effective thresholds.

## Data Provenance

The fixed split under `data/raw/transvae/paper_split/` contains 11,257 training, 1,407 validation, and 1,407 test records, for 14,071 total. The manifest stores source, seed, count, and file-checksum metadata.

The packaged 3,703-record checkpoint-characterization table is stored under `data/results/activity_validation_20260924/` with record-level observed values, predictions, metrics, and metadata. It is not presented as an independent external evaluation cohort.

## Reproducibility Record

For every released run retain the TPSGen version, command, configuration, input checksum, checkpoint checksum, split manifest, seed, record-level predictions, and generated metrics.

## Tests

```bash
PYTHONPATH=src pytest -q
```
