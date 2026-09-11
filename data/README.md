# Data Layout

This repository keeps publication-facing data in four layers so the software package remains inspectable without placing every large optional artifact in git.

## Directories

- `raw/`
  - project input tables and support assets for TransVAE, DNABERT-derived processing and preGAN expression-constraint smoke testing
  - enough to regenerate the retained manuscript figures and motif-processing outputs
- `processed/`
  - intermediate outputs produced by migrated workflows
  - currently centered on the `DNABERT` motif post-processing tables
- `results/`
  - repository-native outputs generated from the unified toolkit
  - includes demo command outputs, SVG figures and retained manuscript result bundles
- `external/`
  - manifests for large files intentionally left outside git
  - these remain candidates for Zenodo, release assets, or an optional download script
- `../models/`
  - bundled TransVAE, DNABERT and preGAN resources used by explicit model-backed routes

## Boundary with `outputs/`

The `data/` tree stores repository-curated artifacts that are meant to remain stable across reruns and support manuscript reproduction.

The root-level `outputs/` directory serves a different purpose:

- it is the default scratch space for local CLI runs
- it may contain temporary checks, exploratory exports, or user-specific runs
- it is not the canonical location for repository-bundled reference results

## Reproducing manuscript-facing data

Run:

```bash
make reproduce-results
```

This command regenerates the retained result pack used by the Application Note figures and supplementary materials under `data/results/reproducible_legacy/`.

## preGAN training data boundary

`data/raw/pregan_expression/pregan_smoke.csv` is a small `realA`/`realB`/`expr` table for testing mask semantics and training-code plumbing. In this format, `realA` contains fixed DNA bases plus `M` symbols marking mutable positions, `realB` contains the completed 165-bp promoter sequence, and `expr` contains the scalar expression target used by the preGAN expression-constraint workflow.

The full masked-promoter training table is not bundled in normal git history or in the wheel. It is registered in `data/external/external_resources.tsv` so full retraining can be distributed separately with checksums, provenance metadata and applicable redistribution terms.

For the broader data-sync utility used during repository maintenance, run:

```bash
PYTHONPATH=src python scripts/sync_repository_data.py
```

This script copies selected raw files into `data/raw`, refreshes processed tables, regenerates demo outputs, regenerates bundled figures, and updates `data/inventory.tsv`.

## Inventory files

- `inventory.tsv`
  - machine-readable list of bundled data files and their provenance
- `source_registry.tsv`
  - higher-level registry of project origins, license scope and redistribution notes
- `summary.json`
  - compact count summary by stage
- `external/external_resources.tsv`
  - optional large artifacts intentionally excluded from normal git history

## Documentation

- `docs/tool_documentation.md`
  - tool usage, manuscript reproduction, data resources, model boundaries and provenance notes
