# TPSGen Tool Documentation

This document is the detailed user and reviewer guide for TPSGen.

TPSGen is a Python command-line framework for tomato promoter analysis and design. It connects promoter sequence validation, motif annotation, motif-conditioned candidate generation, four-tissue scoring, tissue-bias prioritization and reporting within one repository.

## 1. What The Tool Does

TPSGen supports five core tasks:

| Task | Description | Main command |
| --- | --- | --- |
| Input validation | Check FASTA records, identifiers and sequence symbols | `validate-input` |
| Motif annotation | Scan promoter sequences for configured motifs | `annotate` |
| Tissue-associated scoring | Report root, stem, leaf and fruit heuristic scores | `predict` |
| Motif-aware design | Generate candidate promoters for a target tissue | `design` |
| Reporting and figures | Summarize outputs and export simple figures | `report`, `figures` |

The current release also includes the paper-aligned Transformer-VAE/MLP scoring adapter, a bundled DNABERT inference adapter, DNABERT-derived motif post-processing and preGAN generation components. The Transformer-VAE, DNABERT and preGAN resources are bundled under `models/` and in the release wheel. `predict-dnabert` runs checkpoint inference on FASTA input; `annotate-dnabert` is a separate route that consumes matched precomputed attention resources.

The repository also includes an TransVAE training entry point (`scripts/train_transvae.py`), a default training configuration (`configs/training_transvae.yaml`) and training documentation (`docs/training.md`). This means the TransVAE route includes model architecture, training logic, bundled checkpoint loading and downstream scoring commands.

Internal module names and compatibility aliases that include `legacy` refer to adapters retained from earlier project workflows and kept stable for existing scripts. They are supported compatibility routes in this release, not deprecated code paths.

## 2. Release Boundary

The package has three layers:

| Layer | Meaning | Availability |
| --- | --- | --- |
| Package-native modules | Deterministic modules implemented directly in the package | Always runnable after installation |
| Model-backed routes | Explicit checkpoint, precomputed-attention or training-component routes integrated by the project | TransVAE and DNABERT checkpoints are bundled; preGAN generation is available explicitly but is not independently benchmarked or experimentally validated |
| Retained result resources | Curated project tables and regenerated results used for manuscript figures | Distributed as repository data resources |

Together, these layers provide a reproducible framework that integrates project training code, model resources, preprocessing logic and design utilities while retaining route-specific output definitions.

## 3. Installation

From a downloaded release wheel:

```bash
python -m venv .venv
source .venv/bin/activate
pip install tpsgen-0.2.0-py3-none-any.whl
```

The wheel contains the checkpoint and model definition files needed by
`predict-transvae`, `predict-dnabert` and the explicit preGAN generation route.
Package-native commands
do not require PyTorch. Install the optional model dependencies before running
checkpoint-backed or training-component routes. From a published package index,
use:

```bash
pip install "tpsgen[models]"
```

From a local wheel file, use:

```bash
pip install "tpsgen-0.2.0-py3-none-any.whl[models]"
```

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
make test
```

For manuscript-facing result reproduction:

```bash
pip install -e ".[reproduce]"
make reproduce-results
```

## 4. Quick Start

### Integrated workflow (recommended)

Users normally provide one FASTA file and one target tissue. The `run` command
executes motif annotation, motif-preserving candidate generation, scoring of
the original and designed sequences, and report assembly in one reproducible
workflow:

```bash
tpsgen run \
  --input examples/demo_input.fasta \
  --target fruit \
  --candidates 3 \
  --seed 42 \
  --output outputs/demo
```

If genomic coordinates are available, prepare the required tomato promoter
FASTA directly from a reference genome and GFF3 annotation:

```bash
tpsgen extract-promoters \
  --genome tomato.fa \
  --annotation tomato.gff3 \
  --output tomato_promoters.fasta
```

This command processes `gene` records, requires complete 165-bp windows, and
reverse complements negative-strand windows so all output records use
promoter-to-gene orientation. Records with incomplete or ambiguous windows are
skipped and reported rather than padded.

The output directory contains separate, inspectable intermediate and final
files. The package-native route currently used by this command is deterministic:
the motif annotator uses exact configured motif matching, the designer protects
detected motif positions while generating seeded candidates, and the scorer
reports root, stem, leaf and fruit-associated computational scores. This
integrated route is the reproducible default; checkpoint-backed and retained
project-resource routes remain explicitly documented below.

To use the bundled tomato TransVAE checkpoint for both original and candidate
scoring, add `--scoring-backend transvae` to the same `run` command. This route
requires every input record to be exactly 165 bp and contain only `A/C/G/T`.
The selected backend and checkpoint status are recorded in `manifest.json`.

The integrated output also includes `figures/original_score_heatmap.svg`,
`figures/candidate_score_heatmap.svg` and
`figures/candidate_design_summary.svg`. These are generated from the same
workflow tables and are intended for inspection, not as independent biological
validation figures.

The integrated scoring CSV files also contain `score_type` and
`scoring_backend`. In the default `run` route, these fields identify the four
values as package-native tissue-associated heuristic scores. They support
candidate ranking; they are not calibrated expression measurements and are not
TransVAE checkpoint outputs.

`reports/workflow_summary.csv` provides one row per input promoter with its
original preferred tissue, original target score, best candidate rank, target
score, target-bias margin, mutation count and QC status. The validated input FASTA is copied to
`input/validated_promoters.fasta` so that each result directory is self-contained.

The top-level `manifest.json` is the machine-readable provenance record for the
run. Its `backends` section identifies the tomato-specific implementation used
for motif evidence, candidate generation and scoring, together with stage
inputs, outputs, checkpoint status and score definition.
The same manifest records optional model-route status: DNABERT supports both
fresh FASTA inference with the bundled checkpoint and matched precomputed-
attention post-processing. The retained
10,222-row attention resource matches the upstream `vision/dev.tsv` row count,
and its tomato specificity is not verified; it is therefore excluded from the
default tomato workflow. The bundled preGAN generator is available only through
explicit advanced commands and is not independently benchmarked or biologically
validated. TransVAE is available as an explicit scoring backend.

Run the bundled example workflow:

```bash
mkdir -p outputs

tpsgen copy-example \
  --output outputs/demo_input.fasta

tpsgen validate-input \
  --input outputs/demo_input.fasta

tpsgen annotate \
  --input outputs/demo_input.fasta \
  --output outputs/demo_annotate.csv

tpsgen predict \
  --input outputs/demo_input.fasta \
  --output outputs/demo_predict.csv

tpsgen design \
  --input outputs/demo_input.fasta \
  --target fruit \
  --candidates 3 \
  --seed 42 \
  --output outputs/demo_design.csv

tpsgen report \
  --input outputs/demo_design.csv \
  --output outputs/demo_report.json
```

The same workflow can be run with:

```bash
make demo
```

## 5. Input Requirements

Input sequences are provided as FASTA files.

Validation rules:

- FASTA records must exist.
- Sequence identifiers must be unique.
- Empty sequences are rejected.
- Sequences are normalized to uppercase.
- Supported symbols are `A`, `C`, `G`, `T`, `N` and `M`.

The integrated `run` workflow requires canonical 165-bp unambiguous `A/C/G/T`
sequences for both scoring backends. The lower-level package-native commands
can inspect exploratory sequences of other lengths.

### Preparing 165-bp Inputs For TransVAE Scoring

The 165-bp input expected by `predict-transvae` should follow the promoter-window convention used for the retained TransVAE training resource. In the original ITAG3.2 preprocessing workflow, tomato genome FASTA sequences and GFF3 annotation records were parsed, entries annotated as `gene` on the positive strand were retained, and `gene start` was used as an approximate TSS proxy because high-resolution experimental TSS annotations were not available for that resource.

Retained training-resource extraction convention:

| Case | Genomic window to extract | Sequence written to FASTA |
| --- | --- | --- |
| ITAG3.2 positive-strand `gene` record | `[gene_start - 165, gene_start)` | Reference-genome sequence as extracted |

For a positive-strand gene with `gene_start = 16480`, the retained convention extracts `[16480 - 165, 16480)`. Each TransVAE FASTA record must be exactly 165 bp and contain only `A/C/G/T`.

For user-side extensions to negative-strand genes, extract the downstream genomic window relative to reference coordinates and reverse-complement it before writing FASTA, so the sequence is represented in promoter-to-gene orientation. This is an extension rule for preparing compatible user inputs; the retained TransVAE training resource used the positive-strand ITAG3.2 convention. Do not pad short boundary-overlapping windows with `N`; exclude them from TransVAE scoring or analyze them with the package-native commands instead.

## 6. Output Formats

### Motif Annotation Output

Generated by:

```bash
tpsgen annotate
```

Main fields:

| Field | Meaning |
| --- | --- |
| `sequence_id` | Input promoter identifier |
| `motif` | Detected motif sequence |
| `start` | Zero-based motif start coordinate |
| `end` | Zero-based motif end coordinate |
| `score` | Motif match score, equal to motif length for exact matching |
| `backend` | Motif evidence implementation |
| `evidence_type` | Exact configured motif hit |

### Prediction Output

Generated by:

```bash
tpsgen predict
```

Main fields:

| Field | Meaning |
| --- | --- |
| `sequence_id` | Input promoter identifier |
| `sequence` | Normalized sequence |
| `score_root` | Root-associated heuristic score |
| `score_stem` | Stem-associated heuristic score |
| `score_leaf` | Leaf-associated heuristic score |
| `score_fruit` | Fruit-associated heuristic score |
| `preferred_tissue` | Tissue with the highest score |
| `target_tissue` | Tissue selected for candidate prioritization |
| `target_bias_margin` | Target score minus the highest score among the other three tissues |
| `tau` | Four-score concentration index; descriptive computational statistic, not an expression measurement |
| `backend` | Scoring implementation |
| `score_type` | Score definition and scale |

The explicit TransVAE model-backed route uses the same four `score_*` field
names as the package-native scorer, but values are checkpoint-derived and are
not calibrated to the package-native heuristic scale.

### Design Output

Generated by:

```bash
tpsgen design
```

Main fields:

| Field | Meaning |
| --- | --- |
| `sequence_id` | Input promoter identifier |
| `target_tissue` | Requested design tissue |
| `candidate_rank` | Rank within candidates for one input sequence |
| `original_sequence` | Input promoter sequence |
| `designed_sequence` | Generated candidate sequence |
| `score_root`, `score_stem`, `score_leaf`, `score_fruit` | Candidate tissue-associated scores |
| `target_bias_margin` | Candidate target score minus the strongest non-target score |
| `tau` | Four-score concentration index for the candidate profile |
| `preserved_motifs` | Motifs protected during package-native design |
| `num_mutations` | Number of point differences from the original sequence |
| `passes_qc` | Quality-control flag when available |
| `backend` | Candidate-generation implementation |
| `score_type` | Candidate score definition |

## 7. Package-Native Methods

### Motif Scanner

The package-native motif scanner performs deterministic exact matching with the default motif set:

```text
CAAAA, CTATT, ATTTT, TTAAA, TTTAT
```

Overlapping motif occurrences are reported. Coordinates are zero-based.

### Operational Scoring Module

The package-native scorer is deterministic. It combines GC ratio, AT ratio, poly-A content and motif counts into root, stem, leaf and fruit scores. These terms use transparent sequence features available after installation, and the coefficients are fixed software defaults for route-internal ranking rather than fitted expression-model parameters or biological effect sizes. This scorer is included so the software remains installable and testable without external checkpoints.

### Motif-Aware Designer

The package-native designer:

1. annotates motif positions in the input promoter
2. protects detected motif positions
3. applies seeded substitutions to non-protected positions
4. scores candidate sequences
5. ranks candidates by the requested target-tissue score

The default mutation rate is 0.18 per non-protected position. Target-tissue
nucleotide preferences are fixed software defaults used to generate candidate
diversity for route-internal ranking; they are not optimized biological design
parameters.

The default seed is:

```text
42
```

## 8. Project Model Adapter Commands

These commands expose checkpoint-backed or model-derived routes developed within the project workflow:

| Command | Purpose |
| --- | --- |
| `annotate-dnabert` | Run DNABERT-derived attention-to-motif post-processing |
| `predict-transvae` | Run the bundled TransVAE four-tissue checkpoint adapter |
| `model-figures` | Reconstruct retained project model figure bundles |

Example:

```bash
tpsgen annotate-dnabert \
  --dev-tsv data/raw/dnabert/dev.tsv \
  --atten-npy data/raw/dnabert/atten.npy \
  --output-dir outputs/dnabert_motifs
```

The two DNABERT inputs in this example are precomputed resources. Regenerating
`atten.npy` from raw FASTA requires a separately supplied fine-tuned DNABERT
checkpoint and is outside the `annotate-dnabert` post-processing command.

Validate a retained resource pair before post-processing it:

```bash
tpsgen validate-dnabert \
  --dev-tsv data/raw/dnabert/dev.tsv \
  --atten-npy data/raw/dnabert/atten.npy
```

This checks row counts, reconstructed sequence length and class counts; it does
not establish tomato species provenance.

Package-native `predict` and `design` remain deterministic and do not use these checkpoints unless an explicit model adapter command is called.

The current release exposes the retained TransVAE checkpoint for four-tissue
scoring through `predict-transvae`. Latent-space TransVAE candidate generation
is not exposed as a public command in this release. Use package-native `design`
for released candidate generation.

## 9. Reproducing Application Note Results

The Application Note uses two result layers.

Package-native demo:

```bash
make demo
```

Retained manuscript-facing result pack:

```bash
make reproduce-results
```

The retained result pack is written to:

```text
data/results/reproducible_legacy/
```

Important outputs:

| Resource | Use |
| --- | --- |
| `manifest.csv` | Maps retained source tables to regenerated outputs |
| `tables/expression_heatmap_source.csv` | Source table for Supplementary Figure S2 |
| `tables/dnabert_motif_top20.csv` | Source table for Supplementary Figure S3 |
| `tables/kmer_frequency_comparison.csv` | Source table for Supplementary Figure S4 |
| `tables/design_candidate_summary.csv` | Source table for Supplementary Figure S5 |
| `tables/prediction_reference_stats.csv` | Pearson statistic for the quantitative reference |

The polished manuscript and supplementary figures are stored under `docs/fig/`.

## 10. Why Demo Outputs Are Small

The bundled demo uses two promoter sequences so that installation and CLI behavior can be checked quickly. Manuscript figures use retained result tables with broader coverage:

| Result | Scale |
| --- | --- |
| Quantitative reference | 12,575 valid predicted/true scalar-expression pairs |
| Expression heatmap | 100 promoters balanced across four preferred tissues |
| DNABERT-derived motif summary | Top 20 retained motifs |
| 4-mer comparison | 256 4-mers per sequence region |
| Design-candidate summary | 199 retained fruit-targeted candidates |

This design keeps the package lightweight while preserving the project result context needed for manuscript review.

## 11. Supplementary Figure R Scripts

Supplementary figures are retained under `docs/fig/`, with source tables under
`data/results/reproducible_legacy/`. Run `make reproduce-results` to regenerate
the manuscript-facing result pack. R figure-rendering scripts are not part of
the released tool.

## 12. Data And Model Availability

Bundled data layers:

| Directory | Contents |
| --- | --- |
| `data/raw/` | Selected retained raw tables and support files |
| `data/processed/` | Processed intermediate tables |
| `data/results/demo/` | Package-native demo outputs |
| `data/results/reproducible_legacy/` | Retained manuscript-facing project result pack |
| `data/external/` | Manifests for large files not bundled by default |
| `models/` | Bundled TransVAE, DNABERT and preGAN resources plus checksum manifest |

The Transformer-VAE checkpoint and preGAN expression-constraint resource
are bundled in `models/` and packaged into the release wheel. The TransVAE route
includes a compatible training entry point; preGAN includes tested training
components and an explicit checkpoint-backed generation route; DNABERT supports
fresh checkpoint inference as well as precomputed attention post-processing.
Large genomes, HDF5 corpora, BLAST
databases and optional external resources are not bundled into normal git
history.

The bundled preGAN generator checkpoint is runnable, but it is not presented as
an independently benchmarked or biologically validated model. The package-level
preGAN code implements
`M`-mask semantics, fixed-base preservation, conditional generator and
discriminator modules, WGAN-GP loss terms and frozen DenseLSTM expression
constraint plumbing for future retraining.

Provenance files:

```text
data/source_registry.tsv
data/inventory.tsv
data/summary.json
data/external/external_resources.tsv
RESOURCE_LICENSES.md
```

## 13. Training A Compatible TransVAE Checkpoint

The TransVAE model-backed route is supported by both a bundled checkpoint and a repository training script.

Fast smoke test:

```bash
make train-transvae-smoke
```

Default training command:

```bash
PYTHONPATH=src python scripts/train_transvae.py \
  --config configs/training_transvae.yaml
```

Default outputs:

```text
models/transvae/trained_transvae_model.pth
models/transvae/trained_transvae_metrics.json
```

The training output uses the same strict `TransVAEMLP` schema as the released
Transformer-VAE route. Use the bundled paper-aligned checkpoint for the
retained model results, or pass a newly trained compatible checkpoint with:

```bash
tpsgen predict-transvae \
  --input examples/demo_input.fasta \
  --checkpoint models/transvae/best_val_corr_model.pth \
  --output outputs/transvae_predict.csv
```

See `docs/training.md` for the input table schema, objective function and checkpoint compatibility details.

## 14. preGAN Training Components

The repository includes package-level preGAN training components and a trained
generator checkpoint. A small smoke-test table at `data/raw/pregan_expression/pregan_smoke.csv` is also retained. The table
uses the retained masked-promoter format:

| Column | Meaning |
| --- | --- |
| `realA` | 165-bp template with fixed bases and `M` at mutable positions |
| `realB` | Completed 165-bp promoter sequence used as the supervised scoring input |
| `expr` | Scalar expression target used by the expression-constraint route |

Run the smoke test with:

```bash
make train-pregan-smoke
```

This command loads the bundled expression-constraint scorer, runs a small
WGAN-GP training-plumbing check, and writes temporary artifacts under `tmp/`.
These artifacts are explicitly not validated generation models. Full preGAN
retraining requires the separate full masked-promoter table registered in
`data/external/external_resources.tsv`.

Run the trained preGAN generator and score candidates with the bundled
Transformer-VAE model:

```bash
tpsgen run-pregan \
  --input templates.fasta \
  --checkpoint models/pregan/generator_checkpoint.pt \
  --target fruit --candidates 5 --seed 42 --output outputs/pregan_workflow
```

The same workflow is available through the main `run` command by setting
`--design-backend pregan`; in that mode the input must be a masked template
FASTA and `--checkpoint` must point to a trained generator.

## 15. Testing

Run:

```bash
make test
```

The test suite covers:

- FASTA validation
- CSV output
- CLI behavior
- motif annotation
- package-native scoring and design
- report generation
- project model adapter behavior
- figure export utilities

## 16. Citation

Citation metadata is provided in `CITATION.cff`. For publication, cite the Application Note and the archived repository release.
