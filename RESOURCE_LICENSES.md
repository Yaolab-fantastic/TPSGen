# Resource Licensing And Attribution

The repository source code, package documentation, package-native outputs and
project-authored model code are released under the repository MIT license.

The bundled TransVAE checkpoint, preGAN expression-constraint resource, tomato
promoter project tables, processed result tables and manuscript-reproduction
resources were generated within the TPSGen project. They are
distributed with this repository for reproducible research use under the
repository release terms.
File-level origins and roles are recorded in `data/source_registry.tsv`, and
checkpoint checksums are recorded in `models/weights_manifest.json`.

DNABERT-derived processing uses methods and resources based on DNABERT. Users
must retain the relevant DNABERT citation and attribution when redistributing
DNABERT-derived motif resources. TPSGen bundles a fine-tuned checkpoint for the
explicit `predict-dnabert` FASTA-inference route. The separate
`annotate-dnabert` command consumes matched precomputed sequence and attention
inputs for historical motif post-processing.

Large optional genomes, corpora and BLAST databases are not distributed in the
default repository or Python wheel. Their availability and intended use are
listed in `data/external/external_resources.tsv`; their original licenses and
database terms continue to apply.
