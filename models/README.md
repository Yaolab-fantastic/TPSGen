# Models

This directory contains model resources used by the released DNABERT and
preGAN routes. A corrected full-length TransVAE checkpoint is not bundled yet.

## Bundled Resources

| Path | Used by | Notes |
| --- | --- | --- |
| `models/transvae/full_length_joint_model.pth` | `predict-transvae` | Expected output of the corrected full-length joint training route; not bundled until retraining is completed. |
| `models/dnabert/` | `predict-dnabert` | Tomato fine-tuned 6-mer DNABERT checkpoint and tokenizer resources for fresh FASTA inference. |
| `models/pregan/generator_checkpoint.pt` | `pregan-generate` and `run --design-backend pregan` | Local conditional-generator checkpoint trained on the retained tomato paired table; historical quantitative reproducibility has not been independently established and candidates require external validation. |
| `models/pregan_expression/165_mpra_expr_denselstm.pth` | preGAN training smoke test | DenseLSTM scalar-expression checkpoint used as the frozen expression-constraint scorer in the preGAN training-plumbing test. |
| `models/pregan_expression/SeqRegressionModel.py` | preGAN training smoke test | Required class-definition file for loading the bundled expression-constraint checkpoint. |

Checksums and status are recorded in `models/weights_manifest.json`.

## Training Compatibility

The repository includes a TransVAE-MLP training entry point for reproducibility:

```bash
PYTHONPATH=src python scripts/train_transvae.py \
  --config configs/training_transvae.yaml
```

This command generates a strict `TransVAEMLP` state dictionary compatible
with `predict-transvae`. The released checkpoint is loaded without key
remapping or architecture substitution.

## Public Boundary

Package-native commands do not require these files:

```bash
tpsgen predict
tpsgen design
```

The corrected TransVAE checkpoint, once trained, will be used only by the explicit model command:

```bash
tpsgen predict-transvae
```

The preGAN expression-constraint checkpoint is not exposed as a standalone
public prediction mechanism. It supports the preGAN training path and smoke test.
The bundled generator checkpoint can be run through the explicit
`pregan-generate` or integrated `run --design-backend pregan` route, but it is
not presented as an independently benchmarked or experimentally validated
generator.

If the model directory is moved outside the repository, set:

```bash
export TPSGEN_MODELS_DIR=/path/to/models
```
