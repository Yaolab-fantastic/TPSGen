# Training Guide

This document describes the repository training entry point for the TransVAE model-backed scoring route.

The training code is intentionally lightweight and inspectable. It provides the model architecture, dataset reader, supervised objective and checkpoint writer used to train a checkpoint compatible with:

```bash
tpsgen predict-transvae
```

## Files

| File | Purpose |
| --- | --- |
| `src/tomato_promoter_designer/legacy/transvae_tomato.py` | TransVAE model architecture and scoring adapter |
| `src/tomato_promoter_designer/training/transvae.py` | Training dataset, config loader and training loop |
| `scripts/train_transvae.py` | Command-line training wrapper |
| `configs/training_transvae.yaml` | Default training configuration |
| `data/raw/transvae/training_set.csv` | Repository training table used by the default config |

## Training Data Format

The default training table uses one promoter sequence column and four tissue-associated target columns:

| Column | Meaning |
| --- | --- |
| `realB` | 165-bp promoter sequence containing only `A/C/G/T` |
| `expr_tissue_1` | Root-associated training target |
| `expr_tissue_2` | Stem-associated training target |
| `expr_tissue_3` | Leaf-associated training target |
| `expr_tissue_4` | Fruit-associated training target |

Rows with non-165-bp sequences, ambiguous bases or missing numeric targets are skipped by the training dataset loader.

## Full Training Command

```bash
PYTHONPATH=src python scripts/train_transvae.py \
  --config configs/training_transvae.yaml
```

By default this writes:

```text
models/transvae/trained_transvae_model.pth
models/transvae/trained_transvae_metrics.json
```

The training script uses the same `TransVAEMLP` architecture and strict state
dictionary schema as the released scoring route. It trains the four-tissue
supervised scoring head through the Transformer-VAE sequence representation;
it is a reproducibility entry point, not a replacement for the retained
paper checkpoint or a hyperparameter benchmark. The bundled paper-aligned
checkpoint is intended for routine `predict-transvae` use.

Use `models/transvae/best_val_corr_model.pth` for the released paper-aligned
Transformer-VAE scoring route.

## Smoke Test

For a fast check that the training and loading path works:

```bash
make train-transvae-smoke
```

This trains on eight rows for one epoch and writes a temporary legacy training
checkpoint under `tmp/`; it does not claim compatibility with the released
Transformer-VAE checkpoint.

## Training Objective

The training loop optimizes the four-tissue supervised score loss:

```text
total loss = prediction_weight * four-tissue score regression loss
```

The predictor learns four continuous tissue-associated scores from the Transformer-VAE latent representation. Sequence decoding is not part of this released training entry point.

## Notes

The bundled checkpoint in `models/transvae/best_val_corr_model.pth` is provided so users can run the model-backed scoring route immediately. The training script documents how a compatible scoring checkpoint can be regenerated or replaced with a newly trained checkpoint using the same architecture and output format.

## preGAN Smoke Training

preGAN training components are implemented separately from the TransVAE
training route. They support the retained masked-promoter format in which
`realA` contains fixed bases plus `M` symbols for mutable positions, `realB`
contains the completed promoter sequence and `expr` contains the scalar target.

Run the training-plumbing smoke test with:

```bash
make train-pregan-smoke
```

This uses `data/raw/pregan_expression/pregan_smoke.csv` and the bundled
expression-constraint scorer to verify fixed-base preservation, WGAN-GP loss
wiring and frozen expression-constraint behavior. The generated temporary
checkpoint is not a validated preGAN generator and should not be used for
promoter design inference.
