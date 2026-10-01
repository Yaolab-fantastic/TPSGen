# TPSGen Bundled Models

This directory contains the model resources used by the final TPSGen release. Checksums and machine-readable provenance are stored in `weights_manifest.json`.

## TransVAE

Default scoring checkpoint: `models/transvae/historical_compatible_best_val_corr.pth`

- architecture: Transformer variational autoencoder with four-output MLP head
- mode: historical checkpoint compatibility for the reported model-development scoring behavior
- validation PCC recovered with the historical input contract: 0.863 / 0.912 / 0.884 / 0.876
- SHA-256: `cdc55cd2ce796386a5be74b8697790cbad70e7bcb6cfff85e189b234c9628f78`

Development checkpoint: `models/transvae/full_length_joint_model.pth`

- input: corrected full 165 bp A/C/G/T sequence
- training mode: `paper_joint`
- objective: reconstruction + weighted KL + weighted prediction + weighted differentiable 3-mer loss
- training records: 11,257; epochs: 20
- parameters: 2,498,834
- SHA-256: `835f9e5cb7c1b184a19c4db9cb8d5652adb5f6e1eaf73a839929e00327f4aa03`

Training metadata: `models/transvae/full_length_joint_metrics.json`

Fixed split metadata: `data/raw/transvae/paper_split/manifest.json`

## Connected Models

DNABERT supplies retained-position evidence to the preGAN workflow. preGAN generates candidates from the resulting masked template, and the bundled TransVAE checkpoint performs final four-tissue scoring.

## Validation

```bash
tpsgen validate-models
```

Required checkpoints must be reported as available before a final integrated run.
