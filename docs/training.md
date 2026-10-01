# TransVAE Joint Training

The release training path matches the TPSGen model description: a Transformer variational autoencoder processes the complete 165 bp sequence and jointly optimizes reconstruction, KL-divergence, four-tissue prediction, and differentiable 3-mer objectives.

## Environment

```bash
cd /data/zhoujie/Paper/github/TPSGen
conda activate py39
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA availability must print `True` for GPU training.

## Fixed Release Split

`data/raw/transvae/paper_split/manifest.json` defines the historical seed-42
split used by the released training workflow. A differently constructed
split requires retraining and must not be substituted beneath an existing
checkpoint.

| File | Records | Use |
|---|---:|---|
| `train.csv` | 11,257 | model fitting |
| `validation.csv` | 1,407 | validation |
| `test.csv` | 1,407 | final evaluation |

The manifest records the source dataset checksum and each split-file checksum. Release evaluation must use these fixed files without reshuffling.

## Training Command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
python scripts/train_transvae.py \
  --input-csv data/raw/transvae/paper_split/train.csv \
  --output-checkpoint models/transvae/full_length_joint_model.pth \
  --metrics-json models/transvae/full_length_joint_metrics.json \
  --epochs 20 \
  --device cuda
```

The released checkpoint was trained for 20 epochs on the fixed 11,257-record training file. Its SHA-256 is `835f9e5cb7c1b184a19c4db9cb8d5652adb5f6e1eaf73a839929e00327f4aa03`.

The best epoch was selected through an internal deterministic subset of the training file. The separately versioned validation and test files are not used for parameter updates.

## Release Checks

```bash
tpsgen validate-models
PYTHONPATH=src pytest -q
```

Keep the architecture version, tokenizer rules, sequence length, loss weights, seed, split hashes, epoch count, checkpoint checksum, and record-level evaluation outputs with every release.
