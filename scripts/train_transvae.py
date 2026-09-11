#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tomato_promoter_designer.training.transvae import load_training_config, train_transvae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TomatoPromoterDesigner TransVAE model.")
    parser.add_argument("--config", default="configs/training_transvae.yaml", help="Training config path.")
    parser.add_argument("--input-csv", help="Override training CSV path.")
    parser.add_argument("--output-checkpoint", help="Override output checkpoint path.")
    parser.add_argument("--metrics-json", help="Override metrics JSON path.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--max-rows", type=int, help="Use a small subset for smoke tests.")
    parser.add_argument("--device", help="Training device, e.g. cpu or cuda.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    for field in ("input_csv", "output_checkpoint", "metrics_json", "epochs", "max_rows", "device"):
        value = getattr(args, field.replace("-", "_"), None)
        if value is not None:
            setattr(config, field, value)

    metrics = train_transvae(config)
    print(json.dumps({key: metrics[key] for key in ("num_records", "best_validation_loss", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
