#!/usr/bin/env python
"""Train the conditional preGAN generator on paired tomato promoter data."""
from __future__ import annotations

import argparse
import json

from tpsgen.legacy.pregan_expression import PreganExpressionConstraintScorer
from tpsgen.training.pregan import PreGANSmokeConfig, run_pregan_smoke_training


def main() -> None:
    defaults = PreGANSmokeConfig()
    parser = argparse.ArgumentParser(description="Train the conditional tomato preGAN generator.")
    parser.add_argument("--input-csv", required=True, help="Paired CSV with realA, realB and expr columns.")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--expression-checkpoint")
    parser.add_argument("--expression-module-dir")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = PreGANSmokeConfig(
        input_csv=args.input_csv,
        output_checkpoint=args.output_checkpoint,
        metrics_json=args.metrics_json,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    predictor = PreganExpressionConstraintScorer(
        checkpoint_path=args.expression_checkpoint,
        module_dir=args.expression_module_dir,
        device=args.device,
    ).model
    metrics = run_pregan_smoke_training(config, predictor)
    metrics["release_boundary"] = "trained conditional generator; validate externally before biological use"
    with open(args.metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({key: metrics[key] for key in ("num_records", "steps", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
