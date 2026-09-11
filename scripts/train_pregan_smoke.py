#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tpsgen.legacy.pregan_expression import PreganExpressionConstraintScorer
from tpsgen.training.pregan import PreGANSmokeConfig, run_pregan_smoke_training


def main() -> None:
    defaults = PreGANSmokeConfig()
    parser = argparse.ArgumentParser(
        description="Run a preGAN training-plumbing smoke test."
    )
    parser.add_argument("--input-csv", default=defaults.input_csv)
    parser.add_argument("--output-checkpoint", default=defaults.output_checkpoint)
    parser.add_argument("--metrics-json", default=defaults.metrics_json)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--expression-checkpoint", help="Optional expression-constraint checkpoint path.")
    parser.add_argument("--expression-module-dir", help="Optional directory containing SeqRegressionModel.py.")
    args = parser.parse_args()

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
    metrics = run_pregan_smoke_training(config, predictor=predictor)
    print(json.dumps({key: metrics[key] for key in ("num_records", "steps", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
