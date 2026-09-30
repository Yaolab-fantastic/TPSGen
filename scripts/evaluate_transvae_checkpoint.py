#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tpsgen.evaluation.checkpoint_validation import evaluate_checkpoint_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit record-level TransVAE checkpoint predictions and labels."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    result = evaluate_checkpoint_predictions(
        args.input_csv, args.output_json, expected_records=args.expected_records
    )
    print(json.dumps({"num_records": result["num_records"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
