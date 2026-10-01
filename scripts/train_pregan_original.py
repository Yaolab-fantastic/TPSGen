#!/usr/bin/env python
"""Train the thesis preGAN architecture and save state-dict checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tpsgen.legacy.pregan_original import WGAN, get_infinite_batches
from tpsgen.legacy.pregan_original_data import LoadData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--critic-updates", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    predictor_module_dir = str(Path(args.predictor_checkpoint).parent.resolve())
    if predictor_module_dir not in sys.path:
        sys.path.insert(0, predictor_module_dir)
    import SeqRegressionModel  # noqa: F401, required by the legacy pickle
    gpu_ids = "0" if args.device.startswith("cuda") else ""
    dataset = LoadData(path=args.input_csv, split_r=0.8, is_train=True, gpu_ids=gpu_ids)
    batches = get_infinite_batches(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    model = WGAN(
        input_nc=4, output_nc=4, seqL=165, lr=1e-4, gpu_ids=gpu_ids,
        l1_w=50, predictor_path=args.predictor_checkpoint,
    )
    history = []
    for iteration in range(1, args.iterations + 1):
        critic_loss = 0.0
        for _ in range(args.critic_updates):
            batch = next(batches)
            model.backward_d(batch["out"], batch["in"])
            critic_loss += float(model.d_total_loss.detach())
        model.backward_g(batch["in"], batch["expr"])
        if iteration == 1 or iteration % 100 == 0:
            entry = {
                "iteration": iteration,
                "generator_loss": float(model.g_total_loss.detach()),
                "adversarial_loss": float(model.g_loss.detach()),
                "l1_loss": float(model.g_l1.detach()),
                "predictor_loss": float(model.p_loss.detach()),
                "critic_loss": critic_loss / args.critic_updates,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "thesis_original_pregan",
        "generator_state_dict": model.generator.state_dict(),
        "discriminator_state_dict": model.discriminator.state_dict(),
        "sequence_length": 165,
        "ngf": 512,
        "l1_weight": 50.0,
        "iterations": args.iterations,
        "critic_updates": args.critic_updates,
        "seed": args.seed,
    }, output)
    metrics = {
        "architecture": "thesis_original_pregan",
        "num_training_records": len(dataset),
        "source_records": 5100,
        "iterations": args.iterations,
        "critic_updates_per_generator": args.critic_updates,
        "checkpoint": str(output),
        "history": history,
    }
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
