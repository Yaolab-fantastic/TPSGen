from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from tpsgen.models.transvae_mlp import (
    TransVAEMLP,
    encode_dna,
)


@dataclass
class TrainingConfig:
    input_csv: str = "data/raw/transvae/training_set.csv"
    sequence_column: str = "realB"
    tissue_columns: tuple[str, str, str, str] = (
        "expr_tissue_1",
        "expr_tissue_2",
        "expr_tissue_3",
        "expr_tissue_4",
    )
    output_checkpoint: str = "models/transvae/trained_transvae_model.pth"
    metrics_json: str = "models/transvae/trained_transvae_metrics.json"
    sequence_length: int = 165
    latent_dim: int = 64
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    validation_fraction: float = 0.1
    seed: int = 42
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    kl_weight: float = 0.001
    max_rows: int | None = None
    device: str = "cpu"


class TransVAEDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        input_csv: str | Path,
        sequence_column: str,
        tissue_columns: Iterable[str],
        sequence_length: int,
        max_rows: int | None = None,
    ) -> None:
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []
        tissue_columns = tuple(tissue_columns)
        with Path(input_csv).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sequence = row[sequence_column].upper().strip()
                if len(sequence) != sequence_length:
                    continue
                if any(base not in {"A", "C", "G", "T"} for base in sequence):
                    continue
                try:
                    targets = [float(row[column]) for column in tissue_columns]
                except (KeyError, TypeError, ValueError):
                    continue
                self.records.append((encode_dna(sequence), torch.tensor(targets, dtype=torch.float32)))
                if max_rows is not None and len(self.records) >= max_rows:
                    break
        if not self.records:
            raise ValueError(f"No valid {sequence_length}-bp A/C/G/T training records found in {input_csv}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.records[index]


def load_training_config(path: str | Path | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _read_simple_yaml(path)
    if "tissue_columns" in data and isinstance(data["tissue_columns"], list):
        data["tissue_columns"] = tuple(data["tissue_columns"])
    return TrainingConfig(**data)


def _read_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def train_transvae(config: TrainingConfig) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    dataset = TransVAEDataset(
        input_csv=config.input_csv,
        sequence_column=config.sequence_column,
        tissue_columns=config.tissue_columns,
        sequence_length=config.sequence_length,
        max_rows=config.max_rows,
    )
    val_size = max(1, int(len(dataset) * config.validation_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    if val_size:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = TransVAEMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    output_checkpoint = Path(config.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, config, device)
        val_metrics = _run_epoch(model, val_loader, None, config, device) if val_loader is not None else train_metrics
        row = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), output_checkpoint)

    metrics = {
        "config": asdict(config),
        "num_records": len(dataset),
        "num_train_records": train_size,
        "num_validation_records": val_size,
        "best_validation_loss": best_val,
        "history": history,
        "checkpoint": str(output_checkpoint),
    }
    metrics_path = Path(config.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


MpraVAEDataset = TransVAEDataset
train_mpravae = train_transvae


def _run_epoch(
    model: TransVAEMLP,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    if loader is None:
        return {"loss": 0.0, "reconstruction_loss": 0.0, "prediction_loss": 0.0, "kl_loss": 0.0}
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "prediction_loss": 0.0, "kl_loss": 0.0}
    total_items = 0
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        predicted = model.score_tokens(sequences)
        prediction_loss = F.mse_loss(predicted, targets)
        loss = config.prediction_weight * prediction_loss
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch_size = sequences.shape[0]
        total_items += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["reconstruction_loss"] += 0.0
        totals["prediction_loss"] += float(prediction_loss.detach().cpu()) * batch_size
        totals["kl_loss"] += 0.0
    return {key: value / max(total_items, 1) for key, value in totals.items()}
