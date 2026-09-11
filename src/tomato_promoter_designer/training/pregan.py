from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASES = ("A", "T", "C", "G")
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
MASK_SYMBOL = "M"


@dataclass(frozen=True, slots=True)
class MaskedPromoterExample:
    template: str
    target_sequence: str
    expression: float


@dataclass(slots=True)
class PreGANSmokeConfig:
    input_csv: str = "data/raw/pregan_expression/pregan_smoke.csv"
    output_checkpoint: str = "tmp/pregan_smoke_checkpoint.pt"
    metrics_json: str = "tmp/pregan_smoke_metrics.json"
    sequence_length: int = 165
    batch_size: int = 2
    steps: int = 2
    noise_channels: int = 16
    hidden_channels: int = 64
    learning_rate: float = 0.0001
    reconstruction_weight: float = 1.0
    expression_weight: float = 0.0001
    homopolymer_weight: float = 1.0
    gradient_penalty_weight: float = 10.0
    seed: int = 42
    device: str = "cpu"


def encode_dna(sequence: str) -> torch.Tensor:
    encoded = torch.zeros((4, len(sequence)), dtype=torch.float32)
    for index, base in enumerate(sequence.upper()):
        if base not in BASE_TO_INDEX:
            raise ValueError(f"Unsupported DNA base {base!r} at position {index}.")
        encoded[BASE_TO_INDEX[base], index] = 1.0
    return encoded


def encode_masked_template(template: str) -> tuple[torch.Tensor, torch.Tensor]:
    fixed = torch.zeros((4, len(template)), dtype=torch.float32)
    mutable_mask = torch.zeros((len(template),), dtype=torch.bool)
    for index, symbol in enumerate(template.upper()):
        if symbol == MASK_SYMBOL:
            mutable_mask[index] = True
            continue
        if symbol not in BASE_TO_INDEX:
            raise ValueError(f"Unsupported template symbol {symbol!r} at position {index}.")
        fixed[BASE_TO_INDEX[symbol], index] = 1.0
    return fixed, mutable_mask


def apply_masked_template(
    logits: torch.Tensor,
    fixed_bases: torch.Tensor,
    mutable_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if logits.shape != fixed_bases.shape:
        raise ValueError("logits and fixed_bases must both have shape [batch, 4, length].")
    if mutable_mask.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("mutable_mask must have shape [batch, length].")
    mutable = mutable_mask.unsqueeze(1).to(dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(logits / temperature, dim=1)
    return probabilities * mutable + fixed_bases.to(logits.device) * (1.0 - mutable)


def decode_soft_sequence(sequence_tensor: torch.Tensor) -> str:
    if sequence_tensor.ndim != 2 or sequence_tensor.shape[0] != 4:
        raise ValueError("sequence_tensor must have shape [4, length].")
    indices = sequence_tensor.argmax(dim=0).detach().cpu().tolist()
    return "".join(BASES[index] for index in indices)


class MaskedPromoterDataset(Dataset):
    def __init__(self, csv_path: str | Path, sequence_length: int = 165) -> None:
        self.csv_path = Path(csv_path)
        self.sequence_length = sequence_length
        self.examples = self._load_examples()

    def _load_examples(self) -> list[MaskedPromoterExample]:
        examples: list[MaskedPromoterExample] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"realA", "realB", "expr"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
            for row_index, row in enumerate(reader, start=2):
                template = row["realA"].strip().upper()
                target = row["realB"].strip().upper()
                if len(template) != self.sequence_length or len(target) != self.sequence_length:
                    raise ValueError(f"Row {row_index} does not contain {self.sequence_length}-bp sequences.")
                fixed, mutable = encode_masked_template(template)
                target_encoded = encode_dna(target)
                fixed_positions = ~mutable
                if not torch.equal(fixed[:, fixed_positions], target_encoded[:, fixed_positions]):
                    raise ValueError(f"Row {row_index} has fixed template bases that do not match realB.")
                examples.append(
                    MaskedPromoterExample(
                        template=template,
                        target_sequence=target,
                        expression=float(row["expr"]),
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        fixed_bases, mutable_mask = encode_masked_template(example.template)
        return {
            "fixed_bases": fixed_bases,
            "mutable_mask": mutable_mask,
            "target_sequence": encode_dna(example.target_sequence),
            "expression": torch.tensor([example.expression], dtype=torch.float32),
        }


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_channels: int = 16, hidden_channels: int = 64) -> None:
        super().__init__()
        self.noise_channels = noise_channels
        self.net = nn.Sequential(
            nn.Conv1d(noise_channels + 5, hidden_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, 4, kernel_size=1),
        )

    def forward(
        self,
        noise: torch.Tensor,
        fixed_bases: torch.Tensor,
        mutable_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_channel = mutable_mask.unsqueeze(1).to(dtype=fixed_bases.dtype, device=fixed_bases.device)
        return self.net(torch.cat([noise, fixed_bases, mask_channel], dim=1))


def load_pregan_generator(checkpoint: str | Path, device: str = "cpu") -> ConditionalGenerator:
    """Load a generator checkpoint produced by the preGAN training entry point."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"preGAN generator checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    config_data = payload.get("config", {})
    generator = ConditionalGenerator(
        noise_channels=int(config_data.get("noise_channels", 16)),
        hidden_channels=int(config_data.get("hidden_channels", 64)),
    ).to(device)
    state = payload.get("generator_state_dict", payload)
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return generator


def generate_pregan_candidates(
    templates: list[str], checkpoint: str | Path, candidates: int = 5,
    seed: int = 42, device: str = "cpu",
) -> list[dict[str, object]]:
    if candidates < 1:
        raise ValueError("candidates must be positive")
    generator = load_pregan_generator(checkpoint, device=device)
    torch.manual_seed(seed)
    output: list[dict[str, object]] = []
    for template_index, template in enumerate(templates):
        normalized = template.strip().upper()
        if len(normalized) != 165:
            raise ValueError("preGAN generation requires 165-bp templates")
        fixed, mutable = encode_masked_template(normalized)
        fixed = fixed.unsqueeze(0).to(device)
        mutable = mutable.unsqueeze(0).to(device)
        for rank in range(1, candidates + 1):
            noise = sample_noise(1, generator.noise_channels, 165, device)
            logits = generator(noise, fixed, mutable)
            generated = apply_masked_template(logits, fixed, mutable)
            sequence = decode_soft_sequence(generated[0])
            fixed_ok = all(
                normalized[index] == sequence[index]
                for index in range(165) if not bool(mutable[0, index])
            )
            output.append({
                "template_index": template_index,
                "candidate_rank": rank,
                "template": normalized,
                "sequence": sequence,
                "masked_positions": int(mutable.sum().item()),
                "preserves_fixed_positions": fixed_ok,
                "gc_fraction": round((sequence.count("G") + sequence.count("C")) / 165, 6),
                "max_homopolymer": max(
                    (len(run) for run in re.findall(r"A+|C+|G+|T+", sequence)),
                    default=0,
                ),
                "passes_qc": (
                    0.12 <= (sequence.count("G") + sequence.count("C")) / 165 <= 0.52
                    and max(
                        (len(run) for run in re.findall(r"A+|C+|G+|T+", sequence)),
                        default=0,
                    ) <= 13
                ),
                "backend": "pregan_conditional_generator",
            })
    return output


class ConditionalDiscriminator(nn.Module):
    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(9, hidden_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        fixed_bases: torch.Tensor,
        mutable_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_channel = mutable_mask.unsqueeze(1).to(dtype=sequence.dtype, device=sequence.device)
        features = self.features(torch.cat([sequence, fixed_bases, mask_channel], dim=1)).squeeze(-1)
        return self.head(features)


def sample_noise(batch_size: int, noise_channels: int, length: int, device: torch.device | str) -> torch.Tensor:
    return torch.randn(batch_size, noise_channels, length, device=device)


def freeze_predictor(predictor: nn.Module) -> nn.Module:
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


def gradient_penalty(
    discriminator: ConditionalDiscriminator,
    real_sequences: torch.Tensor,
    fake_sequences: torch.Tensor,
    fixed_bases: torch.Tensor,
    mutable_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = real_sequences.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, device=real_sequences.device)
    interpolated = epsilon * real_sequences + (1.0 - epsilon) * fake_sequences.detach()
    interpolated.requires_grad_(True)
    scores = discriminator(interpolated, fixed_bases, mutable_mask)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def discriminator_wgan_gp_loss(
    discriminator: ConditionalDiscriminator,
    real_sequences: torch.Tensor,
    fake_sequences: torch.Tensor,
    fixed_bases: torch.Tensor,
    mutable_mask: torch.Tensor,
    gradient_penalty_weight: float = 10.0,
) -> torch.Tensor:
    real_scores = discriminator(real_sequences, fixed_bases, mutable_mask)
    fake_scores = discriminator(fake_sequences.detach(), fixed_bases, mutable_mask)
    penalty = gradient_penalty(discriminator, real_sequences, fake_sequences, fixed_bases, mutable_mask)
    return fake_scores.mean() - real_scores.mean() + gradient_penalty_weight * penalty


def generator_loss(
    discriminator: ConditionalDiscriminator,
    predictor: nn.Module,
    fake_sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    fixed_bases: torch.Tensor,
    mutable_mask: torch.Tensor,
    target_expression: torch.Tensor,
    reconstruction_weight: float = 1.0,
    expression_weight: float = 0.01,
    homopolymer_weight: float = 1.0,
) -> torch.Tensor:
    adversarial = -discriminator(fake_sequences, fixed_bases, mutable_mask).mean()
    mutable = mutable_mask.unsqueeze(1).to(dtype=fake_sequences.dtype, device=fake_sequences.device)
    reconstruction = F.mse_loss(fake_sequences * mutable, target_sequences * mutable)
    predicted_expression = predictor(fake_sequences)
    expression = F.mse_loss(predicted_expression.reshape_as(target_expression), target_expression)
    five_run = torch.stack(
        [(fake_sequences[:, base, :-4] * fake_sequences[:, base, 1:-3] *
          fake_sequences[:, base, 2:-2] * fake_sequences[:, base, 3:-1] *
          fake_sequences[:, base, 4:]).mean() for base in range(4)]
    ).sum()
    return adversarial + reconstruction_weight * reconstruction + expression_weight * expression + homopolymer_weight * five_run


def run_pregan_smoke_training(config: PreGANSmokeConfig, predictor: nn.Module) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dataset = MaskedPromoterDataset(config.input_csv, sequence_length=config.sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    generator = ConditionalGenerator(
        noise_channels=config.noise_channels,
        hidden_channels=config.hidden_channels,
    ).to(device)
    discriminator = ConditionalDiscriminator(hidden_channels=config.hidden_channels).to(device)
    predictor = freeze_predictor(predictor).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))

    history: list[dict[str, float]] = []
    iterator = iter(loader)
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        fixed_bases = batch["fixed_bases"].to(device)
        mutable_mask = batch["mutable_mask"].to(device)
        target_sequence = batch["target_sequence"].to(device)
        target_expression = batch["expression"].to(device)

        noise = sample_noise(
            target_sequence.shape[0],
            generator.noise_channels,
            target_sequence.shape[-1],
            device=device,
        )
        fake_logits = generator(noise, fixed_bases, mutable_mask)
        fake_sequence = apply_masked_template(fake_logits, fixed_bases, mutable_mask)

        discriminator_optimizer.zero_grad()
        d_loss = discriminator_wgan_gp_loss(
            discriminator,
            target_sequence,
            fake_sequence,
            fixed_bases,
            mutable_mask,
            gradient_penalty_weight=config.gradient_penalty_weight,
        )
        d_loss.backward()
        discriminator_optimizer.step()

        noise = sample_noise(
            target_sequence.shape[0],
            generator.noise_channels,
            target_sequence.shape[-1],
            device=device,
        )
        fake_logits = generator(noise, fixed_bases, mutable_mask)
        fake_sequence = apply_masked_template(fake_logits, fixed_bases, mutable_mask)

        generator_optimizer.zero_grad()
        g_loss = generator_loss(
            discriminator,
            predictor,
            fake_sequence,
            target_sequence,
            fixed_bases,
            mutable_mask,
            target_expression,
            reconstruction_weight=config.reconstruction_weight,
            expression_weight=config.expression_weight,
            homopolymer_weight=config.homopolymer_weight,
        )
        g_loss.backward()
        generator_optimizer.step()
        history.append({"step": float(step), "generator_loss": float(g_loss.detach()), "discriminator_loss": float(d_loss.detach())})

    checkpoint_path = Path(config.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "config": asdict(config),
            "note": "Smoke-test checkpoint for training plumbing only; not a validated generation model.",
        },
        checkpoint_path,
    )
    metrics = {
        "num_records": len(dataset),
        "steps": config.steps,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "release_boundary": "training-smoke only; no validated preGAN inference route is released",
    }
    metrics_path = Path(config.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
