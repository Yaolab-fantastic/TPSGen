from __future__ import annotations

import re
import os
import sysconfig
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpsgen.io.schema import DesignResult, PredictionResult, SequenceRecord
from tpsgen.models.transvae_mlp import (
    TISSUE_ORDER as MODEL_TISSUE_ORDER,
    encode_dna,
    load_transvae_mlp,
)


BASES = ("A", "C", "G", "T")
TISSUE_ORDER = MODEL_TISSUE_ORDER


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_models_dir() -> Path:
    override = os.environ.get("TPSGEN_MODELS_DIR")
    if override:
        return Path(override)
    repo_models = _repo_root() / "models"
    if repo_models.exists():
        return repo_models
    installed_models = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "tpsgen"
        / "models"
    )
    if installed_models.exists():
        return installed_models
    return Path.cwd() / "models"


DEFAULT_TRANSVAE_CHECKPOINT = _default_models_dir() / "transvae" / "best_val_corr_model.pth"

_TATA_PATTERN = re.compile(r"TATA[AT]A[AT]")
TOMATO_GC_RANGE = (0.12, 0.52)
TOMATO_MAX_HOMOPOLYMER = 13


def one_hot_encode(sequence: str) -> torch.Tensor:
    mapping = {
        "A": (1.0, 0.0, 0.0, 0.0),
        "C": (0.0, 1.0, 0.0, 0.0),
        "G": (0.0, 0.0, 1.0, 0.0),
        "T": (0.0, 0.0, 0.0, 1.0),
    }
    encoded = []
    for base in sequence.upper():
        if base not in mapping:
            raise ValueError(
                "TransVAE adapter only supports unambiguous A/C/G/T input. "
                f"Found unsupported base {base!r}."
            )
        encoded.append(mapping[base])
    return torch.tensor(encoded, dtype=torch.float32).transpose(0, 1)


def decode_one_hot(tensor: torch.Tensor) -> str:
    indices = tensor.argmax(dim=0).tolist()
    return "".join(BASES[index] for index in indices)


def softplus_scores(raw_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw_scores)


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


def max_homopolymer_length(sequence: str) -> int:
    if not sequence:
        return 0
    best = 1
    run = 1
    for previous, current in zip(sequence, sequence[1:]):
        if previous == current:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def midpoint_window(sequence: str) -> str:
    midpoint = len(sequence) // 2
    return sequence[max(0, midpoint - 45) : max(0, midpoint - 15)]


def promoter_qc_summary(
    sequence: str,
    gc_range: tuple[float, float] = TOMATO_GC_RANGE,
    max_poly: int = TOMATO_MAX_HOMOPOLYMER,
) -> dict[str, object]:
    window = midpoint_window(sequence)
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)
    max_homopolymer = max_homopolymer_length(sequence)
    tata_like_window = _TATA_PATTERN.search(window) is not None
    window_at_fraction = (window.count("A") + window.count("T")) / max(len(window), 1)
    passes = gc_range[0] <= gc <= gc_range[1] and max_homopolymer <= max_poly
    return {
        "passes": passes,
        "gc_fraction": round(gc, 4),
        "max_homopolymer": max_homopolymer,
        "tata_like_window": tata_like_window,
        "window_at_fraction": round(window_at_fraction, 4),
    }


def is_promoter_like(
    sequence: str,
    gc_range: tuple[float, float] = TOMATO_GC_RANGE,
    max_poly: int = TOMATO_MAX_HOMOPOLYMER,
) -> bool:
    return bool(promoter_qc_summary(sequence, gc_range=gc_range, max_poly=max_poly)["passes"])


class _LegacyConvolutionalVAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_out = self.conv(torch.zeros(1, 4, 165))
        self.flat_dim = dummy_out.shape[1]
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.conv(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar


class _LegacyConvolutionalVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, flat_dim: int = 5248) -> None:
        super().__init__()
        self.flat_dim = flat_dim
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.unflatten_shape = self._infer_unflatten_shape()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, self.unflatten_shape),
            nn.ConvTranspose1d(self.unflatten_shape[0], 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(size=165),
            nn.ConvTranspose1d(16, 4, kernel_size=7, padding=3),
            nn.Softmax(dim=1),
        )

    def _infer_unflatten_shape(self) -> tuple[int, int]:
        for channels in (64, 128, 32, 16):
            if self.flat_dim % channels == 0:
                return channels, self.flat_dim // channels
        raise ValueError(f"Cannot infer unflatten shape from flat_dim={self.flat_dim}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        latent = self.fc(z)
        return self.decoder(latent)


class _LegacyExpressionPredictor(nn.Module):
    def __init__(self, latent_dim: int = 64, num_tissues: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_tissues),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _LegacyJointPromoterModel(nn.Module):
    def __init__(self, latent_dim: int = 64, num_tissues: int = 4) -> None:
        super().__init__()
        self.encoder = _LegacyConvolutionalVAEEncoder(latent_dim=latent_dim)
        self.decoder = _LegacyConvolutionalVAEDecoder(latent_dim=latent_dim, flat_dim=self.encoder.flat_dim)
        self.predictor = _LegacyExpressionPredictor(latent_dim=latent_dim, num_tissues=num_tissues)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        predicted = self.predictor(z)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, predicted


class TransVAETomatoAdapter:
    """Adapter around the paper's Transformer-VAE plus MLP scorer.

    The older convolutional implementation remains private only so that old
    checkpoints fail clearly rather than being silently interpreted as the
    paper model. Released inference uses the strict Transformer-VAE loader.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        latent_dim: int = 128,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_TRANSVAE_CHECKPOINT)
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.model = self._load_model()

    def _load_model(self) -> JointPromoterModel:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"TransVAE checkpoint not found: {self.checkpoint_path}")
        try:
            model = load_transvae_mlp(self.checkpoint_path, device=self.device)
        except RuntimeError as error:
            raise RuntimeError(
                "The supplied checkpoint is not compatible with the released "
                "Transformer-VAE/MLP architecture. Use the bundled tomato "
                "checkpoint or provide a checkpoint with the same state_dict schema."
            ) from error
        return model

    def _tensorize(self, record: SequenceRecord) -> torch.Tensor:
        return encode_dna(record.sequence).unsqueeze(0).to(self.device)

    def _predict_scores(self, input_tensor: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            raw_scores = self.model.score_tokens(input_tensor)
            scores = raw_scores[0].detach().cpu().tolist()
        return dict(zip(TISSUE_ORDER, [round(float(score), 6) for score in scores]))

    def _predict_one(self, record: SequenceRecord) -> PredictionResult:
        tissue_scores = self._predict_scores(self._tensorize(record))
        preferred_tissue = max(tissue_scores, key=tissue_scores.get)
        return PredictionResult(
            sequence_id=record.sequence_id,
            sequence=record.sequence,
            score_root=tissue_scores["root"],
            score_stem=tissue_scores["stem"],
            score_leaf=tissue_scores["leaf"],
            score_fruit=tissue_scores["fruit"],
            preferred_tissue=preferred_tissue,
        )

    def predict(self, records: list[SequenceRecord]) -> list[PredictionResult]:
        results: list[PredictionResult] = []
        for record in records:
            results.append(self._predict_one(record))
        return results

    def design(
        self,
        records: list[SequenceRecord],
        target_tissue: str,
        candidates: int = 5,
        seed: int = 42,
        steps: int = 400,
        learning_rate: float = 0.05,
        gamma: float = 0.05,
        beta: float = 0.1,
        hi_lim: float = 10.0,
    ) -> list[DesignResult]:
        raise NotImplementedError(
            "TransVAE latent-space candidate generation is not released as a "
            "validated design API in the current package. Use predict() for "
            "checkpoint-backed four-tissue scoring and the package-native "
            "design command for released candidate generation."
        )


DEFAULT_MPRAVAE_CHECKPOINT = DEFAULT_TRANSVAE_CHECKPOINT
MpraVAETomatoAdapter = TransVAETomatoAdapter
