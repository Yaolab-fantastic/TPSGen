from __future__ import annotations

import math
import os
import sys
import sysconfig
from pathlib import Path

import torch

from tpsgen.io.schema import LegacyPredictionResult, SequenceRecord


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
        / "tomato-promoter-designer"
        / "models"
    )
    if installed_models.exists():
        return installed_models
    return Path.cwd() / "models"


DEFAULT_DEEPSEED_MODULE_DIR = _default_models_dir() / "pregan_expression"
DEFAULT_DEEPSEED_CHECKPOINT = DEFAULT_DEEPSEED_MODULE_DIR / "165_mpra_expr_denselstm.pth"


def encode_sequence(sequence: str) -> torch.Tensor:
    """Encode a DNA sequence as a [4, L] one-hot tensor for the model adapter."""

    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
    encoded = torch.zeros((4, len(sequence)), dtype=torch.float32)
    for idx, base in enumerate(sequence.upper()):
        if base not in mapping:
            raise ValueError(
                "preGAN expression-constraint scorer only supports unambiguous A/C/G/T input. "
                f"Found unsupported base {base!r} at position {idx}."
            )
        encoded[mapping[base], idx] = 1.0
    return encoded


class PreganExpressionConstraintScorer:
    """Internal adapter around the preGAN expression-constraint DenseLSTM checkpoint."""

    backend_name = "pregan_expression_denselstm_scalar"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        module_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_DEEPSEED_CHECKPOINT)
        self.module_dir = Path(module_dir or DEFAULT_DEEPSEED_MODULE_DIR)
        self.device = torch.device(device)
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"preGAN expression-constraint checkpoint not found: {self.checkpoint_path}"
            )
        if not self.module_dir.exists():
            raise FileNotFoundError(
                f"preGAN expression-constraint module directory not found: {self.module_dir}"
            )
        module_dir_str = str(self.module_dir)
        if module_dir_str not in sys.path:
            sys.path.insert(0, module_dir_str)
        model = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model = model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        records: list[SequenceRecord],
        batch_size: int = 64,
    ) -> list[LegacyPredictionResult]:
        outputs: list[LegacyPredictionResult] = []

        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                encoded = torch.stack([encode_sequence(record.sequence) for record in batch]).to(
                    self.device
                )
                predicted = self.model(encoded).detach().cpu().tolist()
                for record, log2_expr in zip(batch, predicted):
                    linear_expr = math.pow(2.0, float(log2_expr))
                    outputs.append(
                        LegacyPredictionResult(
                            sequence_id=record.sequence_id,
                            sequence=record.sequence,
                            backend=self.backend_name,
                            predicted_log2_expression=round(float(log2_expr), 6),
                            predicted_linear_expression=round(linear_expr, 6),
                        )
                    )
        return outputs


DeepSeedScalarExpressionPredictor = PreganExpressionConstraintScorer
