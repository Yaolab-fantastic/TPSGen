from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_MOTIFS = [
    "CAAAA",
    "CTATT",
    "ATTTT",
    "TTAAA",
    "TTTAT",
]

TISSUE_CHOICES = ("root", "stem", "leaf", "fruit")


@dataclass(slots=True)
class AppConfig:
    canonical_length: int = 165
    motifs: list[str] = field(default_factory=lambda: list(DEFAULT_MOTIFS))
    target_tissues: tuple[str, ...] = TISSUE_CHOICES
    random_seed: int = 42
