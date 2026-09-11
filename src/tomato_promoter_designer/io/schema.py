from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


ALLOWED_BASES = set("ACGTNM")


@dataclass(slots=True)
class SequenceRecord:
    sequence_id: str
    sequence: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class MotifHit:
    sequence_id: str
    motif: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PredictionResult:
    sequence_id: str
    sequence: str
    score_root: float
    score_stem: float
    score_leaf: float
    score_fruit: float
    preferred_tissue: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DesignResult:
    sequence_id: str
    target_tissue: str
    candidate_rank: int
    original_sequence: str
    designed_sequence: str
    score_root: float
    score_stem: float
    score_leaf: float
    score_fruit: float
    preserved_motifs: str
    design_status: str = ""
    num_mutations: int = 0
    passes_qc: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LegacyPredictionResult:
    sequence_id: str
    sequence: str
    backend: str
    predicted_log2_expression: float
    predicted_linear_expression: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LegacyMotifRecord:
    motif: str
    num_instances: int
    enrichment_ratio: float
    adjusted_p_value: float
    indices: list[int]
    window_positions: list[tuple[int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "motif": self.motif,
            "num_instances": self.num_instances,
            "enrichment_ratio": self.enrichment_ratio,
            "adjusted_p_value": self.adjusted_p_value,
            "indices": str(self.indices),
            "window_positions": str(self.window_positions),
        }


@dataclass(slots=True)
class LegacyLabeledSequence:
    sequence_id: int
    label: int
    original_sequence: str
    labeled_sequence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper()


def validate_sequence(sequence: str) -> None:
    cleaned = normalize_sequence(sequence)
    if not cleaned:
        raise ValueError("Sequence is empty.")
    invalid = sorted(set(cleaned) - ALLOWED_BASES)
    if invalid:
        raise ValueError(f"Sequence contains unsupported symbols: {''.join(invalid)}")


def validate_records(records: Iterable[SequenceRecord]) -> list[SequenceRecord]:
    validated = []
    seen = set()
    for record in records:
        if record.sequence_id in seen:
            raise ValueError(f"Duplicate sequence id detected: {record.sequence_id}")
        validate_sequence(record.sequence)
        seen.add(record.sequence_id)
        validated.append(SequenceRecord(record.sequence_id, normalize_sequence(record.sequence)))
    return validated
