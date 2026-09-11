from __future__ import annotations

from tomato_promoter_designer.io.schema import DesignResult, SequenceRecord
from tomato_promoter_designer.models.generator import MotifPreservingDesigner


def run_design(
    records: list[SequenceRecord],
    target_tissue: str,
    candidates: int = 5,
    seed: int | None = None,
) -> list[DesignResult]:
    designer = MotifPreservingDesigner()
    return designer.design(records, target_tissue=target_tissue, candidates=candidates, seed=seed)
