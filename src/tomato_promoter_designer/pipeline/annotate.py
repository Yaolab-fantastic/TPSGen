from __future__ import annotations

from tomato_promoter_designer.io.schema import MotifHit, SequenceRecord
from tomato_promoter_designer.models.motif_annotator import MotifAnnotator


def run_annotation(records: list[SequenceRecord]) -> list[MotifHit]:
    annotator = MotifAnnotator()
    return annotator.annotate(records)

