from __future__ import annotations

from tpsgen.io.schema import MotifHit, SequenceRecord
from tpsgen.models.motif_annotator import MotifAnnotator


def run_annotation(records: list[SequenceRecord]) -> list[MotifHit]:
    annotator = MotifAnnotator()
    return annotator.annotate(records)

