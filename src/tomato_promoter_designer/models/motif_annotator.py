from __future__ import annotations

from tomato_promoter_designer.config import AppConfig
from tomato_promoter_designer.io.schema import MotifHit, SequenceRecord


class MotifAnnotator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

    def annotate(self, records: list[SequenceRecord]) -> list[MotifHit]:
        hits: list[MotifHit] = []
        for record in records:
            for motif in self.config.motifs:
                start = 0
                while True:
                    idx = record.sequence.find(motif, start)
                    if idx == -1:
                        break
                    score = float(len(motif))
                    hits.append(
                        MotifHit(
                            sequence_id=record.sequence_id,
                            motif=motif,
                            start=idx,
                            end=idx + len(motif),
                            score=score,
                        )
                    )
                    start = idx + 1
        return hits

