from __future__ import annotations

from collections import Counter

from tpsgen.io.schema import PredictionResult, SequenceRecord


class HeuristicExpressionPredictor:
    """A deterministic baseline scorer.

    This transparent operational baseline keeps the core workflow runnable
    without invoking a checkpoint-backed model.
    """

    def predict(self, records: list[SequenceRecord]) -> list[PredictionResult]:
        return [self.predict_one(record) for record in records]

    def predict_one(self, record: SequenceRecord) -> PredictionResult:
        seq = record.sequence
        length = max(len(seq), 1)
        base_counts = Counter(seq)
        gc_ratio = (base_counts["G"] + base_counts["C"]) / length
        at_ratio = (base_counts["A"] + base_counts["T"]) / length
        poly_a = seq.count("AAAAA") / max(length - 4, 1)
        motif_counts = {motif: seq.count(motif) for motif in {"CAAAA", "CTATT", "ATTTT", "TTAAA", "TTTAT"}}

        root = 4.2 * gc_ratio + 2.0 * motif_counts["CTATT"] + 0.5 * motif_counts["TTAAA"]
        stem = 3.2 * at_ratio + 1.8 * motif_counts["TTTAT"] + 0.6 * motif_counts["ATTTT"]
        leaf = 2.5 * gc_ratio + 1.1 * motif_counts["CAAAA"] + 0.7 * motif_counts["CTATT"]
        fruit = 3.5 * at_ratio + 2.2 * motif_counts["CAAAA"] + 1.3 * motif_counts["ATTTT"] + 2.0 * poly_a

        scores = {
            "root": round(root, 4),
            "stem": round(stem, 4),
            "leaf": round(leaf, 4),
            "fruit": round(fruit, 4),
        }
        preferred_tissue = max(scores, key=scores.get)
        return PredictionResult(
            sequence_id=record.sequence_id,
            sequence=seq,
            score_root=scores["root"],
            score_stem=scores["stem"],
            score_leaf=scores["leaf"],
            score_fruit=scores["fruit"],
            preferred_tissue=preferred_tissue,
        )
