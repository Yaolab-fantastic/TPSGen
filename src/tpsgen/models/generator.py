from __future__ import annotations

import random

from tpsgen.config import AppConfig
from tpsgen.io.schema import DesignResult, SequenceRecord
from tpsgen.models.expression_predictor import HeuristicExpressionPredictor
from tpsgen.models.motif_annotator import MotifAnnotator


class MotifPreservingDesigner:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.predictor = HeuristicExpressionPredictor()
        self.annotator = MotifAnnotator(self.config)

    def design(
        self,
        records: list[SequenceRecord],
        target_tissue: str,
        candidates: int = 5,
        seed: int | None = None,
    ) -> list[DesignResult]:
        if target_tissue not in self.config.target_tissues:
            raise ValueError(f"Unsupported target tissue: {target_tissue}")

        random_seed = self.config.random_seed if seed is None else seed
        rng = random.Random(random_seed)
        results: list[DesignResult] = []

        for record in records:
            protected = self._protected_positions(record)
            ranked = self._rank_candidates(record, target_tissue, candidates, rng, protected)
            results.extend(ranked)
        return results

    def _protected_positions(self, record: SequenceRecord) -> set[int]:
        protected: set[int] = set()
        for hit in self.annotator.annotate([record]):
            protected.update(range(hit.start, hit.end))
        return protected

    def _rank_candidates(
        self,
        record: SequenceRecord,
        target_tissue: str,
        candidates: int,
        rng: random.Random,
        protected: set[int],
    ) -> list[DesignResult]:
        pool: list[tuple[float, str, dict[str, float]]] = []
        for _ in range(max(candidates * 3, candidates)):
            designed = self._mutate(record.sequence, target_tissue, rng, protected)
            prediction = self.predictor.predict_one(SequenceRecord(record.sequence_id, designed))
            score_map = {
                "root": prediction.score_root,
                "stem": prediction.score_stem,
                "leaf": prediction.score_leaf,
                "fruit": prediction.score_fruit,
            }
            pool.append((score_map[target_tissue], designed, score_map))

        pool.sort(key=lambda item: item[0], reverse=True)
        motif_names = sorted({hit.motif for hit in self.annotator.annotate([record])})
        preserved_motifs = ",".join(motif_names) if motif_names else "none"
        ranked: list[DesignResult] = []

        for idx, (_, designed, score_map) in enumerate(pool[:candidates], start=1):
            num_mutations = sum(base != original for base, original in zip(designed, record.sequence))
            ranked.append(
                DesignResult(
                    sequence_id=record.sequence_id,
                    target_tissue=target_tissue,
                    candidate_rank=idx,
                    original_sequence=record.sequence,
                    designed_sequence=designed,
                    score_root=score_map["root"],
                    score_stem=score_map["stem"],
                    score_leaf=score_map["leaf"],
                    score_fruit=score_map["fruit"],
                    preserved_motifs=preserved_motifs,
                    design_status="baseline_motif_preserving",
                    num_mutations=num_mutations,
                    passes_qc=self._passes_qc(designed),
                )
            )
        return ranked

    @staticmethod
    def _passes_qc(sequence: str) -> bool:
        """Apply the lightweight sequence checks used by the public baseline route."""
        if not sequence or set(sequence) - set("ACGT"):
            return False
        gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
        longest = 1
        current = 1
        for previous, base in zip(sequence, sequence[1:]):
            current = current + 1 if previous == base else 1
            longest = max(longest, current)
        return 0.12 <= gc_fraction <= 0.52 and longest <= 13

    def _mutate(
        self,
        sequence: str,
        target_tissue: str,
        rng: random.Random,
        protected: set[int],
    ) -> str:
        favored = {
            "root": ("G", "C", "A", "T"),
            "stem": ("T", "A", "C", "G"),
            "leaf": ("C", "G", "A", "T"),
            "fruit": ("A", "T", "A", "C"),
        }[target_tissue]
        chars = list(sequence)
        mutation_rate = 0.18
        for idx, base in enumerate(chars):
            if idx in protected:
                continue
            if rng.random() < mutation_rate:
                chars[idx] = rng.choice(favored)
            elif base not in "ACGT":
                chars[idx] = rng.choice(favored[:2])
        return "".join(chars)
