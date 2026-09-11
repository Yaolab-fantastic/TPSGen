from __future__ import annotations

from pathlib import Path

from tomato_promoter_designer.io.schema import PredictionResult, SequenceRecord
from tomato_promoter_designer.legacy.transvae_tomato import TransVAETomatoAdapter


def run_transvae_prediction(
    records: list[SequenceRecord],
    checkpoint_path: str | Path | None = None,
) -> list[PredictionResult]:
    predictor = TransVAETomatoAdapter(checkpoint_path=checkpoint_path)
    return predictor.predict(records)


run_legacy_mpravae_prediction = run_transvae_prediction
