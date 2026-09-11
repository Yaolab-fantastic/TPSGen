from __future__ import annotations

from tpsgen.io.schema import PredictionResult, SequenceRecord
from tpsgen.models.expression_predictor import HeuristicExpressionPredictor


def run_prediction(records: list[SequenceRecord]) -> list[PredictionResult]:
    predictor = HeuristicExpressionPredictor()
    return predictor.predict(records)
