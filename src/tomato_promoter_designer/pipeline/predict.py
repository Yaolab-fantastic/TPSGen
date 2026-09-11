from __future__ import annotations

from tomato_promoter_designer.io.schema import PredictionResult, SequenceRecord
from tomato_promoter_designer.models.expression_predictor import HeuristicExpressionPredictor


def run_prediction(records: list[SequenceRecord]) -> list[PredictionResult]:
    predictor = HeuristicExpressionPredictor()
    return predictor.predict(records)
