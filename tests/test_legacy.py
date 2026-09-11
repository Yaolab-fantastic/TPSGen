import unittest
from pathlib import Path

from tpsgen.io.schema import SequenceRecord
from tpsgen.legacy.pregan_expression import (
    DEFAULT_DEEPSEED_CHECKPOINT,
    PreganExpressionConstraintScorer,
    encode_sequence,
)


class TestLegacyAdapters(unittest.TestCase):
    def test_encode_sequence_shape(self) -> None:
        encoded = encode_sequence("ATCG")
        self.assertEqual(tuple(encoded.shape), (4, 4))
        self.assertEqual(float(encoded.sum().item()), 4.0)

    def test_pregan_expression_constraint_prediction_if_checkpoint_available(self) -> None:
        if not Path(DEFAULT_DEEPSEED_CHECKPOINT).exists():
            self.skipTest("Bundled preGAN expression-constraint checkpoint not available.")

        predictor = PreganExpressionConstraintScorer()
        results = predictor.predict(
            [
                SequenceRecord(
                    "seq1",
                    "AAATTGTAACAAATAATACAAAATATTTGTGAATACTAATGATTTCCAAATGGGATACCTTTTTGTTGTAAATAAGTGGAAAGGCAAAGTAGATAAATTCGCCTTTCCTAAGTATCCTTTTGGTCACAATTTCCAAGAGAAAAGAACAGAAAAGAAAAGAGAGAA",
                )
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].backend, "pregan_expression_denselstm_scalar")


if __name__ == "__main__":
    unittest.main()
