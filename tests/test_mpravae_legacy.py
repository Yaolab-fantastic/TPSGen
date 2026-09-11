import unittest
from pathlib import Path

from tomato_promoter_designer.io.schema import SequenceRecord
from tomato_promoter_designer.legacy.transvae_tomato import (
    DEFAULT_TRANSVAE_CHECKPOINT,
    TransVAETomatoAdapter,
    decode_one_hot,
    is_promoter_like,
    one_hot_encode,
    promoter_qc_summary,
)


class TestTransVAEAdapter(unittest.TestCase):
    def test_one_hot_roundtrip(self) -> None:
        sequence = "ACGTACGT"
        encoded = one_hot_encode(sequence)
        self.assertEqual(tuple(encoded.shape), (4, 8))
        self.assertEqual(decode_one_hot(encoded), sequence)

    def test_promoter_like_returns_bool(self) -> None:
        sequence = "A" * 60 + "TATATAA" + "C" * 60
        self.assertIsInstance(is_promoter_like(sequence), bool)

    def test_training_informed_qc_accepts_tomato_like_example(self) -> None:
        sequence = (
            "AAATTGTAACAAATAATACAAAATATTTGTGAATACTAATGATTTCCAAATGGGATACCTTTTTGTT"
            "GTAAATAAGTGGAAAGGCAAAGTAGATAAATTCGCCTTTCCTAAGTATCCTTTTGGTCACAATTTCCA"
            "AGAGAAAAGAACAGAAAAGAAAAGAGAGAA"
        )
        qc = promoter_qc_summary(sequence)
        self.assertTrue(qc["passes"])
        self.assertGreaterEqual(qc["gc_fraction"], 0.12)
        self.assertLessEqual(qc["gc_fraction"], 0.52)

    def test_transvae_prediction_if_checkpoint_available(self) -> None:
        if not Path(DEFAULT_TRANSVAE_CHECKPOINT).exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")

        adapter = TransVAETomatoAdapter()
        results = adapter.predict(
            [
                SequenceRecord(
                    "seq1",
                    "AAATTGTAACAAATAATACAAAATATTTGTGAATACTAATGATTTCCAAATGGGATACCTTTTTGTTGTAAATAAGTGGAAAGGCAAAGTAGATAAATTCGCCTTTCCTAAGTATCCTTTTGGTCACAATTTCCAAGAGAAAAGAACAGAAAAGAAAAGAGAGAA",
                )
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertIn(results[0].preferred_tissue, {"root", "stem", "leaf", "fruit"})

    def test_transvae_prediction_is_deterministic_if_checkpoint_available(self) -> None:
        if not Path(DEFAULT_TRANSVAE_CHECKPOINT).exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")

        adapter = TransVAETomatoAdapter()
        record = SequenceRecord(
            "seq1",
            "AAATTGTAACAAATAATACAAAATATTTGTGAATACTAATGATTTCCAAATGGGATACCTTTTTGTTGTAAATAAGTGGAAAGGCAAAGTAGATAAATTCGCCTTTCCTAAGTATCCTTTTGGTCACAATTTCCAAGAGAAAAGAACAGAAAAGAAAAGAGAGAA",
        )
        first = adapter.predict([record])[0]
        second = adapter.predict([record])[0]
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_transvae_design_api_is_not_released(self) -> None:
        if not Path(DEFAULT_TRANSVAE_CHECKPOINT).exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")

        adapter = TransVAETomatoAdapter()
        record = SequenceRecord(
            "seq1",
            "AAATTGTAACAAATAATACAAAATATTTGTGAATACTAATGATTTCCAAATGGGATACCTTTTTGTTGTAAATAAGTGGAAAGGCAAAGTAGATAAATTCGCCTTTCCTAAGTATCCTTTTGGTCACAATTTCCAAGAGAAAAGAACAGAAAAGAAAAGAGAGAA",
        )
        with self.assertRaisesRegex(NotImplementedError, "not released as a validated design API"):
            adapter.design([record], target_tissue="fruit", candidates=2, seed=42)


if __name__ == "__main__":
    unittest.main()
