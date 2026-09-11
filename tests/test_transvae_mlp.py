import tempfile
import unittest
from pathlib import Path

import torch

from tomato_promoter_designer.models.transvae_mlp import (
    TransVAEMLP,
    encode_dna,
    fruit_bias_fitness,
    load_transvae_mlp,
)


class TestTransVAEMLP(unittest.TestCase):
    def test_encode_dna_accepts_exact_unambiguous_sequence(self) -> None:
        encoded = encode_dna("ACGT" * 41 + "A")
        self.assertEqual(tuple(encoded.shape), (165,))
        self.assertEqual(encoded[:4].tolist(), [0, 1, 2, 3])

    def test_encode_dna_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 165-bp"):
            encode_dna("A" * 164)

    def test_encode_dna_rejects_ambiguous_bases(self) -> None:
        with self.assertRaisesRegex(ValueError, "only unambiguous"):
            encode_dna("A" * 164 + "N")

    def test_historical_prediction_head_is_deterministic_and_four_dimensional(self) -> None:
        model = TransVAEMLP().eval()
        tokens = encode_dna("ACGT" * 41 + "A").unsqueeze(0)
        with torch.no_grad():
            first = model.score_tokens(tokens)
            second = model.score_tokens(tokens)
        self.assertEqual(tuple(first.shape), (1, 4))
        torch.testing.assert_close(first, second)

    def test_scoring_rejects_non_165_token_tensors(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[batch, 165\]"):
            TransVAEMLP().score_tokens(torch.zeros((1, 164), dtype=torch.long))

    def test_checkpoint_loader_requires_an_exact_state_dict(self) -> None:
        model = TransVAEMLP()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(model.state_dict(), checkpoint)
            loaded = load_transvae_mlp(checkpoint)
        self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))

    def test_checkpoint_loader_rejects_missing_keys(self) -> None:
        state = TransVAEMLP().state_dict()
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "incomplete.pth"
            torch.save(state, checkpoint)
            with self.assertRaises(RuntimeError):
                load_transvae_mlp(checkpoint)

    def test_fruit_bias_fitness_matches_thesis_equation(self) -> None:
        scores = torch.tensor([[1.0, 2.0, 3.0, 7.0], [-1.0, 2.0, -0.5, 5.0]])
        fitness = fruit_bias_fitness(scores, negative_penalty=10.0)
        torch.testing.assert_close(fitness, torch.tensor([4.0, -12.0]))

    def test_fruit_bias_fitness_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[batch, 4\]"):
            fruit_bias_fitness(torch.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fruit_bias_fitness(torch.zeros((2, 4)), negative_penalty=-1.0)


if __name__ == "__main__":
    unittest.main()
