import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from tomato_promoter_designer.training.pregan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    MaskedPromoterDataset,
    PreGANSmokeConfig,
    apply_masked_template,
    discriminator_wgan_gp_loss,
    encode_dna,
    encode_masked_template,
    freeze_predictor,
    generator_loss,
    run_pregan_smoke_training,
    sample_noise,
)


class TinyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence.mean(dim=(1, 2), keepdim=False).unsqueeze(1) * self.weight


class TestPreGANTrainingComponents(unittest.TestCase):
    def test_masked_template_preserves_fixed_bases(self) -> None:
        fixed, mutable = encode_masked_template("AMCGM")
        logits = torch.randn(1, 4, 5)
        generated = apply_masked_template(logits, fixed.unsqueeze(0), mutable.unsqueeze(0))

        self.assertTrue(torch.equal(generated[0, :, 0], encode_dna("A")[:, 0]))
        self.assertTrue(torch.equal(generated[0, :, 2], encode_dna("C")[:, 0]))
        self.assertTrue(torch.equal(generated[0, :, 3], encode_dna("G")[:, 0]))
        self.assertAlmostEqual(float(generated[0, :, 1].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(generated[0, :, 4].sum()), 1.0, places=6)

    def test_dataset_validates_fixed_template_matches_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "masked.csv"
            path.write_text(
                "realA,realB,expr\n"
                f"{'A' * 164}M,{'A' * 164}T,3.0\n",
                encoding="utf-8",
            )
            dataset = MaskedPromoterDataset(path)
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(item["fixed_bases"].shape), (4, 165))
        self.assertEqual(tuple(item["mutable_mask"].shape), (165,))
        self.assertTrue(bool(item["mutable_mask"][-1]))

    def test_dataset_rejects_fixed_template_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_masked.csv"
            path.write_text(
                "realA,realB,expr\n"
                f"C{'A' * 164},A{'A' * 164},3.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fixed template bases"):
                MaskedPromoterDataset(path)

    def test_wgan_losses_are_differentiable_and_predictor_is_frozen(self) -> None:
        torch.manual_seed(42)
        generator = ConditionalGenerator(noise_channels=3, hidden_channels=8)
        discriminator = ConditionalDiscriminator(hidden_channels=8)
        predictor = freeze_predictor(TinyPredictor())

        fixed, mutable = encode_masked_template("A" * 10 + "M" * 5)
        fixed_batch = fixed.unsqueeze(0).repeat(2, 1, 1)
        mutable_batch = mutable.unsqueeze(0).repeat(2, 1)
        real = encode_dna("A" * 10 + "T" * 5).unsqueeze(0).repeat(2, 1, 1)
        target_expression = torch.ones(2, 1)

        noise = sample_noise(2, generator.noise_channels, 15, device="cpu")
        logits = generator(noise, fixed_batch, mutable_batch)
        fake = apply_masked_template(logits, fixed_batch, mutable_batch)

        d_loss = discriminator_wgan_gp_loss(discriminator, real, fake, fixed_batch, mutable_batch)
        self.assertTrue(torch.isfinite(d_loss))

        g_loss = generator_loss(
            discriminator,
            predictor,
            fake,
            real,
            fixed_batch,
            mutable_batch,
            target_expression,
        )
        g_loss.backward()

        generator_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in generator.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(generator_grad, 0.0)
        self.assertFalse(any(parameter.requires_grad for parameter in predictor.parameters()))

    def test_smoke_training_writes_metadata_checkpoint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PreGANSmokeConfig(
                input_csv=str(repo_root / "data" / "raw" / "pregan_expression" / "pregan_smoke.csv"),
                output_checkpoint=str(Path(temp_dir) / "pregan_smoke.pt"),
                metrics_json=str(Path(temp_dir) / "pregan_smoke_metrics.json"),
                steps=1,
                batch_size=2,
                noise_channels=3,
                hidden_channels=8,
            )
            metrics = run_pregan_smoke_training(config, predictor=TinyPredictor())
            checkpoint = torch.load(metrics["checkpoint"], map_location="cpu", weights_only=False)

        self.assertEqual(metrics["num_records"], 4)
        self.assertIn("training-smoke only", metrics["release_boundary"])
        self.assertIn("generator_state_dict", checkpoint)
        self.assertIn("not a validated generation model", checkpoint["note"])


if __name__ == "__main__":
    unittest.main()
