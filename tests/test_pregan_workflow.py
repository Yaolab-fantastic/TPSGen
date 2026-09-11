import tempfile
import unittest
import csv
from pathlib import Path

from tpsgen.io.schema import SequenceRecord
from tpsgen.pipeline.pregan_workflow import run_pregan_workflow


class TestPreGANWorkflow(unittest.TestCase):
    def test_workflow_can_add_dnabert_evidence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "models" / "pregan" / "generator_checkpoint.pt"
        dnabert = root / "models" / "dnabert" / "pytorch_model.bin"
        if not checkpoint.exists() or not dnabert.exists():
            self.skipTest("required model checkpoints are not present")
        template = "M" * 12 + "A" * 153
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_pregan_workflow(
                [SequenceRecord("template_1", template)], checkpoint, 1, 42,
                directory, "fruit", "dnabert"
            )
            self.assertEqual(manifest["motif_backend"], "dnabert")
            self.assertTrue((Path(directory) / "dnabert_attention_evidence.csv").exists())
    def test_workflow_writes_candidates_and_scores(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "models" / "pregan" / "generator_checkpoint.pt"
        if not checkpoint.exists():
            self.skipTest("trained preGAN checkpoint is not present")
        template = "M" * 12 + "A" * 153
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_pregan_workflow(
                [SequenceRecord("template_1", template)], checkpoint, 2, 42, directory, "fruit"
            )
            output = Path(directory)
            self.assertEqual(manifest["num_candidates"], 2)
            self.assertEqual(manifest["target_tissue"], "fruit")
            self.assertTrue((output / "pregan_candidates.csv").exists())
            self.assertTrue((output / "pregan_candidates.fasta").exists())
            self.assertTrue((output / "transvae_candidate_scores.csv").exists())
            self.assertTrue((output / "manifest.json").exists())
            with (output / "pregan_candidates.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            eligible = [int(row["final_rank"]) for row in rows if row["final_rank"]]
            self.assertEqual(eligible, sorted(eligible))
            self.assertTrue(all(row["eligible_for_ranking"] == "True" for row in rows if row["final_rank"]))


if __name__ == "__main__":
    unittest.main()
