import tempfile
import unittest
import csv
from pathlib import Path

from tomato_promoter_designer.io.fasta import read_fasta
from tomato_promoter_designer.pipeline.integrated import run_integrated_workflow


class TestIntegratedWorkflow(unittest.TestCase):
    def test_run_with_dnabert_records_fresh_evidence(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        records = read_fasta(repo_root / "examples" / "demo_input.fasta")
        model = repo_root / "models" / "dnabert" / "pytorch_model.bin"
        if not model.exists():
            self.skipTest("DNABERT checkpoint is not present")
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_integrated_workflow(
                records, "fruit", 1, 42, temp_dir,
                scoring_backend="native", motif_backend="dnabert"
            )
            self.assertEqual(
                manifest["backends"]["motif_evidence"]["name"],
                "tomato_dnabert_6mer_attention_inference",
            )
            self.assertTrue((Path(temp_dir) / "motif" / "dnabert_attention_evidence.csv").exists())
    def test_run_writes_all_core_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        records = read_fasta(repo_root / "examples" / "demo_input.fasta")
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_integrated_workflow(records, "fruit", 2, 42, temp_dir)
            output = Path(temp_dir)
            expected = [
                "manifest.json",
                "input/validated_promoters.fasta",
                "motif/motif_annotations.csv",
                "design/candidate_metadata.csv",
                "design/designed_candidates.fasta",
                "scoring/original_scores.csv",
                "scoring/candidate_scores.csv",
                "reports/workflow_report.json",
                "reports/workflow_summary.csv",
            ]
            self.assertEqual(manifest["num_input_sequences"], 2)
            self.assertEqual(manifest["num_candidate_sequences"], 4)
            self.assertTrue(all((output / path).exists() for path in expected))

    def test_run_records_selected_scoring_backend(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        records = read_fasta(repo_root / "examples" / "demo_input.fasta")
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_integrated_workflow(records, "fruit", 1, 42, temp_dir)
            self.assertEqual(manifest["backends"]["tissue_scoring"]["name"], "package_native_deterministic_tissue_associated_scoring")
            with open(Path(temp_dir) / "scoring" / "all_sequence_scores.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["scoring_backend"] == "package_native_deterministic_tissue_associated_scoring" for row in rows))


if __name__ == "__main__":
    unittest.main()
