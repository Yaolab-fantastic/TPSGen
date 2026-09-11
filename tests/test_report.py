import json
import tempfile
import unittest
from pathlib import Path

from tomato_promoter_designer.io.csv import write_dict_rows
from tomato_promoter_designer.pipeline.report import build_report


class TestReport(unittest.TestCase):
    def test_build_report_adds_readable_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "design.csv"
            output_json = temp_path / "report.json"
            write_dict_rows(
                [
                    {
                        "sequence_id": "seq1",
                        "target_tissue": "fruit",
                        "candidate_rank": 1,
                        "original_sequence": "AAAA",
                        "designed_sequence": "AAAT",
                        "score_root": 1.0,
                        "score_stem": 1.2,
                        "score_leaf": 0.8,
                        "score_fruit": 2.1,
                        "preserved_motifs": "not_tracked",
                        "design_status": "mpravae_decoded",
                        "num_mutations": 1,
                        "passes_qc": True,
                    },
                    {
                        "sequence_id": "seq1",
                        "target_tissue": "fruit",
                        "candidate_rank": 2,
                        "original_sequence": "AAAA",
                        "designed_sequence": "AATT",
                        "score_root": 1.1,
                        "score_stem": 1.0,
                        "score_leaf": 0.9,
                        "score_fruit": 1.5,
                        "preserved_motifs": "not_tracked",
                        "design_status": "mpravae_repaired_by_reversion",
                        "num_mutations": 2,
                        "passes_qc": True,
                    },
                    {
                        "sequence_id": "seq2",
                        "target_tissue": "fruit",
                        "candidate_rank": 1,
                        "original_sequence": "CCCC",
                        "designed_sequence": "CCCT",
                        "score_root": 0.7,
                        "score_stem": 0.6,
                        "score_leaf": 0.5,
                        "score_fruit": 0.9,
                        "preserved_motifs": "motifA",
                        "design_status": "baseline_motif_preserving",
                        "num_mutations": 1,
                        "passes_qc": False,
                    },
                ],
                input_csv,
            )

            report = build_report(input_csv, output_json)

            self.assertEqual(report["num_unique_designed_sequences"], 3)
            self.assertEqual(report["num_qc_passed"], 2)
            self.assertAlmostEqual(report["qc_pass_rate"], 2 / 3, places=4)
            self.assertEqual(report["design_status_display_counts"]["TransVAE decoded"], 1)
            self.assertEqual(report["design_status_display_counts"]["QC-repaired candidate"], 1)
            self.assertEqual(report["design_status_display_counts"]["Baseline motif-preserving"], 1)
            self.assertEqual(len(report["best_candidate_by_sequence"]), 2)
            self.assertEqual(report["best_candidate_by_sequence"][0]["sequence_id"], "seq1")
            self.assertEqual(report["best_candidate_by_sequence"][0]["best_candidate_rank"], 1)
            self.assertEqual(report["best_candidate_by_sequence"][0]["design_status_label"], "TransVAE decoded")

            saved = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["best_candidate_by_sequence"][1]["sequence_id"], "seq2")


if __name__ == "__main__":
    unittest.main()
