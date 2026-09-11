import json
import tempfile
import unittest
from pathlib import Path

from tomato_promoter_designer.io.csv import write_dict_rows
from tomato_promoter_designer.pipeline.figures import run_figure_export


class TestFigureExport(unittest.TestCase):
    def test_prediction_figure_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "predict.csv"
            output_dir = temp_path / "predict_figures"
            write_dict_rows(
                [
                    {
                        "sequence_id": "seq1",
                        "sequence": "ACGT",
                        "score_root": 1.0,
                        "score_stem": 2.0,
                        "score_leaf": 1.5,
                        "score_fruit": 0.9,
                        "preferred_tissue": "stem",
                    }
                ],
                input_csv,
            )
            manifest = run_figure_export(input_csv, output_dir)
            self.assertEqual(manifest["figure_type"], "prediction")
            self.assertTrue((output_dir / "prediction_heatmap.svg").exists())

    def test_design_figure_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "design.csv"
            output_dir = temp_path / "design_figures"
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
                        "score_fruit": 1.9,
                        "preserved_motifs": "none",
                        "design_status": "mpravae_decoded",
                        "num_mutations": 1,
                        "passes_qc": True,
                    }
                ],
                input_csv,
            )
            manifest = run_figure_export(input_csv, output_dir)
            self.assertEqual(manifest["figure_type"], "design")
            self.assertTrue((output_dir / "design_summary.svg").exists())

    def test_motif_figure_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "motif_summary.csv"
            output_dir = temp_path / "motif_figures"
            write_dict_rows(
                [
                    {
                        "motif": "CAAAA",
                        "num_instances": 12,
                        "enrichment_ratio": 0.8,
                        "adjusted_p_value": 0.01,
                        "indices": "[1,2]",
                        "window_positions": "[(1,5)]",
                    }
                ],
                input_csv,
            )
            manifest = run_figure_export(input_csv, output_dir, top_n=10)
            self.assertEqual(manifest["figure_type"], "motif")
            self.assertTrue((output_dir / "motif_enrichment.svg").exists())
            saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["files"], ["motif_enrichment.svg"])


if __name__ == "__main__":
    unittest.main()
