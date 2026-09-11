import json
import tempfile
import unittest
from pathlib import Path

from tpsgen.pipeline.legacy_figures import run_legacy_figure_export


class TestLegacyFigureExport(unittest.TestCase):
    def test_legacy_figure_bundle_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "legacy_figures"

            mpravae_loss = temp_path / "loss_history.csv"
            mpravae_loss.write_text(
                "train_loss,val_loss,train_recon,val_recon,train_kl,val_kl,train_pred,val_pred,train_trimer,val_trimer\n"
                "10,9,8,7,1,1,1.5,1.4,0.2,0.2\n"
                "8,7,6,5,0.8,0.7,1.2,1.1,0.1,0.1\n",
                encoding="utf-8",
            )

            designed = temp_path / "designed_promoters.csv"
            designed.write_text(
                "input_index,orig_sequence,pred_expr1,pred_expr2,pred_expr3,pred_expr4,generated_sequence\n"
                "0,AAAACCCCTTTTGGGG,1,1,1,2,AAAAGCCCTTTTGGGA\n"
                "1,TTTTAAAACCCCGGGG,1,1,1,2,TTTGAAAACCCCGGGA\n",
                encoding="utf-8",
            )

            pred = temp_path / "generated_prediction_results.csv"
            pred.write_text(
                "Generated_Seq,Real_Seq,Predicted_Expr,True_Expr\n"
                "AAAA,AAAT,1.0,1.2\n"
                "CCCC,CCCT,2.0,1.9\n",
                encoding="utf-8",
            )

            deepseed = temp_path / "deepseed_training.csv"
            deepseed.write_text(
                "train_loss,test_coefs,test_loss\n"
                "10,0.1,9\n"
                "8,0.2,7\n",
                encoding="utf-8",
            )

            mutated = temp_path / "mutated_file.csv"
            mutated.write_text(
                "orig_sequence,generated_sequence\n"
                "AAAACCCCTTTTGGGG,AAAAGCCCTTTTGGGA\n"
                "TTTTAAAACCCCGGGG,TTTGAAAACCCCGGGA\n",
                encoding="utf-8",
            )

            random_promoters = temp_path / "random_promoters_200.csv"
            random_promoters.write_text(
                "random_sequence\n"
                "ACGTACGTACGTACGT\n"
                "TGCATGCATGCATGCA\n",
                encoding="utf-8",
            )

            training_set = temp_path / "training_set.csv"
            training_set.write_text(
                "realB,expr_tissue_1,expr_tissue_2,expr_tissue_3,expr_tissue_4\n"
                "AAAACCCCTTTTGGGG,1,1,1,1\n"
                "TTTTAAAACCCCGGGG,1,1,1,1\n",
                encoding="utf-8",
            )

            motif_summary = temp_path / "motif_summary.csv"
            motif_summary.write_text(
                "motif,num_instances,enrichment_ratio,adjusted_p_value,indices,window_positions\n"
                "CAAAA,12,0.8,0.01,\"[1,2]\",\"[(1,5)]\"\n",
                encoding="utf-8",
            )

            tfbs_dir = temp_path / "tfbs"
            tfbs_dir.mkdir()
            (tfbs_dir / "TFBS_CAAAA.png").write_bytes(b"fakepng")

            deepseed_scatter = temp_path / "scatter_legacy.png"
            deepseed_scatter.write_bytes(b"fakepng")

            blast_dir = temp_path / "blast"
            blast_dir.mkdir()
            (blast_dir / "blast_example.png").write_bytes(b"fakepng")

            diversity_dir = temp_path / "diversity"
            diversity_dir.mkdir()
            (diversity_dir / "edit_distance_diversity.png").write_bytes(b"fakepng")

            manifest = run_legacy_figure_export(
                output_dir=output_dir,
                transvae_loss_history=mpravae_loss,
                transvae_designed_promoters=designed,
                transvae_prediction_results=pred,
                pregan_expression_training_log=deepseed,
                transvae_mutated_file=mutated,
                transvae_random_promoters=random_promoters,
                transvae_training_set=training_set,
                dnabert_motif_summary=motif_summary,
                dnabert_tfbs_dir=tfbs_dir,
                pregan_expression_scatter_png=deepseed_scatter,
                transvae_blast_dir=blast_dir,
                transvae_diversity_dir=diversity_dir,
            )
            generated = set(manifest["generated"])
            self.assertIn("transvae_loss_dashboard.svg", generated)
            self.assertIn("transvae_kmer_scatter.svg", generated)
            self.assertIn("transvae_prediction_scatter.svg", generated)
            self.assertIn("pregan_expression_training_curve.svg", generated)
            self.assertIn("transvae_edit_distance_diversity.svg", generated)
            self.assertIn("transvae_semantic_space.svg", generated)
            self.assertIn("dnabert_tfbs_assets/TFBS_CAAAA.png", generated)
            self.assertIn("scatter_legacy.png", generated)
            self.assertIn("blast/blast_example.png", generated)
            self.assertIn("diversity/edit_distance_diversity.png", generated)

            saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(saved_manifest["generated"]), generated)

            self.assertIn('width="1100" height="860"', (output_dir / "transvae_loss_dashboard.svg").read_text(encoding="utf-8"))
            self.assertIn('width="1040" height="700"', (output_dir / "pregan_expression_training_curve.svg").read_text(encoding="utf-8"))
            self.assertIn('width="780" height="660"', (output_dir / "transvae_prediction_scatter.svg").read_text(encoding="utf-8"))
            semantic_svg = (output_dir / "transvae_semantic_space.svg").read_text(encoding="utf-8")
            self.assertIn("Semantic Sequence Space", semantic_svg)
            self.assertIn("Principal component 1", semantic_svg)


if __name__ == "__main__":
    unittest.main()
