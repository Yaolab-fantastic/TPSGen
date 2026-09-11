import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestCLI(unittest.TestCase):
    def test_validate_models_reports_runtime_boundaries(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        result = subprocess.run(
            [sys.executable, "-m", "tpsgen.cli", "validate-models"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["transvae"]["available_for_runtime_scoring"])
        self.assertTrue(report["dnabert"]["available_for_arbitrary_fasta"])
        self.assertTrue(report["pregan"]["available_for_arbitrary_fasta_generation"])
    def test_copy_example_command_writes_bundled_fasta(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "demo_input.fasta"
            cmd = [
                sys.executable,
                "-m",
                "tpsgen.cli",
                "copy-example",
                "--output",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                (repo_root / "examples" / "demo_input.fasta").read_text(encoding="utf-8"),
            )

    def test_validate_input_command(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        fasta_path = repo_root / "examples" / "demo_input.fasta"
        cmd = [
            sys.executable,
            "-m",
            "tpsgen.cli",
            "validate-input",
            "--input",
            str(fasta_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["num_records"], 2)

    def test_extract_promoters_command_writes_fasta_and_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "genome.fa").write_text(">chr1\n" + "ACGT" * 100 + "\n", encoding="utf-8")
            (root / "genes.gff3").write_text(
                "chr1\tsource\tgene\t200\t220\t.\t+\t.\tID=plus1\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tpsgen.cli",
                    "extract-promoters",
                    "--genome",
                    str(root / "genome.fa"),
                    "--annotation",
                    str(root / "genes.gff3"),
                    "--output",
                    str(root / "promoters.fa"),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            summary = json.loads(result.stdout)
            self.assertEqual(summary["retained_promoters"], 1)
            self.assertEqual(len((root / "promoters.fa").read_text().splitlines()[1]), 165)

    def test_integrated_run_transvae_backend_writes_model_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint = repo_root / "models" / "transvae" / "best_val_corr_model.pth"
        if not checkpoint.exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tpsgen.cli",
                    "run",
                    "--input",
                    str(repo_root / "examples" / "demo_input.fasta"),
                    "--target",
                    "fruit",
                    "--candidates",
                    "1",
                    "--scoring-backend",
                    "transvae",
                    "--output",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["backends"]["tissue_scoring"]["checkpoint_used"], True)
            rows = (output_dir / "scoring" / "all_sequence_scores.csv").read_text(encoding="utf-8")
            self.assertIn("tomato_transvae_mlp_checkpoint_scoring", rows)

    def test_annotate_command_writes_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        fasta_path = repo_root / "examples" / "demo_input.fasta"
        output_path = repo_root / "tmp" / "annotate_cli_test.csv"
        if output_path.exists():
            output_path.unlink()

        cmd = [
            sys.executable,
            "-m",
            "tpsgen.cli",
            "annotate",
            "--input",
            str(fasta_path),
            "--output",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        self.assertTrue(output_path.exists())
        output_path.unlink()

    def test_predict_transvae_command_writes_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = repo_root / "models" / "transvae" / "best_val_corr_model.pth"
        if not checkpoint_path.exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        fasta_path = repo_root / "examples" / "demo_input.fasta"
        output_path = repo_root / "tmp" / "predict_transvae_cli_test.csv"
        if output_path.exists():
            output_path.unlink()

        cmd = [
            sys.executable,
            "-m",
            "tpsgen.cli",
            "predict-transvae",
            "--input",
            str(fasta_path),
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        self.assertTrue(output_path.exists())
        output_path.unlink()

    def test_predict_transvae_rejects_noncanonical_length(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = repo_root / "models" / "transvae" / "best_val_corr_model.pth"
        if not checkpoint_path.exists():
            self.skipTest("Bundled TransVAE checkpoint not available.")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = Path(temp_dir) / "bad_length.fasta"
            output_path = Path(temp_dir) / "bad_length.csv"
            fasta_path.write_text(">too_short\nACGTACGTACGT\n", encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "tpsgen.cli",
                "predict-transvae",
                "--input",
                str(fasta_path),
                "--output",
                str(output_path),
                "--checkpoint",
                str(checkpoint_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TransVAE commands require", result.stderr)
        self.assertIn("too_short=12 bp", result.stderr)

    def test_integrated_run_rejects_noncanonical_length(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = Path(temp_dir) / "bad_length.fasta"
            fasta_path.write_text(">too_short\nACGTACGTACGT\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tpsgen.cli",
                    "run",
                    "--input",
                    str(fasta_path),
                    "--target",
                    "fruit",
                    "--output",
                    str(Path(temp_dir) / "out"),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TransVAE commands require", result.stderr)

    def test_annotate_legacy_dnabert_command_writes_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dev_tsv = temp_path / "dev.tsv"
            atten_npy = temp_path / "atten.npy"
            output_dir = temp_path / "out"
            dev_tsv.write_text(
                "sequence\tlabel\n"
                "AAA AAC ACG CGT GTT TTA\t1\n"
                "AAA AAC ACG CGT GTT TTA\t1\n"
                "CCC CCG CGA GAA AAA AAT\t0\n"
                "GGG GGA GAT ATT TTT TTA\t0\n",
                encoding="utf-8",
            )
            np.save(
                atten_npy,
                np.array(
                    [
                        [0.01, 0.01, 0.6, 0.7, 0.8, 0.7, 0.01, 0.01],
                        [0.01, 0.01, 0.6, 0.7, 0.8, 0.7, 0.01, 0.01],
                        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                    ],
                    dtype=float,
                ),
            )

            cmd = [
                sys.executable,
                "-m",
                "tpsgen.cli",
                "annotate-dnabert",
                "--dev-tsv",
                str(dev_tsv),
                "--atten-npy",
                str(atten_npy),
                "--output-dir",
                str(output_dir),
                "--window-size",
                "4",
                "--min-len",
                "3",
                "--pval-cutoff",
                "0.5",
                "--min-n-motif",
                "2",
            ]
            subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            self.assertTrue((output_dir / "motif_summary.csv").exists())
            self.assertTrue((output_dir / "processed_sequences.csv").exists())

    def test_figures_command_writes_svg(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "predict.csv"
            output_dir = temp_path / "figures"
            input_csv.write_text(
                "sequence_id,sequence,score_root,score_stem,score_leaf,score_fruit,preferred_tissue\n"
                "seq1,ACGT,1.0,2.0,1.5,0.8,stem\n",
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "tpsgen.cli",
                "figures",
                "--input",
                str(input_csv),
                "--output-dir",
                str(output_dir),
            ]
            subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            self.assertTrue((output_dir / "prediction_heatmap.svg").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

    def test_legacy_figures_command_writes_svg(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "legacy_figures"
            cmd = [
                sys.executable,
                "-m",
                "tpsgen.cli",
                "model-figures",
                "--output-dir",
                str(output_dir),
                "--transvae-loss-history",
                str(repo_root / "data" / "raw" / "transvae" / "loss_history.csv"),
                "--transvae-designed-promoters",
                str(repo_root / "data" / "raw" / "transvae" / "designed_promoters.csv"),
                "--transvae-prediction-results",
                str(repo_root / "data" / "raw" / "transvae" / "generated_prediction_results.csv"),
                "--pregan-expression-training-log",
                str(repo_root / "data" / "raw" / "pregan_expression" / "training_log165_mpra_expr_denselstm.csv"),
                "--transvae-mutated-file",
                str(repo_root / "data" / "raw" / "transvae" / "mutated_file.csv"),
                "--transvae-random-promoters",
                str(repo_root / "data" / "raw" / "transvae" / "random_promoters_200.csv"),
                "--transvae-training-set",
                str(repo_root / "data" / "raw" / "transvae" / "training_set.csv"),
                "--dnabert-motif-summary",
                str(repo_root / "data" / "processed" / "dnabert_legacy" / "motif_summary.csv"),
                "--dnabert-tfbs-dir",
                str(repo_root / "data" / "raw" / "dnabert" / "tfbs_assets"),
            ]
            subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "transvae_loss_dashboard.svg").exists())
            self.assertTrue((output_dir / "transvae_edit_distance_diversity.svg").exists())
            self.assertTrue((output_dir / "transvae_semantic_space.svg").exists())


if __name__ == "__main__":
    unittest.main()
