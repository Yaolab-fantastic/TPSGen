import tempfile
import unittest
from pathlib import Path

from tomato_promoter_designer.visualization.blast_png import render_legacy_blast_figure_pair


class TestBlastPng(unittest.TestCase):
    def test_render_legacy_blast_figure_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sequence_csv = temp_path / "sequences_with_id.csv"
            blast_txt = temp_path / "blast_result.txt"
            output_dir = temp_path / "blast"

            sequence_csv.write_text(
                "ID,Sequence\n"
                "Natural_1,AAAA\n"
                "VAE_1,CCCC\n"
                "preGAN_promoter1,GGGG\n"
                "WGAN_1,TTTT\n"
                "Random_1,ACGT\n",
                encoding="utf-8",
            )
            blast_txt.write_text(
                "Natural_1\tchr1\t100\t4\t0\t0\t1\t4\t1\t4\t1e-40\t100\n"
                "VAE_1\tchr1\t100\t4\t0\t0\t1\t4\t1\t4\t1e-10\t100\n"
                "preGAN_promoter1\tchr1\t100\t4\t0\t0\t1\t4\t1\t4\t1e-20\t100\n"
                "WGAN_1\tchr1\t100\t4\t0\t0\t1\t4\t1\t4\t1e-15\t100\n",
                encoding="utf-8",
            )

            outputs = render_legacy_blast_figure_pair(sequence_csv, blast_txt, output_dir)
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                self.assertTrue(output.exists())
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
