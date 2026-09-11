import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from Bio import __version__ as biopython_version

from tpsgen.legacy.dnabert_motif import (
    analyze_attention_dataset,
    kmer2seq,
    read_dnabert_dev_tsv,
    run_dnabert_legacy_annotation,
    validate_dnabert_resources,
)


class TestDNABERTLegacyAdapter(unittest.TestCase):
    def test_supported_biopython_alignment_version(self) -> None:
        major, minor = (int(part) for part in biopython_version.split(".")[:2])
        self.assertGreaterEqual((major, minor), (1, 85))

    def test_kmer2seq(self) -> None:
        self.assertEqual(kmer2seq("ATC TCG CGA"), "ATCGA")

    def test_validate_dnabert_resources_checks_matching_rows(self) -> None:
        dev_tsv = self._write_dev_tsv()
        with tempfile.TemporaryDirectory() as temp_dir:
            attention_path = Path(temp_dir) / "atten.npy"
            np.save(attention_path, np.zeros((4, 8), dtype=float))
            summary = validate_dnabert_resources(dev_tsv, attention_path, expected_length=8)
            self.assertEqual(summary["num_sequences"], 4)
            self.assertEqual(summary["attention_shape"], [4, 8])
            self.assertFalse(summary["tomato_specificity_verified"])

    def test_small_attention_analysis(self) -> None:
        dataset = read_dnabert_dev_tsv(self._write_dev_tsv())
        attention = np.array(
            [
                [0.01, 0.01, 0.6, 0.7, 0.8, 0.7, 0.01, 0.01],
                [0.01, 0.01, 0.6, 0.7, 0.8, 0.7, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            dtype=float,
        )
        motif_records, labeled_sequences, metadata = analyze_attention_dataset(
            dataset,
            attention_scores=attention,
            window_size=4,
            min_len=3,
            p_value_cutoff=0.5,
            min_n_motif=2,
        )
        self.assertEqual(metadata["num_positive_sequences"], 2)
        self.assertTrue(len(motif_records) >= 1)
        self.assertEqual(len(labeled_sequences), 2)
        self.assertIn("M", labeled_sequences[0].labeled_sequence)

    def test_run_annotation_writes_outputs(self) -> None:
        dev_tsv = self._write_dev_tsv()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            atten_path = Path(temp_dir) / "atten.npy"
            np.save(
                atten_path,
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
            metadata = run_dnabert_legacy_annotation(
                dev_tsv_path=dev_tsv,
                attention_scores_path=atten_path,
                output_dir=output_dir,
                window_size=4,
                min_len=3,
                p_value_cutoff=0.5,
                min_n_motif=2,
            )
            self.assertTrue((output_dir / "motif_summary.csv").exists())
            self.assertTrue((output_dir / "processed_sequences.csv").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())
            self.assertEqual(json.loads((output_dir / "run_metadata.json").read_text())["num_positive_sequences"], 2)
            self.assertEqual(metadata["num_negative_sequences"], 2)

    def test_nonsignificant_motifs_are_not_returned_as_fallback(self) -> None:
        dataset = read_dnabert_dev_tsv(self._write_dev_tsv())
        attention = np.full((4, 8), 0.01, dtype=float)
        attention[:2, 2:6] = 0.7

        motif_records, _, metadata = analyze_attention_dataset(
            dataset,
            attention_scores=attention,
            window_size=4,
            min_len=3,
            p_value_cutoff=0.0,
            min_n_motif=1,
        )

        self.assertEqual(motif_records, [])
        self.assertEqual(metadata["num_significant_exact_motifs"], 0)
        self.assertEqual(metadata["num_retained_motifs"], 0)
        self.assertFalse(metadata["used_fallback_ranking"])

    def test_retained_dnabert_resource_counts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dev_tsv = repo_root / "data" / "raw" / "dnabert" / "dev.tsv"
        atten_npy = repo_root / "data" / "raw" / "dnabert" / "atten.npy"
        if not dev_tsv.exists() or not atten_npy.exists():
            self.skipTest("Retained DNABERT resources are not available.")

        dataset = read_dnabert_dev_tsv(dev_tsv)
        motif_records, _, metadata = analyze_attention_dataset(
            dataset,
            attention_scores=np.load(atten_npy),
        )
        top = sorted(motif_records, key=lambda record: record.num_instances, reverse=True)[:5]

        self.assertEqual(metadata["num_total_sequences"], 10222)
        self.assertEqual(metadata["num_extracted_exact_motifs"], 8096)
        self.assertEqual(metadata["num_significant_exact_motifs"], 787)
        self.assertEqual(metadata["num_merged_motif_groups"], 77)
        self.assertEqual(metadata["num_retained_motifs"], 53)
        self.assertEqual(
            [(record.motif, record.num_instances) for record in top],
            [
                ("ACTATA", 127),
                ("CTCAAA", 126),
                ("TAATTT", 96),
                ("ACTTAT", 95),
                ("TTAAA", 90),
            ],
        )

    def _write_dev_tsv(self) -> Path:
        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir) / "dev.tsv"
        path.write_text(
            "sequence\tlabel\n"
            "AAA AAC ACG CGT GTT TTA\t1\n"
            "AAA AAC ACG CGT GTT TTA\t1\n"
            "CCC CCG CGA GAA AAA AAT\t0\n"
            "GGG GGA GAT ATT TTT TTA\t0\n",
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
