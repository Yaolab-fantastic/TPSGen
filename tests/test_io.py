import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tomato_promoter_designer.io.schema import SequenceRecord, validate_records
from tomato_promoter_designer.io.fasta import read_fasta
from tomato_promoter_designer.preprocessing.kmer_encode import seq_to_kmers
from tomato_promoter_designer.preprocessing.extract_promoters import extract_gene_promoters, upstream_window


class TestIOAndSchema(unittest.TestCase):
    def test_validate_records_normalizes_case(self) -> None:
        records = [SequenceRecord("seq1", "acgttt")]
        validated = validate_records(records)
        self.assertEqual(validated[0].sequence, "ACGTTT")

    def test_seq_to_kmers(self) -> None:
        self.assertEqual(seq_to_kmers("ACGT", 2), ["AC", "CG", "GT"])

    def test_upstream_window_is_exact_and_rejects_boundary(self) -> None:
        self.assertEqual(upstream_window(200, 165), (35, 199))
        with self.assertRaises(ValueError):
            upstream_window(100, 165)

    def test_extract_promoters_handles_both_strands_and_skips_incomplete(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            genome = root / "genome.fa"
            gff = root / "genes.gff3"
            output = root / "promoters.fa"
            genome.write_text(">chr1\n" + "ACGT" * 150 + "\n", encoding="utf-8")
            gff.write_text(
                "##gff-version 3\n"
                "chr1\tsource\tgene\t200\t250\t.\t+\t.\tID=plus1\n"
                "chr1\tsource\tgene\t300\t350\t.\t-\t.\tID=minus1\n"
                "chr1\tsource\tgene\t10\t20\t.\t+\t.\tID=too_close\n",
                encoding="utf-8",
            )
            summary = extract_gene_promoters(genome, gff, output)
            extracted = read_fasta(output)
            self.assertEqual(summary, {"gene_records": 3, "retained_promoters": 2, "skipped_records": 1})
            self.assertEqual([record.sequence_id for record in extracted], ["plus1", "minus1"])
            self.assertTrue(all(len(record.sequence) == 165 for record in extracted))

    def test_extract_promoters_skips_duplicate_gene_ids(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "genome.fa").write_text(">chr1\n" + "ACGT" * 100 + "\n", encoding="utf-8")
            (root / "genes.gff3").write_text(
                "chr1\tsource\tgene\t200\t220\t.\t+\t.\tID=same\n"
                "chr1\tsource\tgene\t250\t270\t.\t+\t.\tID=same\n",
                encoding="utf-8",
            )
            summary = extract_gene_promoters(root / "genome.fa", root / "genes.gff3", root / "out.fa")
            self.assertEqual(summary["retained_promoters"], 1)
            self.assertEqual(summary["skipped_records"], 1)


if __name__ == "__main__":
    unittest.main()
