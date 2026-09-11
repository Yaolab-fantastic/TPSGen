from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

from tomato_promoter_designer.io.fasta import write_fasta
from tomato_promoter_designer.io.schema import SequenceRecord


@dataclass(slots=True)
class PromoterWindow:
    chromosome: str
    gene_id: str
    strand: str
    start: int
    end: int


def upstream_window(gene_start: int, window_size: int = 165) -> tuple[int, int]:
    """Return the retained positive-strand upstream promoter window.

    The retained MpraVAE training resource used ITAG3.2 positive-strand gene
    entries and treated gene_start as the TSS proxy. For negative-strand user
    inputs, extract the downstream reference-genome window and reverse-
    complement it before writing FASTA; this helper does not implement that
    extension.
    """
    if gene_start <= 1:
        raise ValueError("Gene start must be positive.")
    promoter_end = gene_start - 1
    promoter_start = promoter_end - window_size + 1
    if promoter_start < 1:
        raise ValueError("Gene is too close to the chromosome start for a complete promoter window.")
    return promoter_start, promoter_end


def extract_gene_promoters(
    genome_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
    window_size: int = 165,
    include_negative_strand: bool = True,
) -> dict[str, int]:
    """Extract complete promoter windows from GFF3 ``gene`` records.

    Coordinates in GFF3 are 1-based and inclusive. Output records are written
    in promoter-to-gene orientation, with the negative-strand window reverse
    complemented. Records with missing chromosomes, invalid coordinates or
    incomplete windows are skipped rather than padded.
    """
    with Path(genome_path).open("r", encoding="utf-8") as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    records: list[SequenceRecord] = []
    seen_ids: set[str] = set()
    total = retained = skipped = 0
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2].lower() != "gene":
                continue
            total += 1
            chromosome, _, _, start, end, _, strand, _, attributes = fields
            if strand not in {"+", "-"} or (strand == "-" and not include_negative_strand):
                skipped += 1
                continue
            try:
                gene_start = int(start)
                gene_end = int(end)
                if gene_start < 1 or gene_end < gene_start or chromosome not in genome:
                    raise ValueError
                gene_id = _gene_id(attributes, total)
                if gene_id in seen_ids:
                    raise ValueError
                if strand == "+":
                    window_start, window_end = upstream_window(gene_start, window_size)
                    sequence = str(genome[chromosome].seq[window_start - 1 : window_end]).upper()
                else:
                    window_start = gene_end + 1
                    window_end = gene_end + window_size
                    sequence = str(genome[chromosome].seq[window_start - 1 : window_end]).upper()
                    sequence = str(Seq(sequence).reverse_complement())
                if len(sequence) != window_size or set(sequence) - set("ACGT"):
                    raise ValueError
            except (ValueError, KeyError):
                skipped += 1
                continue
            records.append(SequenceRecord(gene_id, sequence))
            seen_ids.add(gene_id)
            retained += 1
    if not records:
        raise ValueError("No complete A/C/G/T promoter windows were extracted from the supplied files.")
    write_fasta(records, output_path)
    return {"gene_records": total, "retained_promoters": retained, "skipped_records": skipped}


def _gene_id(attributes: str, fallback: int) -> str:
    for field in attributes.split(";"):
        key, separator, value = field.partition("=")
        if separator and key in {"ID", "Name", "gene_id"} and value:
            return value
    return f"gene_{fallback}"
