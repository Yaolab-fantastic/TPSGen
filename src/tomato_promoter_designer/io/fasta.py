from __future__ import annotations

from pathlib import Path

from tomato_promoter_designer.io.schema import SequenceRecord, validate_records


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    path = Path(path)
    records: list[SequenceRecord] = []
    current_id: str | None = None
    sequence_parts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(SequenceRecord(current_id, "".join(sequence_parts)))
                current_id = line[1:].strip() or f"sequence_{len(records) + 1}"
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if current_id is not None:
        records.append(SequenceRecord(current_id, "".join(sequence_parts)))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return validate_records(records)


def write_fasta(records: list[SequenceRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.sequence_id}\n")
            handle.write(f"{record.sequence}\n")

