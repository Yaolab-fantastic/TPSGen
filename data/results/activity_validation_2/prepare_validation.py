"""Prepare the held-out validation records used by the original TransVAE run."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("/data/zhoujie/Paper/MpraVAE/data/vaedata/val.csv")
OUT = Path(__file__).resolve().parent

with SOURCE.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

records = []
observed = []
for index, row in enumerate(rows, start=1):
    sequence = row["realB"].strip().upper()
    sequence_id = f"validation_promoter_{index:04d}"
    records.append((sequence_id, sequence))
    observed.append(
        {
            "sequence_id": sequence_id,
            "sequence": sequence,
            "expr_tissue_1": row["expr_tissue_1"],
            "expr_tissue_2": row["expr_tissue_2"],
            "expr_tissue_3": row["expr_tissue_3"],
            "expr_tissue_4": row["expr_tissue_4"],
        }
    )

with (OUT / "validation_promoters.fasta").open("w", encoding="utf-8") as handle:
    for sequence_id, sequence in records:
        handle.write(f">{sequence_id}\n{sequence}\n")

with (OUT / "observed_expression.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(observed[0]))
    writer.writeheader()
    writer.writerows(observed)

(OUT / "source_manifest.json").write_text(
    json.dumps(
        {
            "source_file": str(SOURCE),
            "source_role": "held-out validation records used by the original TransVAE evaluation",
            "num_records": len(records),
            "sequence_length_bp": 165,
            "tissue_columns": ["expr_tissue_1", "expr_tissue_2", "expr_tissue_3", "expr_tissue_4"],
            "reported_validation_pcc": {"expr_tissue_1": 0.7732, "expr_tissue_2": 0.8159, "expr_tissue_3": 0.7954, "expr_tissue_4": 0.7876},
            "independent_test_set": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
