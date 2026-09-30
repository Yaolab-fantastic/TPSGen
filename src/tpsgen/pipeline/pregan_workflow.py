from __future__ import annotations

import json
from pathlib import Path

from tpsgen.io.csv import write_dict_rows
from tpsgen.io.fasta import write_fasta
from tpsgen.io.schema import SequenceRecord
from tpsgen.pipeline.predict_transvae import run_transvae_prediction
from tpsgen.training.pregan import generate_pregan_candidates


def run_pregan_workflow(
    records: list[SequenceRecord], checkpoint: str | Path, candidates: int,
    seed: int, output_dir: str | Path, target_tissue: str = "fruit",
    motif_backend: str = "native",
) -> dict[str, object]:
    if target_tissue not in {"root", "stem", "leaf", "fruit"}:
        raise ValueError(f"Unsupported target tissue: {target_tissue}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    motif_file = None
    templates = [record.sequence for record in records]
    retained_positions: dict[str, list[int]] = {}
    if motif_backend not in {"native", "dnabert"}:
        raise ValueError(f"Unsupported motif backend: {motif_backend}")
    if motif_backend == "dnabert":
        from tpsgen.models.dnabert_inference import DNABERTTomatoAdapter, evidence_to_masked_template
        from tpsgen.resources import find_model
        adapter = DNABERTTomatoAdapter(find_model("dnabert/pytorch_model.bin").parent)
        # DNABERT accepts only unambiguous DNA. Existing masked templates are
        # therefore completed with deterministic A placeholders for evidence
        # extraction; the original M positions remain editable in preGAN.
        evidence_records = [
            SequenceRecord(record.sequence_id, record.sequence.replace("M", "A"))
            for record in records
        ]
        evidence = adapter.predict(evidence_records)
        evidence_by_id = {str(row["sequence_id"]): row for row in evidence}
        converted = []
        for record in records:
            template, positions = evidence_to_masked_template(record.sequence, evidence_by_id[record.sequence_id])
            converted.append(template)
            retained_positions[record.sequence_id] = positions
        templates = converted
    template_rows = [
        {
            "sequence_id": record.sequence_id,
            "source_sequence": record.sequence,
            "masked_template": template,
            "retained_positions": ";".join(str(position) for position in retained_positions.get(record.sequence_id, [])),
            "template_backend": "dnabert_attention_top_fraction" if motif_backend == "dnabert" else "user_supplied",
        }
        for record, template in zip(records, templates)
    ]
    template_path = root / "pregan_masked_templates.csv"
    write_dict_rows(template_rows, template_path)
    rows = generate_pregan_candidates(templates, checkpoint, candidates, seed)
    expanded = [record for record in records for _ in range(candidates)]
    for row, record in zip(rows, expanded):
        row["sequence_id"] = record.sequence_id
        row["candidate_id"] = f"{record.sequence_id}__pregan_{row['candidate_rank']}"
    candidate_csv = root / "pregan_candidates.csv"
    candidate_fasta = root / "pregan_candidates.fasta"
    write_dict_rows(rows, candidate_csv)
    write_fasta(
        [SequenceRecord(str(row["candidate_id"]), str(row["sequence"])) for row in rows],
        candidate_fasta,
    )
    if motif_backend == "dnabert":
        motif_file = root / "dnabert_attention_evidence.csv"
        write_dict_rows(evidence, motif_file)
    scored = run_transvae_prediction(
        [SequenceRecord(str(row["candidate_id"]), str(row["sequence"])) for row in rows]
    )
    score_rows = [item.to_dict() for item in scored]
    for row in score_rows:
        row["backend"] = "tomato_transvae_mlp_checkpoint_scoring"
        row["score_type"] = "transvae_model_score"
        row["target_tissue"] = target_tissue
        row["target_score"] = row[f"score_{target_tissue}"]
    score_by_id = {row["sequence_id"]: row for row in score_rows}
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        score = score_by_id[row["candidate_id"]]
        row["target_tissue"] = target_tissue
        row["target_score"] = score["target_score"]
        row["eligible_for_ranking"] = bool(row["passes_qc"])
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    for group in grouped.values():
        eligible = sorted(
            (row for row in group if row["eligible_for_ranking"]),
            key=lambda row: float(row["target_score"]), reverse=True
        )
        for rank, row in enumerate(eligible, start=1):
            row["final_rank"] = rank
        for row in group:
            if not row["eligible_for_ranking"]:
                row["final_rank"] = ""
    write_dict_rows(rows, candidate_csv)
    score_path = root / "transvae_candidate_scores.csv"
    write_dict_rows(score_rows, score_path)
    manifest = {
        "workflow": "pregan_to_transvae_promoter_design",
        "input_domain": "masked_tomato_promoter_templates",
        "candidates_per_input": candidates,
        "seed": seed,
        "target_tissue": target_tissue,
        "motif_backend": motif_backend,
        "dnabert_template_policy": {
            "top_fraction": 0.2,
            "evidence_placeholder_for_input_M": "A",
            "retained_positions_by_input": retained_positions,
            "masked_symbol": "M",
        } if motif_backend == "dnabert" else None,
        "backends": {
            "design": "pregan_conditional_generator",
            "scoring": "tomato_transvae_mlp_checkpoint_scoring",
            "motif": "tomato_dnabert_6mer_attention_inference" if motif_backend == "dnabert" else "package_native_exact_motif_annotation",
        },
        "files": {
            "candidates_csv": candidate_csv.name,
            "candidates_fasta": candidate_fasta.name,
            "candidate_scores": score_path.name,
            "motif_evidence": motif_file.name if motif_file else None,
            "masked_templates": template_path.name,
        },
        "information_flow": {
            "dnabert_evidence_to_masked_template": motif_backend == "dnabert",
            "masked_template_to_pregan": True,
            "pregan_to_transvae_scoring": True,
        },
        "num_input_templates": len(records),
        "num_candidates": len(rows),
        "external_validation_required": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
