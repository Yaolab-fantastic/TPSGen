from __future__ import annotations

import json
import math
from pathlib import Path

from tpsgen.io.csv import write_dict_rows
from tpsgen.io.fasta import write_fasta
from tpsgen.io.schema import SequenceRecord
from tpsgen.pipeline.predict_transvae import run_transvae_prediction
from tpsgen.training.pregan import generate_pregan_candidates


def _softplus(value: float) -> float:
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def run_pregan_workflow(
    records: list[SequenceRecord], checkpoint: str | Path, candidates: int,
    seed: int, output_dir: str | Path, target_tissue: str = "fruit",
    motif_backend: str = "native", min_margin: float = 0.0, min_tau: float = 0.7,
    adaptive_tau: bool = True, adaptive_tau_floor: float = 0.05,
) -> dict[str, object]:
    if target_tissue not in {"root", "stem", "leaf", "fruit"}:
        raise ValueError(f"Unsupported target tissue: {target_tissue}")
    if not 0 <= adaptive_tau_floor <= min_tau:
        raise ValueError("adaptive_tau_floor must be between zero and min_tau")
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
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") == "thesis_original_pregan":
        from tpsgen.legacy.pregan_original import generate_original_candidates
        rows = generate_original_candidates(templates, checkpoint, candidates, seed)
    else:
        rows = generate_pregan_candidates(templates, checkpoint, candidates, seed)
    expanded = [record for record in records for _ in range(candidates)]
    expanded_templates = [template for template in templates for _ in range(candidates)]
    for row, record, template in zip(rows, expanded, expanded_templates):
        row["sequence_id"] = record.sequence_id
        row["candidate_id"] = f"{record.sequence_id}__pregan_{row['candidate_rank']}"
        retained = [index for index, base in enumerate(template) if base != "M"]
        retained_matches = sum(str(row["sequence"])[index] == template[index] for index in retained)
        row["retained_position_count"] = len(retained)
        row["retained_position_matches"] = retained_matches
        row["retained_position_rate"] = retained_matches / len(retained) if retained else 1.0
        row["passes_retained_positions"] = retained_matches == len(retained)
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
        values = {tissue: _softplus(float(row[f"score_{tissue}"])) for tissue in ("root", "stem", "leaf", "fruit")}
        maximum = max(values.values())
        row["target_margin"] = values[target_tissue] - max(
            value for tissue, value in values.items() if tissue != target_tissue
        )
        row["preferred_tissue"] = max(values, key=values.get)
        row["tau"] = (
            sum(1.0 - value / maximum for value in values.values()) / 3.0
            if maximum > 0 else None
        )
    score_by_id = {row["sequence_id"]: row for row in score_rows}
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        score = score_by_id[row["candidate_id"]]
        row["target_tissue"] = target_tissue
        row["target_score"] = score["target_score"]
        row["target_margin"] = score["target_margin"]
        row["tau"] = score["tau"]
        row["preferred_tissue"] = score["preferred_tissue"]
        for tissue in ("root", "stem", "leaf", "fruit"):
            row[f"score_{tissue}"] = score[f"score_{tissue}"]
        row["eligible_for_ranking"] = bool(
            row["passes_qc"]
            and row["passes_retained_positions"]
            and score["preferred_tissue"] == target_tissue
            and float(score["target_margin"]) >= min_margin
            and score["tau"] is not None
            and float(score["tau"]) >= min_tau
        )
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
    fallback_rows: list[dict[str, object]] = []
    for record, template in zip(records, templates):
        group = grouped.get(record.sequence_id, [])
        if any(bool(row["eligible_for_ranking"]) for row in group):
            continue
        from tpsgen.legacy.transvae_tomato import TransVAETomatoAdapter
        seed_sequence = (
            record.sequence if set(record.sequence) <= {"A", "C", "G", "T"}
            else next(str(row["sequence"]) for row in group if row.get("passes_qc"))
        )
        mutable_positions = [index for index, base in enumerate(template) if base == "M"]
        adapter = TransVAETomatoAdapter(checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu")
        designs = adapter.constrained_search(
            seed_sequence, mutable_positions, target_tissue,
            population=max(128, candidates * 16), generations=30, seed=seed,
        )[:candidates]
        for candidate_rank, design in enumerate(designs, 1):
            values = {tissue: float(design["scores"][tissue]) for tissue in ("root", "stem", "leaf", "fruit")}
            maximum = max(values.values())
            margin = values[target_tissue] - max(v for k, v in values.items() if k != target_tissue)
            tau = float(design["tau"])
            retained = [index for index, base in enumerate(template) if base != "M"]
            retained_matches = sum(str(design["sequence"])[index] == template[index] for index in retained)
            passes = bool(design["passes_qc"] and retained_matches == len(retained)
                          and max(values, key=values.get) == target_tissue
                          and margin >= min_margin and tau is not None and tau >= min_tau)
            fallback_rows.append({"sequence_id": record.sequence_id, "candidate_rank": candidate_rank,
                "sequence": design["sequence"], "candidate_id": f"{record.sequence_id}__constrained_{candidate_rank}",
                "passes_qc": design["passes_qc"], "target_tissue": target_tissue,
                "retained_position_count": len(retained), "retained_position_matches": retained_matches,
                "retained_position_rate": retained_matches / len(retained) if retained else 1.0,
                "passes_retained_positions": retained_matches == len(retained),
                **{f"score_{tissue}": value for tissue, value in values.items()},
                "target_score": values[target_tissue], "target_margin": margin, "tau": tau,
                "preferred_tissue": max(values, key=values.get), "eligible_for_ranking": passes,
                "final_rank": "", "backend": "historical_transvae_constrained_search"})
        eligible = sorted((row for row in fallback_rows if row["sequence_id"] == record.sequence_id and row["eligible_for_ranking"]),
                          key=lambda row: float(row["target_score"]), reverse=True)
        for rank, row in enumerate(eligible, start=1):
            row["final_rank"] = rank
    effective_tau_by_input: dict[str, float] = {}
    specificity_mode_by_input: dict[str, str] = {}
    for record in records:
        combined = grouped.get(record.sequence_id, []) + [
            row for row in fallback_rows if row["sequence_id"] == record.sequence_id
        ]
        strict = [row for row in combined if row["eligible_for_ranking"]]
        effective_tau = min_tau
        mode = "paper_strict"
        if not strict and adaptive_tau:
            target_biased = [
                row for row in combined
                if row.get("passes_qc")
                and row.get("passes_retained_positions")
                and row.get("preferred_tissue") == target_tissue
                and float(row.get("target_margin", float("-inf"))) >= min_margin
                and row.get("tau") is not None
            ]
            if target_biased:
                ordered_tau = sorted(float(row["tau"]) for row in target_biased)
                q90_index = max(0, math.ceil(0.9 * len(ordered_tau)) - 1)
                effective_tau = max(adaptive_tau_floor, min(min_tau, ordered_tau[q90_index]))
                mode = "adaptive_target_biased"
                for row in target_biased:
                    row["eligible_for_ranking"] = float(row["tau"]) >= effective_tau
        effective_tau_by_input[record.sequence_id] = effective_tau
        specificity_mode_by_input[record.sequence_id] = mode
        eligible = sorted(
            (row for row in combined if row["eligible_for_ranking"]),
            key=lambda row: (float(row["target_margin"]), float(row["target_score"])),
            reverse=True,
        )
        for rank, row in enumerate(eligible, 1):
            row["final_rank"] = rank
            row["specificity_level"] = "strong" if mode == "paper_strict" else "target_biased"
            row["threshold_mode"] = mode
            row["effective_tau_threshold"] = effective_tau
        for row in combined:
            row.setdefault("specificity_level", "not_eligible")
            row.setdefault("threshold_mode", mode)
            row.setdefault("effective_tau_threshold", effective_tau)
    # Search diagnostics remain internal; publish fallback sequences only when
    # they pass either the paper-strict or explicitly labelled adaptive gate.
    rows.extend(row for row in fallback_rows if row["eligible_for_ranking"])
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    for row in rows:
        for field in all_fields:
            row.setdefault(field, "")
    write_dict_rows(rows, candidate_csv)
    write_fasta(
        [
            SequenceRecord(str(row["candidate_id"]), str(row.get("sequence", row.get("designed_sequence"))))
            for row in rows
        ],
        candidate_fasta,
    )
    score_path = root / "transvae_candidate_scores.csv"
    write_dict_rows(score_rows, score_path)
    manifest = {
        "workflow": "pregan_to_transvae_promoter_design",
        "input_domain": "masked_tomato_promoter_templates",
        "candidates_per_input": candidates,
        "seed": seed,
        "target_tissue": target_tissue,
        "specificity_thresholds": {
            "minimum_target_margin": min_margin,
            "minimum_tau": min_tau,
            "adaptive_tau_enabled": adaptive_tau,
            "adaptive_tau_floor": adaptive_tau_floor,
            "effective_tau_by_input": effective_tau_by_input,
            "threshold_mode_by_input": specificity_mode_by_input,
            "require_target_as_highest_tissue": True,
            "require_all_retained_positions": True,
        },
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
        "num_latent_fallback_candidates": len(fallback_rows),
        "num_specificity_pass": sum(bool(row["eligible_for_ranking"]) for row in rows),
        "external_validation_required": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
