from __future__ import annotations

import json
from pathlib import Path

from tpsgen.io.csv import write_dict_rows
from tpsgen.io.fasta import write_fasta
from tpsgen.io.schema import SequenceRecord
from tpsgen.pipeline.annotate import run_annotation
from tpsgen.pipeline.design import run_design
from tpsgen.pipeline.predict import run_prediction
from tpsgen.pipeline.report import build_report
from tpsgen.visualization.svg import (
    render_design_summary,
    render_prediction_heatmap,
)


def run_integrated_workflow(
    records: list[SequenceRecord],
    target_tissue: str,
    candidates: int,
    seed: int,
    output_dir: str | Path,
    scoring_backend: str = "native",
    checkpoint_path: str | Path | None = None,
    motif_backend: str = "native",
) -> dict[str, object]:
    """Run the reproducible, one-command package workflow.

    The workflow keeps intermediate representations visible so that users can
    inspect how motif evidence, candidate design and scoring are connected.
    """
    root = Path(output_dir)
    motif_dir = root / "motif"
    design_dir = root / "design"
    scoring_dir = root / "scoring"
    report_dir = root / "reports"
    figure_dir = root / "figures"
    input_dir = root / "input"
    for directory in (motif_dir, design_dir, scoring_dir, report_dir, figure_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)

    validated_input = input_dir / "validated_promoters.fasta"
    write_fasta(records, validated_input)

    motifs = run_annotation(records)
    motif_rows = [hit.to_dict() for hit in motifs]
    if not motif_rows:
        motif_rows = [
            {"sequence_id": record.sequence_id, "motif": "none", "start": -1, "end": -1, "score": 0.0}
            for record in records
        ]
    for row in motif_rows:
        row["backend"] = "package_native_exact_motif_annotation"
        row["evidence_type"] = "exact_configured_motif_hit"
    motif_path = motif_dir / "motif_annotations.csv"
    write_dict_rows(motif_rows, motif_path)
    dnabert_path = None
    if motif_backend == "dnabert":
        from tpsgen.models.dnabert_inference import DNABERTTomatoAdapter

        dnabert_path = motif_dir / "dnabert_attention_evidence.csv"
        from tpsgen.resources import find_model
        dnabert_rows = DNABERTTomatoAdapter(find_model("dnabert/pytorch_model.bin").parent).predict(records)
        write_dict_rows(dnabert_rows, dnabert_path)
    elif motif_backend != "native":
        raise ValueError(f"Unsupported motif backend: {motif_backend}")

    designs = run_design(records, target_tissue=target_tissue, candidates=candidates, seed=seed)
    design_rows = [item.to_dict() for item in designs]
    for row in design_rows:
        row["backend"] = "package_native_motif_preserving_design"
        row["score_type"] = "tissue_associated_heuristic_score"
    design_path = design_dir / "candidate_metadata.csv"
    write_dict_rows(design_rows, design_path)
    candidate_records = [
        SequenceRecord(
            f"{item.sequence_id}__candidate_{item.candidate_rank}",
            item.designed_sequence,
        )
        for item in designs
    ]
    candidate_fasta = design_dir / "designed_candidates.fasta"
    write_fasta(candidate_records, candidate_fasta)

    if scoring_backend == "native":
        score_fn = run_prediction
        score_backend_name = "package_native_deterministic_tissue_associated_scoring"
    elif scoring_backend == "transvae":
        from tpsgen.pipeline.predict_transvae import run_transvae_prediction

        score_fn = lambda items: run_transvae_prediction(items, checkpoint_path=checkpoint_path)
        score_backend_name = "tomato_transvae_mlp_checkpoint_scoring"
    else:
        raise ValueError(f"Unsupported scoring backend: {scoring_backend}")

    original_scores = score_fn(records)
    candidate_scores = score_fn(candidate_records)
    candidate_score_by_id = {item.sequence_id: item for item in candidate_scores}
    def add_bias_metrics(row: dict[str, object]) -> dict[str, object]:
        values = {tissue: float(row[f"score_{tissue}"]) for tissue in ("root", "stem", "leaf", "fruit")}
        target_value = values[target_tissue]
        runner_up = max(value for tissue, value in values.items() if tissue != target_tissue)
        maximum = max(values.values())
        minimum = min(values.values())
        # tau is a descriptive concentration index; retain the raw scores too.
        row["target_tissue"] = target_tissue
        row["target_bias_margin"] = round(target_value - runner_up, 4)
        row["tau"] = round(
            sum(1.0 - value / maximum for value in values.values()) / 3.0,
            4,
        ) if maximum > 0 else None
        row["preferred_tissue"] = max(values, key=values.get)
        return row

    original_score_rows = [add_bias_metrics(item.to_dict()) for item in original_scores]
    candidate_score_rows = [add_bias_metrics(item.to_dict()) for item in candidate_scores]
    for row in original_score_rows + candidate_score_rows:
        row["score_type"] = "tissue_associated_heuristic_score" if scoring_backend == "native" else "transvae_model_score"
        row["scoring_backend"] = score_backend_name
    original_score_path = scoring_dir / "original_scores.csv"
    candidate_score_path = scoring_dir / "candidate_scores.csv"
    write_dict_rows(original_score_rows, original_score_path)
    write_dict_rows(candidate_score_rows, candidate_score_path)

    combined_scores = []
    for item in original_scores:
        row = add_bias_metrics(item.to_dict())
        row["sequence_role"] = "original"
        row["score_type"] = "tissue_associated_heuristic_score" if scoring_backend == "native" else "transvae_model_score"
        row["scoring_backend"] = score_backend_name
        combined_scores.append(row)
    for item in candidate_scores:
        row = add_bias_metrics(item.to_dict())
        row["sequence_role"] = "designed_candidate"
        row["score_type"] = "tissue_associated_heuristic_score" if scoring_backend == "native" else "transvae_model_score"
        row["scoring_backend"] = score_backend_name
        combined_scores.append(row)
    combined_score_path = scoring_dir / "all_sequence_scores.csv"
    write_dict_rows(combined_scores, combined_score_path)

    render_prediction_heatmap(original_score_rows, figure_dir / "original_score_heatmap.svg")
    render_prediction_heatmap(candidate_score_rows, figure_dir / "candidate_score_heatmap.svg")
    render_design_summary(design_rows, figure_dir / "candidate_design_summary.svg")

    report_path = report_dir / "workflow_report.json"
    report = build_report(design_path, report_path)

    original_by_id = {item.sequence_id: item for item in original_scores}
    best_by_id = {}
    for item in designs:
        candidate_id = f"{item.sequence_id}__candidate_{item.candidate_rank}"
        scored_candidate = candidate_score_by_id[candidate_id]
        current = best_by_id.get(item.sequence_id)
        if current is None:
            best_by_id[item.sequence_id] = (item, scored_candidate)
            continue
        _, current_score = current
        if getattr(scored_candidate, f"score_{target_tissue}") > getattr(current_score, f"score_{target_tissue}"):
            best_by_id[item.sequence_id] = (item, scored_candidate)
    summary_rows = []
    for record in records:
        original = original_by_id[record.sequence_id]
        best, best_score = best_by_id[record.sequence_id]
        summary_rows.append(
            {
                "sequence_id": record.sequence_id,
                "target_tissue": target_tissue,
                "original_preferred_tissue": original.preferred_tissue,
                "original_target_score": getattr(original, f"score_{target_tissue}"),
                "best_candidate_rank": best.candidate_rank,
                "best_candidate_target_score": getattr(best_score, f"score_{target_tissue}"),
                "best_candidate_target_bias_margin": round(
                    getattr(best_score, f"score_{target_tissue}")
                    - max(
                        getattr(best_score, f"score_{tissue}")
                        for tissue in ("root", "stem", "leaf", "fruit")
                        if tissue != target_tissue
                    ),
                    4,
                ),
                "best_candidate_num_mutations": best.num_mutations,
                "best_candidate_passes_qc": best.passes_qc,
            }
        )
    summary_path = report_dir / "workflow_summary.csv"
    write_dict_rows(summary_rows, summary_path)
    manifest = {
        "workflow": "integrated_promoter_design",
        "workflow_version": "0.2.0",
        "input_domain": "tomato_promoter_sequences",
        "sequence_requirements": {
            "accepted_input": "FASTA with one or more tomato promoter sequences",
            "canonical_model_length_bp": 165,
            "default_route_allows_noncanonical_length": True,
            "alphabet": "A/C/G/T/N/M before validation; A/C/G/T for generated candidates",
        },
        "target_tissue": target_tissue,
        "motif_backend": motif_backend,
        "candidates_per_input": candidates,
        "seed": seed,
        "num_input_sequences": len(records),
        "num_candidate_sequences": len(candidate_records),
        "backends": {
            "motif_evidence": {
                "name": "tomato_dnabert_6mer_attention_inference" if motif_backend == "dnabert" else "package_native_exact_motif_annotation",
                "input": "validated tomato promoter FASTA records",
                "output": "motif/motif_annotations.csv",
                "checkpoint_used": motif_backend == "dnabert",
                "optional_dnabert_evidence": str(dnabert_path.relative_to(root)) if dnabert_path else None,
            },
            "candidate_generation": {
                "name": "package_native_motif_preserving_design",
                "input": "validated promoter records and detected motif coordinates",
                "output": "design/designed_candidates.fasta and design/candidate_metadata.csv",
                "checkpoint_used": False,
                "generation_rank": "native target-tissue score before optional model-backed rescoring",
            },
            "tissue_scoring": {
                "name": score_backend_name,
                "input": "original and generated tomato promoter sequences",
                "output": "scoring/*.csv",
                "checkpoint_used": scoring_backend == "transvae",
                "score_definition": "ranking score, not calibrated expression" if scoring_backend == "native" else "TransVAE four-tissue model score; scale is checkpoint-specific",
                "final_candidate_selection": "best target-tissue score within each input promoter",
            },
        },
        "model_routes": {
            "dnabert": {
                "status": "bundled_checkpoint_inference_available; retained_attention_postprocessing_requires_provenance_review",
                "not_used_by_default_run": True,
                "resource": "data/raw/dnabert/atten.npy",
                "resource_rows": 10222,
                "matched_source_in_upstream_tree": "DNABERT/examples/sample_data/vision/dev.tsv",
                "tomato_specificity_verified": False,
            },
            "pregan": {
                "status": "explicit_generator_route_available; historical_quantitative_and_biological_validation_not_established",
                "not_used_by_default_run": True,
                "resource": "models/pregan/generator_checkpoint.pt",
                "training_records": 5100,
                "training_steps": 1000,
            },
            "transvae": {
                "status": "available_as_optional_scoring_backend",
                "not_used_by_default_run": scoring_backend != "transvae",
                "resource": "models/transvae/best_val_corr_model.pth",
            },
        },
        "files": {
            "validated_input": str(validated_input.relative_to(root)),
            "motif_annotations": str(motif_path.relative_to(root)),
            "candidate_metadata": str(design_path.relative_to(root)),
            "designed_candidates_fasta": str(candidate_fasta.relative_to(root)),
            "original_scores": str(original_score_path.relative_to(root)),
            "candidate_scores": str(candidate_score_path.relative_to(root)),
            "all_sequence_scores": str(combined_score_path.relative_to(root)),
            "workflow_report": str(report_path.relative_to(root)),
            "workflow_summary": str(summary_path.relative_to(root)),
            "figures": [
                "figures/original_score_heatmap.svg",
                "figures/candidate_score_heatmap.svg",
                "figures/candidate_design_summary.svg",
            ],
        },
        "report_summary": {
            "average_num_mutations": report["average_num_mutations"],
            "num_qc_passed": report["num_qc_passed"],
            "num_rows": report["num_rows"],
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
