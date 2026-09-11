from __future__ import annotations

import json
from pathlib import Path

from tpsgen.config import AppConfig
from tpsgen.evaluation.edit_distance import levenshtein
from tpsgen.io.csv import read_dict_rows


def _display_design_status(status: str) -> str:
    labels = {
        "mpravae_decoded": "TransVAE decoded",
        "mpravae_repaired_by_reversion": "QC-repaired candidate",
        "mpravae_qc_fallback": "Original sequence retained",
        "baseline_motif_preserving": "Baseline motif-preserving",
    }
    return labels.get(status, status.replace("_", " "))


def _coerce_bool(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _score_column(row: dict[str, str], tissue: str) -> str:
    score_key = f"score_{tissue}"
    if score_key in row:
        return score_key
    expr_key = f"expr_{tissue}"
    if expr_key in row:
        return expr_key
    raise KeyError(f"Missing score column for tissue: {tissue}")


def _best_candidate_summary(row: dict[str, str]) -> dict[str, object]:
    target_tissue = row["target_tissue"]
    target_score = float(row[_score_column(row, target_tissue)])
    competing_scores = [
        float(row[_score_column(row, tissue)])
        for tissue in ("root", "stem", "leaf", "fruit")
        if tissue != target_tissue
    ]
    runner_up_score = max(competing_scores) if competing_scores else target_score
    return {
        "sequence_id": row["sequence_id"],
        "target_tissue": target_tissue,
        "best_candidate_rank": int(row["candidate_rank"]),
        "design_status": row.get("design_status", ""),
        "design_status_label": _display_design_status(row.get("design_status", "")),
        "passes_qc": _coerce_bool(row.get("passes_qc")),
        "num_mutations": int(row.get("num_mutations") or 0),
        "target_score": round(target_score, 4),
        "runner_up_score": round(runner_up_score, 4),
        "target_margin": round(target_score - runner_up_score, 4),
    }


def _target_margin(row: dict[str, str]) -> float:
    target_tissue = row["target_tissue"]
    target_score = float(row[_score_column(row, target_tissue)])
    competing_scores = [
        float(row[_score_column(row, tissue)])
        for tissue in ("root", "stem", "leaf", "fruit")
        if tissue != target_tissue
    ]
    return target_score - max(competing_scores) if competing_scores else target_score


def build_report(input_csv: str | Path, output_json: str | Path) -> dict[str, object]:
    rows = read_dict_rows(input_csv)
    if not rows:
        raise ValueError("Input design table is empty.")

    config = AppConfig()
    unique_sequences = {row["sequence_id"] for row in rows}
    target_tissues = sorted({row["target_tissue"] for row in rows})
    edit_distances = [
        levenshtein(row["original_sequence"], row["designed_sequence"])
        for row in rows
        if "original_sequence" in row and "designed_sequence" in row
    ]
    mutation_counts = [
        int(row["num_mutations"])
        for row in rows
        if row.get("num_mutations") not in (None, "")
    ]
    qc_values = [_coerce_bool(row.get("passes_qc")) for row in rows]
    qc_observed = [value for value in qc_values if value is not None]
    design_status_counts: dict[str, int] = {}
    design_status_display_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("design_status", "")
        if status:
            design_status_counts[status] = design_status_counts.get(status, 0) + 1
            display_status = _display_design_status(status)
            design_status_display_counts[display_status] = design_status_display_counts.get(display_status, 0) + 1

    best_rows_by_sequence: dict[str, dict[str, str]] = {}
    for row in rows:
        sequence_id = row["sequence_id"]
        existing = best_rows_by_sequence.get(sequence_id)
        if existing is None:
            best_rows_by_sequence[sequence_id] = row
            continue
        current_margin = _target_margin(row)
        best_margin = _target_margin(existing)
        if current_margin > best_margin:
            best_rows_by_sequence[sequence_id] = row
        elif current_margin == best_margin and int(row["candidate_rank"]) < int(existing["candidate_rank"]):
            best_rows_by_sequence[sequence_id] = row

    report = {
        "num_rows": len(rows),
        "num_input_sequences": len(unique_sequences),
        "num_unique_designed_sequences": len({row["designed_sequence"] for row in rows if row.get("designed_sequence")}),
        "target_tissues": target_tissues,
        "canonical_length": config.canonical_length,
        "average_edit_distance": round(sum(edit_distances) / len(edit_distances), 4) if edit_distances else 0.0,
        "average_num_mutations": round(sum(mutation_counts) / len(mutation_counts), 4) if mutation_counts else 0.0,
        "min_num_mutations": min(mutation_counts) if mutation_counts else 0,
        "max_num_mutations": max(mutation_counts) if mutation_counts else 0,
        "num_qc_passed": sum(1 for value in qc_observed if value),
        "qc_pass_rate": round(sum(1 for value in qc_observed if value) / len(qc_observed), 4) if qc_observed else None,
        "design_status_counts": design_status_counts,
        "design_status_display_counts": design_status_display_counts,
        "best_candidate_by_sequence": [
            _best_candidate_summary(best_rows_by_sequence[sequence_id])
            for sequence_id in sorted(best_rows_by_sequence)
        ],
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
