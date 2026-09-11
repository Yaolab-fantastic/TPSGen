from __future__ import annotations

from pathlib import Path

from tomato_promoter_designer.legacy.dnabert_motif import run_dnabert_legacy_annotation


def run_legacy_dnabert_motif_annotation(
    dev_tsv_path: str | Path,
    attention_scores_path: str | Path,
    output_dir: str | Path,
    window_size: int = 24,
    min_len: int = 5,
    p_value_cutoff: float = 0.005,
    min_n_motif: int = 3,
) -> dict[str, object]:
    return run_dnabert_legacy_annotation(
        dev_tsv_path=dev_tsv_path,
        attention_scores_path=attention_scores_path,
        output_dir=output_dir,
        window_size=window_size,
        min_len=min_len,
        p_value_cutoff=p_value_cutoff,
        min_n_motif=min_n_motif,
    )
