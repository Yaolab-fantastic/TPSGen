from __future__ import annotations

from pathlib import Path

from tpsgen.visualization.legacy_svg import export_legacy_figure_bundle


def run_legacy_figure_export(
    output_dir: str | Path,
    transvae_loss_history: str | Path | None = None,
    transvae_designed_promoters: str | Path | None = None,
    transvae_prediction_results: str | Path | None = None,
    pregan_expression_training_log: str | Path | None = None,
    transvae_mutated_file: str | Path | None = None,
    transvae_random_promoters: str | Path | None = None,
    transvae_training_set: str | Path | None = None,
    dnabert_motif_summary: str | Path | None = None,
    dnabert_tfbs_dir: str | Path | None = None,
    pregan_expression_scatter_png: str | Path | None = None,
    transvae_blast_dir: str | Path | None = None,
    transvae_diversity_dir: str | Path | None = None,
) -> dict[str, object]:
    return export_legacy_figure_bundle(
        output_dir=output_dir,
        mpravae_loss_history=transvae_loss_history,
        mpravae_designed_promoters=transvae_designed_promoters,
        mpravae_prediction_results=transvae_prediction_results,
        deepseed_training_log=pregan_expression_training_log,
        mpravae_mutated_file=transvae_mutated_file,
        mpravae_random_promoters=transvae_random_promoters,
        mpravae_training_set=transvae_training_set,
        dnabert_motif_summary=dnabert_motif_summary,
        dnabert_tfbs_dir=dnabert_tfbs_dir,
        deepseed_scatter_png=pregan_expression_scatter_png,
        mpravae_blast_dir=transvae_blast_dir,
        mpravae_diversity_dir=transvae_diversity_dir,
    )
