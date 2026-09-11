from __future__ import annotations

import html
import json
from pathlib import Path

from tomato_promoter_designer.io.csv import read_dict_rows


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _linear_color(value: float, minimum: float, maximum: float) -> str:
    if maximum <= minimum:
        ratio = 0.5
    else:
        ratio = (value - minimum) / (maximum - minimum)
    ratio = _clamp(ratio, 0.0, 1.0)
    red = int(245 - 110 * ratio)
    green = int(247 - 90 * ratio)
    blue = int(244 - 180 * ratio)
    return f"rgb({red},{green},{blue})"


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _write_svg(lines: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _display_design_status(status: str) -> str:
    labels = {
        "mpravae_decoded": "TransVAE decoded",
        "mpravae_repaired_by_reversion": "QC-repaired candidate",
        "mpravae_qc_fallback": "Original sequence retained",
        "baseline_motif_preserving": "Baseline motif-preserving",
    }
    return labels.get(status, status.replace("_", " "))


def _score_key(row: dict[str, str], tissue: str) -> str:
    score_key = f"score_{tissue}"
    if score_key in row:
        return score_key
    expr_key = f"expr_{tissue}"
    if expr_key in row:
        return expr_key
    raise KeyError(f"Missing score column for tissue: {tissue}")


def _has_tissue_scores(row: dict[str, str]) -> bool:
    return all(f"score_{tissue}" in row or f"expr_{tissue}" in row for tissue in ("root", "stem", "leaf", "fruit"))


def render_prediction_heatmap(rows: list[dict[str, str]], output_path: str | Path) -> None:
    tissues = ("root", "stem", "leaf", "fruit")
    values = [
        float(row[_score_key(row, tissue)])
        for row in rows
        for tissue in tissues
        if row.get(_score_key(row, tissue)) not in (None, "")
    ]
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 1.0
    width = 880
    height = 160 + len(rows) * 54
    left = 210
    top = 92
    cell_w = 130
    cell_h = 38

    lines = _svg_header(width, height)
    lines.append('<text x="40" y="42" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">Tissue-associated Score Heatmap</text>')
    lines.append('<text x="40" y="66" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Generated from tissue-score results for package reports and manuscript figures.</text>')

    for column_index, tissue in enumerate(tissues):
        x = left + column_index * cell_w
        lines.append(
            f'<text x="{x + cell_w / 2:.1f}" y="{top - 18}" text-anchor="middle" '
            f'font-size="13" font-family="Arial, sans-serif" fill="#33454d">{_escape(tissue)}</text>'
        )

    for row_index, row in enumerate(rows):
        y = top + row_index * 54
        lines.append(
            f'<text x="42" y="{y + 24}" font-size="12" font-family="Arial, sans-serif" fill="#22333b">{_escape(row["sequence_id"])}</text>'
        )
        preferred = row.get("preferred_tissue", "")
        lines.append(
            f'<text x="42" y="{y + 40}" font-size="11" font-family="Arial, sans-serif" fill="#7a5c2e">preferred: {_escape(preferred)}</text>'
        )
        for column_index, tissue in enumerate(tissues):
            value = float(row[_score_key(row, tissue)])
            x = left + column_index * cell_w
            fill = _linear_color(value, minimum, maximum)
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 10}" height="{cell_h}" rx="7" fill="{fill}" stroke="#d7d2c8"/>'
            )
            lines.append(
                f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{y + 24}" text-anchor="middle" '
                f'font-size="13" font-family="Arial, sans-serif" fill="#182126">{value:.3f}</text>'
            )

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def render_design_summary(rows: list[dict[str, str]], output_path: str | Path) -> None:
    width = 1100
    height = 190 + len(rows) * 56
    left = 330
    top = 100
    plot_width = 620
    max_gap = max(
        abs(float(row[_score_key(row, "fruit")]) - max(float(row[_score_key(row, "root")]), float(row[_score_key(row, "stem")]), float(row[_score_key(row, "leaf")])))
        for row in rows
    ) if rows else 1.0
    max_gap = max(max_gap, 0.1)
    max_mutations = max(int(row.get("num_mutations") or 0) for row in rows) if rows else 1

    lines = _svg_header(width, height)
    lines.append('<text x="40" y="42" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">Designed Candidate Comparison</text>')
    lines.append('<text x="40" y="66" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Bars show target-tissue margin over the strongest competing tissue. Circles encode mutation burden.</text>')

    axis_y = top - 16
    lines.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_width}" y2="{axis_y}" stroke="#c5bfb2"/>')
    lines.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - 32}" stroke="#c5bfb2"/>')
    lines.append(f'<line x1="{left + plot_width / 2:.1f}" y1="{top - 8}" x2="{left + plot_width / 2:.1f}" y2="{height - 32}" stroke="#e2ded5" stroke-dasharray="4 4"/>')
    lines.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{axis_y - 10}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">target advantage</text>'
    )

    for row_index, row in enumerate(rows):
        y = top + row_index * 56
        target_tissue = row["target_tissue"]
        target_value = float(row[_score_key(row, target_tissue)])
        competitor = max(
            float(row[_score_key(row, tissue)])
            for tissue in ("root", "stem", "leaf", "fruit")
            if tissue != target_tissue
        )
        gap = target_value - competitor
        bar_width = abs(gap) / max_gap * (plot_width / 2 - 16)
        bar_x = left + plot_width / 2 if gap >= 0 else left + plot_width / 2 - bar_width
        fill = "#d97a34" if gap >= 0 else "#7b8f98"
        mutations = int(row.get("num_mutations") or 0)
        radius = 4 + (mutations / max_mutations) * 12 if max_mutations else 4
        status = row.get("design_status", "")
        display_status = _display_design_status(status)

        lines.append(
            f'<text x="42" y="{y + 17}" font-size="12" font-family="Arial, sans-serif" fill="#22333b">{_escape(row["sequence_id"])} #{_escape(row["candidate_rank"])}</text>'
        )
        lines.append(
            f'<text x="42" y="{y + 34}" font-size="11" font-family="Arial, sans-serif" fill="#7a5c2e">{_escape(display_status)}</text>'
        )
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_width:.1f}" height="24" rx="6" fill="{fill}" opacity="0.9"/>'
        )
        lines.append(
            f'<text x="{left + plot_width + 20}" y="{y + 16}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">gap {gap:.3f}</text>'
        )
        lines.append(
            f'<text x="{left + plot_width + 20}" y="{y + 32}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">mut {mutations}</text>'
        )
        lines.append(
            f'<circle cx="{left + plot_width + 120}" cy="{y + 12}" r="{radius:.1f}" fill="#c94f4f" opacity="0.75"/>'
        )

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def render_motif_enrichment(rows: list[dict[str, str]], output_path: str | Path, top_n: int = 15) -> None:
    filtered = sorted(
        rows,
        key=lambda row: float(row.get("enrichment_ratio") or 0.0),
        reverse=True,
    )[:top_n]
    width = 980
    height = 150 + len(filtered) * 34
    left = 220
    top = 88
    max_ratio = max(float(row.get("enrichment_ratio") or 0.0) for row in filtered) if filtered else 1.0
    max_ratio = max(max_ratio, 0.01)

    lines = _svg_header(width, height)
    lines.append('<text x="40" y="42" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">DNABERT Motif Enrichment Summary</text>')
    lines.append('<text x="40" y="66" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Top motifs ranked by enrichment ratio from attention-based motif post-processing.</text>')

    for row_index, row in enumerate(filtered):
        y = top + row_index * 34
        ratio = float(row.get("enrichment_ratio") or 0.0)
        instances = int(row.get("num_instances") or 0)
        p_value = float(row.get("adjusted_p_value") or 0.0)
        bar_width = ratio / max_ratio * 500
        fill = _linear_color(ratio, 0.0, max_ratio)
        lines.append(
            f'<text x="40" y="{y + 16}" font-size="12" font-family="Arial, sans-serif" fill="#22333b">{_escape(row["motif"])}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="20" rx="6" fill="{fill}" stroke="#d7d2c8"/>'
        )
        lines.append(
            f'<text x="{left + bar_width + 12:.1f}" y="{y + 14}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">ratio {ratio:.4f} | n {instances} | q {p_value:.3g}</text>'
        )

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def export_svg_figures(
    input_csv: str | Path,
    output_dir: str | Path,
    top_n: int = 15,
) -> dict[str, object]:
    rows = read_dict_rows(input_csv)
    if not rows:
        raise ValueError("Input table is empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_row = rows[0]
    generated: list[str] = []

    if _has_tissue_scores(first_row) and "preferred_tissue" in first_row:
        output_path = output_dir / "prediction_heatmap.svg"
        render_prediction_heatmap(rows, output_path)
        figure_type = "prediction"
        generated.append(output_path.name)
    elif {"target_tissue", "designed_sequence"}.issubset(first_row) and _has_tissue_scores(first_row):
        output_path = output_dir / "design_summary.svg"
        render_design_summary(rows, output_path)
        figure_type = "design"
        generated.append(output_path.name)
    elif {"motif", "num_instances", "enrichment_ratio", "adjusted_p_value"}.issubset(first_row):
        output_path = output_dir / "motif_enrichment.svg"
        render_motif_enrichment(rows, output_path, top_n=top_n)
        figure_type = "motif"
        generated.append(output_path.name)
    else:
        raise ValueError("Unsupported input schema for figure export.")

    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "figure_type": figure_type,
        "files": generated,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
