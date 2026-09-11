from __future__ import annotations

from pathlib import Path

from tpsgen.visualization.svg import export_svg_figures


def run_figure_export(
    input_csv: str | Path,
    output_dir: str | Path,
    top_n: int = 15,
) -> dict[str, object]:
    return export_svg_figures(input_csv=input_csv, output_dir=output_dir, top_n=top_n)
