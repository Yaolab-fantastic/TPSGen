from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read_sequence_ids(path: str | Path) -> list[str]:
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence_id = str(row.get("ID", "")).strip()
            if sequence_id:
                ids.append(sequence_id)
    return ids


def _read_best_evalues(path: str | Path) -> dict[str, float]:
    best: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 11:
                continue
            sequence_id = str(row[0]).strip()
            try:
                evalue = float(row[10])
            except ValueError:
                continue
            if sequence_id not in best or evalue < best[sequence_id]:
                best[sequence_id] = evalue
    return best


def _canonical_group(sequence_id: str) -> str:
    sequence_id = str(sequence_id)
    if sequence_id.startswith("Natural"):
        return "Natural"
    if sequence_id.startswith("VAE"):
        return "VAE-generated"
    if sequence_id.startswith("GAN") or sequence_id.startswith("preGAN"):
        return "preGAN-generated"
    if sequence_id.startswith("WGAN"):
        return "WGAN-generated"
    if sequence_id.startswith("Random"):
        return "Random"
    return "Other"


def _prepare_group_values(sequence_id_csv: str | Path, blast_result_txt: str | Path) -> dict[str, list[float]]:
    ids = _read_sequence_ids(sequence_id_csv)
    best_hits = _read_best_evalues(blast_result_txt)
    grouped: dict[str, list[float]] = defaultdict(list)
    for sequence_id in ids:
        evalue = best_hits.get(sequence_id, 1.0)
        grouped[_canonical_group(sequence_id)].append(math.log10(evalue + 1e-300))
    return dict(grouped)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_rotated_text(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    angle: int = 90,
) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    measure_draw = ImageDraw.Draw(dummy)
    width, height = _text_size(measure_draw, text, font)
    text_image = Image.new("RGBA", (width + 12, height + 12), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_image)
    draw.text((6, 6), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=position)


def render_blast_histogram(
    sequence_id_csv: str | Path,
    blast_result_txt: str | Path,
    output_path: str | Path,
    include_groups: list[str],
    display_labels: dict[str, str],
    title: str = "BLAST e-value Histogram by Sequence Type",
    canvas_size: tuple[int, int] = (2400, 1500),
) -> None:
    grouped = _prepare_group_values(sequence_id_csv, blast_result_txt)
    filtered = {group: grouped.get(group, []) for group in include_groups}
    all_values = [value for values in filtered.values() for value in values]
    if not all_values:
        raise ValueError("No BLAST values available for histogram rendering.")

    colors = {
        "Natural": "#6b6ecf",
        "VAE-generated": "#bcbd22",
        "preGAN-generated": "#d62728",
        "WGAN-generated": "#ff7f0e",
        "Random": "#17becf",
    }

    edges = np.linspace(min(all_values), max(all_values), 31)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    densities: dict[str, np.ndarray] = {}
    stacked_total = np.zeros(len(centers), dtype=float)
    for group in include_groups:
        values = filtered[group]
        counts, _ = np.histogram(values, bins=edges)
        density = counts.astype(float) / max(len(values), 1)
        densities[group] = density
        stacked_total += density

    max_density = float(max(stacked_total.max(), 1e-6))
    x_min = float(edges[0])
    x_max = float(edges[-1])
    width_px, height_px = canvas_size
    left = 220
    right = 90
    top = 130
    bottom = 140
    plot_left = left
    plot_right = width_px - right
    plot_top = top
    plot_bottom = height_px - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGBA", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(88, bold=True)
    axis_font = _load_font(54, bold=True)
    tick_font = _load_font(34, bold=True)
    legend_font = _load_font(36, bold=True)
    legend_title_font = _load_font(42, bold=True)

    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=3)

    def scale_x(value: float) -> float:
        return plot_left + ((value - x_min) / max(x_max - x_min, 1e-6)) * plot_width

    def scale_y(value: float) -> float:
        return plot_bottom - (value / max_density) * plot_height

    bottom_stack = np.zeros(len(centers), dtype=float)
    for group in include_groups:
        density = densities[group]
        for index, center in enumerate(centers):
            x0 = scale_x(float(center - widths[index] / 2.0))
            x1 = scale_x(float(center + widths[index] / 2.0))
            y0 = scale_y(float(bottom_stack[index] + density[index]))
            y1 = scale_y(float(bottom_stack[index]))
            draw.rectangle([x0, y0, x1, y1], fill=colors[group], outline="black", width=1)
        bottom_stack += density

    draw.text((width_px // 2, 46), title, font=title_font, fill="black", anchor="ma")
    draw.text((plot_left + plot_width // 2, height_px - 62), r"log10 e-value", font=axis_font, fill="black", anchor="ma")
    _draw_rotated_text(image, (42, plot_top + plot_height // 2 - 150), "Density", axis_font, "black")

    x_tick_start = int(math.floor(x_min / 10.0) * 10)
    x_tick_end = int(math.ceil(x_max / 10.0) * 10)
    for tick in range(x_tick_start, x_tick_end + 1, 10):
        x = scale_x(float(tick))
        draw.line([x, plot_bottom, x, plot_bottom + 20], fill="black", width=3)
        draw.text((x, plot_bottom + 34), str(tick), font=tick_font, fill="black", anchor="ma")

    for fraction in np.linspace(0.0, max_density, 6):
        y = scale_y(float(fraction))
        draw.line([plot_left - 18, y, plot_left, y], fill="black", width=3)
        draw.text((plot_left - 30, y), f"{fraction:.2f}", font=tick_font, fill="black", anchor="ra")

    legend_x = plot_left + 28
    legend_y = plot_top + 28
    legend_width = 520
    legend_height = 72 + 64 * len(include_groups)
    draw.rounded_rectangle(
        [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
        radius=18,
        outline="#d0d0d0",
        width=3,
        fill="white",
    )
    draw.text((legend_x + 24, legend_y + 18), "Group", font=legend_title_font, fill="black")
    for index, group in enumerate(include_groups):
        row_y = legend_y + 84 + index * 64
        draw.rectangle([legend_x + 26, row_y + 10, legend_x + 58, row_y + 42], fill=colors[group], outline="black", width=2)
        draw.text((legend_x + 78, row_y + 2), display_labels[group], font=legend_font, fill="black")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output)


def render_legacy_blast_figure_pair(
    sequence_id_csv: str | Path,
    blast_result_txt: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_one = output_dir / "blast_evalue_hist_stacked.png"
    render_blast_histogram(
        sequence_id_csv=sequence_id_csv,
        blast_result_txt=blast_result_txt,
        output_path=output_one,
        include_groups=["Natural", "VAE-generated", "preGAN-generated", "Random"],
        display_labels={
            "Natural": "Natural",
            "VAE-generated": "VAE-generated",
            "preGAN-generated": "GAN-generated",
            "Random": "Random",
        },
        canvas_size=(2400, 1500),
    )

    output_two = output_dir / "blast_evalue_hist_stacked2.png"
    render_blast_histogram(
        sequence_id_csv=sequence_id_csv,
        blast_result_txt=blast_result_txt,
        output_path=output_two,
        include_groups=["Natural", "VAE-generated", "preGAN-generated", "WGAN-generated", "Random"],
        display_labels={
            "Natural": "Natural",
            "VAE-generated": "VAE-generated",
            "preGAN-generated": "preGAN-generated",
            "WGAN-generated": "WGAN-generated",
            "Random": "Random",
        },
        canvas_size=(2700, 1500),
    )

    return [output_one, output_two]
