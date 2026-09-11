from __future__ import annotations

import csv
import html
import itertools
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from tomato_promoter_designer.evaluation.edit_distance import levenshtein


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mean_x = _mean(x_values)
    mean_y = _mean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _write_svg(lines: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _prefer_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _provided_or_default(provided: str | Path | None, default: Path) -> Path:
    if provided is None:
        return default
    path = Path(provided)
    return path if path.exists() else default


def _polyline(values: list[float], x0: float, y0: float, width: float, height: float) -> str:
    if not values:
        return ""
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum if maximum > minimum else 1.0
    points: list[str] = []
    for index, value in enumerate(values):
        x = x0 + index / max(len(values) - 1, 1) * width
        y = y0 + height - ((value - minimum) / span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_deepseed_training_curve(rows: list[dict[str, str]], output_path: str | Path) -> None:
    width = 1040
    height = 700
    left = 90
    top = 110
    plot_w = 880
    plot_h = 150
    block_gap = 60
    train_loss = [float(row["train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]
    test_coefs = [float(row["test_coefs"]) for row in rows]

    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">DeepSeed Predictor Training Summary</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Reconstructed from the archived training log. Curves summarize optimization and validation correlation.</text>')

    for block_index, (title, values, color) in enumerate(
        (
            ("Train Loss", train_loss, "#c85c38"),
            ("Validation Loss", test_loss, "#4f6d7a"),
            ("Validation Pearson", test_coefs, "#c0a145"),
        )
    ):
        y = top + block_index * (plot_h + block_gap)
        lines.append(f'<text x="{left}" y="{y - 18}" font-size="13" font-family="Arial, sans-serif" fill="#33454d">{_escape(title)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="{plot_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')
        lines.append(f'<line x1="{left + 18}" y1="{y + plot_h - 16}" x2="{left + plot_w - 18}" y2="{y + plot_h - 16}" stroke="#d8d2c7"/>')
        lines.append(f'<line x1="{left + 18}" y1="{y + 12}" x2="{left + 18}" y2="{y + plot_h - 16}" stroke="#d8d2c7"/>')
        polyline = _polyline(values, left + 18, y + 14, plot_w - 36, plot_h - 28)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        lines.append(f'<text x="{left + plot_w - 8}" y="{y + 18}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">epochs {len(values)}</text>')
        lines.append(f'<text x="{left + plot_w - 8}" y="{y + 34}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">last {values[-1]:.4f}</text>')

    lines.append(
        f'<text x="{left}" y="{height - 28}" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">DeepSeed summary recreated from the archived optimization log; expanded margins prevent bottom-panel clipping.</text>'
    )
    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def render_mpravae_loss_dashboard(rows: list[dict[str, str]], output_path: str | Path) -> None:
    metrics = [
        ("train_loss", "Total Train", "#c85c38"),
        ("val_loss", "Total Val", "#4f6d7a"),
        ("train_recon", "Recon Train", "#7aa36f"),
        ("val_recon", "Recon Val", "#557153"),
        ("train_pred", "Pred Train", "#c0a145"),
        ("val_pred", "Pred Val", "#8d6b2f"),
    ]
    width = 1100
    height = 860
    panel_w = 470
    panel_h = 180
    left = 70
    top = 110
    gap_x = 40
    gap_y = 70

    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">TransVAE Training Loss Dashboard</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Reconstructed from the archived loss history used by the original training workflow.</text>')

    for metric_index, (column, title, color) in enumerate(metrics):
        row_index = metric_index // 2
        col_index = metric_index % 2
        x = left + col_index * (panel_w + gap_x)
        y = top + row_index * (panel_h + gap_y)
        values = [float(row[column]) for row in rows]
        lines.append(f'<text x="{x}" y="{y - 18}" font-size="13" font-family="Arial, sans-serif" fill="#33454d">{_escape(title)}</text>')
        lines.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')
        polyline = _polyline(values, x + 18, y + 14, panel_w - 36, panel_h - 28)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{polyline}"/>')
        lines.append(f'<text x="{x + panel_w - 10}" y="{y + 18}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">min {min(values):.4f}</text>')
        lines.append(f'<text x="{x + panel_w - 10}" y="{y + 34}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">last {values[-1]:.4f}</text>')

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def _kmer_frequency(sequences: list[str], k: int, start: int | None = None, end: int | None = None) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for sequence in sequences:
        subsequence = sequence[slice(start, end)] if (start is not None or end is not None) else sequence
        for index in range(len(subsequence) - k + 1):
            kmer = subsequence[index : index + k]
            counts[kmer] += 1
            total += 1
    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def _sample_evenly(sequences: list[str], limit: int) -> list[str]:
    if len(sequences) <= limit:
        return sequences
    indices = np.linspace(0, len(sequences) - 1, num=limit, dtype=int)
    return [sequences[index] for index in indices.tolist()]


def _build_kmer_vocab(k: int) -> list[str]:
    return ["".join(kmer) for kmer in itertools.product("ACGT", repeat=k)]


def _kmer_vector(sequence: str, k: int, vocab_index: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab_index), dtype=float)
    sequence = sequence.upper()
    valid_bases = {"A", "C", "G", "T"}
    total = 0
    for index in range(len(sequence) - k + 1):
        kmer = sequence[index : index + k]
        if set(kmer) <= valid_bases:
            vector[vocab_index[kmer]] += 1.0
            total += 1
    if total:
        vector /= float(total)
    return vector


def _pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), [0.0, 0.0]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=float), [0.0, 0.0]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] < 2:
        pad = np.zeros((components.shape[0], 2 - components.shape[1]), dtype=float)
        components = np.hstack([components, pad])
    coordinates = centered @ components[:, :2]
    total_variance = float(np.sum(singular_values**2))
    explained = []
    for index in range(2):
        if index < len(singular_values) and total_variance > 0:
            explained.append(float((singular_values[index] ** 2) / total_variance))
        else:
            explained.append(0.0)
    return coordinates, explained


def render_kmer_scatter_panels(rows: list[dict[str, str]], output_path: str | Path) -> None:
    original_sequences = [row["orig_sequence"] for row in rows]
    designed_sequences = [row["generated_sequence"] for row in rows]
    regions = [
        ("All Region", None, None),
        ("Proximal 20 bp", 0, 20),
        ("Distal 20 bp", -20, None),
    ]
    width = 1080
    height = 420
    panel_w = 300
    panel_h = 250
    start_x = 40
    top = 110
    gap = 30
    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">TransVAE k-mer Distribution Check</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">k-mer frequency comparison between natural and generated promoters.</text>')

    for panel_index, (title, start, end) in enumerate(regions):
        x = start_x + panel_index * (panel_w + gap)
        y = top
        original = _kmer_frequency(original_sequences, 4, start=start, end=end)
        designed = _kmer_frequency(designed_sequences, 4, start=start, end=end)
        kmers = sorted(set(original) | set(designed))
        x_values = [original.get(kmer, 0.0) for kmer in kmers]
        y_values = [designed.get(kmer, 0.0) for kmer in kmers]
        maximum = max(x_values + y_values) if kmers else 1.0
        maximum = max(maximum, 1e-6)
        corr = _pearson(x_values, y_values)

        lines.append(f'<text x="{x}" y="{y - 18}" font-size="13" font-family="Arial, sans-serif" fill="#33454d">{_escape(title)}</text>')
        lines.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')
        lines.append(f'<line x1="{x + 30}" y1="{y + panel_h - 26}" x2="{x + panel_w - 20}" y2="{y + panel_h - 26}" stroke="#c8c3b7"/>')
        lines.append(f'<line x1="{x + 30}" y1="{y + panel_h - 26}" x2="{x + 30}" y2="{y + 18}" stroke="#c8c3b7"/>')
        lines.append(f'<line x1="{x + 30}" y1="{y + panel_h - 26}" x2="{x + panel_w - 20}" y2="{y + 18}" stroke="#e3ded3" stroke-dasharray="4 4"/>')
        for xv, yv in zip(x_values, y_values):
            px = x + 30 + xv / maximum * (panel_w - 50)
            py = y + panel_h - 26 - yv / maximum * (panel_h - 44)
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="#b15a36" opacity="0.7"/>')
        lines.append(f'<text x="{x + panel_w - 12}" y="{y + 20}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">r {corr:.3f}</text>')
        lines.append(f'<text x="{x + panel_w - 12}" y="{y + 36}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6a7478">n {len(kmers)}</text>')

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def render_semantic_sequence_space(
    designed_rows: list[dict[str, str]],
    training_rows: list[dict[str, str]],
    output_path: str | Path,
    limit: int = 220,
    k: int = 4,
) -> None:
    natural_sequences = [row.get("realB", "").strip() for row in training_rows if row.get("realB", "").strip()]
    generated_sequences = [row.get("generated_sequence", "").strip() for row in designed_rows if row.get("generated_sequence", "").strip()]
    if not natural_sequences:
        natural_sequences = [row.get("orig_sequence", "").strip() for row in designed_rows if row.get("orig_sequence", "").strip()]
    natural_sequences = _sample_evenly(natural_sequences, limit)
    generated_sequences = _sample_evenly(generated_sequences, limit)
    if not natural_sequences or not generated_sequences:
        raise ValueError("Semantic space rendering requires both natural and generated promoter sequences.")

    vocab = _build_kmer_vocab(k)
    vocab_index = {kmer: index for index, kmer in enumerate(vocab)}
    natural_matrix = np.vstack([_kmer_vector(sequence, k, vocab_index) for sequence in natural_sequences])
    generated_matrix = np.vstack([_kmer_vector(sequence, k, vocab_index) for sequence in generated_sequences])
    combined = np.vstack([natural_matrix, generated_matrix])
    coordinates, explained = _pca_2d(combined)
    natural_coords = coordinates[: len(natural_sequences)]
    generated_coords = coordinates[len(natural_sequences) :]

    natural_mean = natural_matrix.mean(axis=0)
    generated_mean = generated_matrix.mean(axis=0)
    corr = _pearson(natural_mean.tolist(), generated_mean.tolist())

    all_x = coordinates[:, 0].tolist() if len(coordinates) else [0.0]
    all_y = coordinates[:, 1].tolist() if len(coordinates) else [0.0]
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    width = 980
    height = 700
    left = 110
    top = 110
    plot_w = 700
    plot_h = 470

    def scale_x(value: float) -> float:
        return left + ((value - min_x) / span_x) * plot_w

    def scale_y(value: float) -> float:
        return top + plot_h - ((value - min_y) / span_y) * plot_h

    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">TransVAE Semantic Sequence Space</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Dependency-light sequence-space reconstruction using PCA on 4-mer frequency vectors.</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')

    zero_x = scale_x(0.0) if min_x <= 0.0 <= max_x else left
    zero_y = scale_y(0.0) if min_y <= 0.0 <= max_y else top + plot_h
    lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#ddd7cb" stroke-dasharray="4 4"/>')
    lines.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{top + plot_h}" stroke="#ddd7cb" stroke-dasharray="4 4"/>')

    for x_value, y_value in natural_coords:
        lines.append(f'<circle cx="{scale_x(float(x_value)):.2f}" cy="{scale_y(float(y_value)):.2f}" r="3.0" fill="#4f6d7a" opacity="0.55"/>')
    for x_value, y_value in generated_coords:
        lines.append(f'<circle cx="{scale_x(float(x_value)):.2f}" cy="{scale_y(float(y_value)):.2f}" r="3.0" fill="#c85c38" opacity="0.55"/>')

    natural_centroid = natural_coords.mean(axis=0)
    generated_centroid = generated_coords.mean(axis=0)
    for label, centroid, color in (
        ("Natural centroid", natural_centroid, "#2f4b57"),
        ("Generated centroid", generated_centroid, "#9d4224"),
    ):
        cx = scale_x(float(centroid[0]))
        cy = scale_y(float(centroid[1]))
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        lines.append(f'<text x="{cx + 12:.2f}" y="{cy - 10:.2f}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">{_escape(label)}</text>')

    legend_x = left + plot_w + 26
    legend_y = top + 30
    lines.append(f'<text x="{legend_x}" y="{legend_y - 12}" font-size="13" font-family="Arial, sans-serif" fill="#33454d">Groups</text>')
    lines.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 8}" r="5" fill="#4f6d7a" opacity="0.7"/>')
    lines.append(f'<text x="{legend_x + 24}" y="{legend_y + 12}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">Natural promoters (n={len(natural_sequences)})</text>')
    lines.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 34}" r="5" fill="#c85c38" opacity="0.7"/>')
    lines.append(f'<text x="{legend_x + 24}" y="{legend_y + 38}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">Generated promoters (n={len(generated_sequences)})</text>')

    stat_y = legend_y + 82
    lines.append(f'<text x="{legend_x}" y="{stat_y}" font-size="13" font-family="Arial, sans-serif" fill="#33454d">Embedding summary</text>')
    lines.append(f'<text x="{legend_x}" y="{stat_y + 24}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">PC1 variance {explained[0] * 100:.1f}%</text>')
    lines.append(f'<text x="{legend_x}" y="{stat_y + 44}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">PC2 variance {explained[1] * 100:.1f}%</text>')
    lines.append(f'<text x="{legend_x}" y="{stat_y + 64}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">Mean 4-mer Pearson r {corr:.3f}</text>')
    lines.append(f'<text x="{legend_x}" y="{stat_y + 84}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">Interpretation goal: inspect manifold overlap</text>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 42}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#33454d">Principal component 1</text>')
    lines.append(f'<text x="42" y="{top + plot_h / 2:.1f}" font-size="12" font-family="Arial, sans-serif" fill="#33454d">Principal component 2</text>')
    lines.append(f'<text x="{left}" y="{height - 28}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">This replaces the original ad hoc t-SNE script with a deterministic projection that is easier to reproduce inside the unified tool.</text>')
    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def render_prediction_scatter(rows: list[dict[str, str]], output_path: str | Path) -> None:
    predicted = [float(row["Predicted_Expr"]) for row in rows]
    truth = [float(row["True_Expr"]) for row in rows]
    corr = _pearson(predicted, truth)
    minimum = min(predicted + truth) if rows else 0.0
    maximum = max(predicted + truth) if rows else 1.0
    span = maximum - minimum if maximum > minimum else 1.0
    width = 780
    height = 660
    left = 120
    top = 110
    plot_w = 560
    plot_h = 430

    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">TransVAE Prediction Scatter</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Predicted versus measured expression values from the archived validation table.</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#c8c3b7"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#c8c3b7"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#e1dbd0" stroke-dasharray="5 5"/>')

    for pred, true in zip(predicted, truth):
        x = left + (pred - minimum) / span * plot_w
        y = top + plot_h - (true - minimum) / span * plot_h
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="#b15a36" opacity="0.65"/>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 46}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#33454d">Predicted expression</text>')
    lines.append(f'<text x="48" y="{top + plot_h / 2:.1f}" font-size="12" font-family="Arial, sans-serif" fill="#33454d">True expression</text>')
    lines.append(f'<text x="{left + plot_w - 10}" y="{top + 24}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#6a7478">Pearson r {corr:.3f}</text>')
    lines.append(f'<text x="{left + plot_w - 10}" y="{top + 42}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#6a7478">n {len(rows)}</text>')
    lines.append(f'<text x="{left}" y="{top + plot_h + 68}" font-size="11" font-family="Arial, sans-serif" fill="#5b666b">Expanded margins preserve axis labels and prevent edge-cropping in exported viewers.</text>')

    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


def _all_vs_others(group: list[str], others_a: list[str], others_b: list[str]) -> list[int]:
    distances: list[int] = []
    for sequence in group:
        distances.extend(levenshtein(sequence, other) for other in others_a)
        distances.extend(levenshtein(sequence, other) for other in others_b)
    return distances


def _histogram(values: list[int], bins: int = 18) -> tuple[list[float], list[float]]:
    if not values:
        return [0.0], [0.0]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [float(minimum)], [1.0]
    width = (maximum - minimum) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int((value - minimum) / width))
        counts[index] += 1
    total = sum(counts) or 1
    centers = [minimum + (index + 0.5) * width for index in range(bins)]
    densities = [count / total for count in counts]
    return centers, densities


def render_edit_distance_diversity(
    mutated_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
    output_path: str | Path,
    limit: int = 60,
) -> None:
    original_sequences = [row["orig_sequence"] for row in mutated_rows[:limit]]
    generated_sequences = [row["generated_sequence"] for row in mutated_rows[:limit]]
    random_sequences = [row[next(iter(row.keys()))] for row in random_rows[:limit]]

    natural_distances = _all_vs_others(original_sequences, random_sequences, generated_sequences)
    random_distances = _all_vs_others(random_sequences, original_sequences, generated_sequences)
    generated_distances = _all_vs_others(generated_sequences, original_sequences, random_sequences)

    all_values = natural_distances + random_distances + generated_distances
    minimum = min(all_values) if all_values else 0
    maximum = max(all_values) if all_values else 1
    span = maximum - minimum if maximum > minimum else 1.0

    width = 900
    height = 560
    left = 90
    top = 100
    plot_w = 720
    plot_h = 340

    lines = _svg_header(width, height)
    lines.append('<text x="36" y="40" font-size="24" font-family="Arial, sans-serif" fill="#1d2a2f">TransVAE Edit-distance Diversity</text>')
    lines.append('<text x="36" y="64" font-size="12" font-family="Arial, sans-serif" fill="#5b666b">Diversity summary comparing natural, random, and generated promoter groups.</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="8" fill="#fffdf8" stroke="#ddd7cb"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#c8c3b7"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left}" y2="{top}" stroke="#c8c3b7"/>')

    series = [
        ("Natural", natural_distances, "#5d6cc1"),
        ("Random", random_distances, "#d4795b"),
        ("VAE", generated_distances, "#5ca36d"),
    ]

    for label, distances, color in series:
        centers, densities = _histogram(distances)
        max_density = max(densities) if densities else 1.0
        points: list[str] = []
        for center, density in zip(centers, densities):
            x = left + ((center - minimum) / span) * plot_w
            y = top + plot_h - (density / max_density) * (plot_h - 40)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{" ".join(points)}"/>')

    legend_x = left + plot_w - 120
    for index, (label, distances, color) in enumerate(series):
        y = top + 24 + index * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 32}" y="{y + 4}" font-size="11" font-family="Arial, sans-serif" fill="#33454d">{_escape(label)} mean {(_mean(distances)):.1f}</text>'
        )

    lines.append(f'<text x="{left}" y="{top + plot_h + 28}" font-size="12" font-family="Arial, sans-serif" fill="#33454d">Edit distance</text>')
    lines.append(f'<text x="{left}" y="{top - 12}" font-size="12" font-family="Arial, sans-serif" fill="#33454d">Relative frequency</text>')
    lines.extend(_svg_footer())
    _write_svg(lines, output_path)


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_MPRAVAE_LOSS = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "loss_history.csv",
    WORKSPACE_ROOT / "MpraVAE" / "code" / "transformerresult" / "results1" / "loss_curves" / "loss_history.csv",
)
DEFAULT_MPRAVAE_DESIGNED = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "designed_promoters.csv",
    WORKSPACE_ROOT / "MpraVAE" / "results" / "designed_promoters.csv",
)
DEFAULT_MPRAVAE_PRED = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "generated_prediction_results.csv",
    WORKSPACE_ROOT / "MpraVAE" / "results" / "generated_prediction_results.csv",
)
DEFAULT_MPRAVAE_MUTATED = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "mutated_file.csv",
    WORKSPACE_ROOT / "MpraVAE" / "data" / "bianjijuli" / "mutated_file.csv",
)
DEFAULT_MPRAVAE_RANDOM = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "random_promoters_200.csv",
    WORKSPACE_ROOT / "MpraVAE" / "data" / "random_promoters_200.csv",
)
DEFAULT_MPRAVAE_TRAINING_SET = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "training_set.csv",
    WORKSPACE_ROOT / "MpraVAE" / "data" / "vaedata" / "training_set.csv",
)
DEFAULT_DNABERT_MOTIF_SUMMARY = _prefer_existing(
    REPO_ROOT / "data" / "processed" / "dnabert_legacy" / "motif_summary.csv",
    REPO_ROOT / "outputs" / "dnabert_legacy" / "motif_summary.csv",
)
DEFAULT_DNABERT_TFBS_DIRS = [
    REPO_ROOT / "data" / "raw" / "dnabert" / "tfbs_assets",
    WORKSPACE_ROOT / "DNABERT" / "motif" / "result" / "6-2",
    WORKSPACE_ROOT / "DNABERT" / "motif" / "result" / "6-3",
]
DEFAULT_DEEPSEED_SCATTER = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "pregan_expression" / "scatter_165_mpra_expr_denselstm.png",
    WORKSPACE_ROOT / "deepseed" / "Predictor" / "results" / "scatter_fig" / "scatter_165_mpra_expr_denselstm.png",
)
DEFAULT_MPRAVAE_BLAST_DIR = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "blast",
    WORKSPACE_ROOT / "MpraVAE" / "results" / "blast",
)
DEFAULT_MPRAVAE_DIVERSITY_DIR = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "transvae" / "diversity",
    WORKSPACE_ROOT / "MpraVAE" / "results" / "bianjijuli",
)
DEFAULT_DEEPSEED_TRAIN = _prefer_existing(
    REPO_ROOT / "data" / "raw" / "pregan_expression" / "training_log165_mpra_expr_denselstm.csv",
    WORKSPACE_ROOT / "deepseed" / "Predictor" / "results" / "training_log165_mpra_expr_denselstm.csv",
)


def export_legacy_figure_bundle(
    output_dir: str | Path,
    mpravae_loss_history: str | Path | None = None,
    mpravae_designed_promoters: str | Path | None = None,
    mpravae_prediction_results: str | Path | None = None,
    deepseed_training_log: str | Path | None = None,
    mpravae_mutated_file: str | Path | None = None,
    mpravae_random_promoters: str | Path | None = None,
    mpravae_training_set: str | Path | None = None,
    dnabert_motif_summary: str | Path | None = None,
    dnabert_tfbs_dir: str | Path | None = None,
    deepseed_scatter_png: str | Path | None = None,
    mpravae_blast_dir: str | Path | None = None,
    mpravae_diversity_dir: str | Path | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"output_dir": str(output_dir), "generated": []}

    loss_path = _provided_or_default(mpravae_loss_history, DEFAULT_MPRAVAE_LOSS)
    if loss_path.exists():
        rows = _read_csv(loss_path)
        file_path = output_dir / "transvae_loss_dashboard.svg"
        render_mpravae_loss_dashboard(rows, file_path)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(file_path.name)

    designed_path = _provided_or_default(mpravae_designed_promoters, DEFAULT_MPRAVAE_DESIGNED)
    if designed_path.exists():
        rows = _read_csv(designed_path)
        file_path = output_dir / "transvae_kmer_scatter.svg"
        render_kmer_scatter_panels(rows, file_path)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(file_path.name)

        training_path = _provided_or_default(mpravae_training_set, DEFAULT_MPRAVAE_TRAINING_SET)
        training_rows = _read_csv(training_path) if training_path.exists() else []
        if training_rows or any(row.get("orig_sequence") for row in rows):
            semantic_path = output_dir / "transvae_semantic_space.svg"
            render_semantic_sequence_space(rows, training_rows, semantic_path)
            cast = manifest["generated"]
            assert isinstance(cast, list)
            cast.append(semantic_path.name)

    prediction_path = _provided_or_default(mpravae_prediction_results, DEFAULT_MPRAVAE_PRED)
    if prediction_path.exists():
        rows = _read_csv(prediction_path)
        file_path = output_dir / "transvae_prediction_scatter.svg"
        render_prediction_scatter(rows, file_path)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(file_path.name)

    deepseed_path = _provided_or_default(deepseed_training_log, DEFAULT_DEEPSEED_TRAIN)
    if deepseed_path.exists():
        rows = _read_csv(deepseed_path)
        file_path = output_dir / "pregan_expression_training_curve.svg"
        render_deepseed_training_curve(rows, file_path)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(file_path.name)

    mutated_path = _provided_or_default(mpravae_mutated_file, DEFAULT_MPRAVAE_MUTATED)
    random_path = _provided_or_default(mpravae_random_promoters, DEFAULT_MPRAVAE_RANDOM)
    if mutated_path.exists() and random_path.exists():
        mutated_rows = _read_csv(mutated_path)
        random_rows = _read_csv(random_path)
        file_path = output_dir / "transvae_edit_distance_diversity.svg"
        render_edit_distance_diversity(mutated_rows, random_rows, file_path)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(file_path.name)

    motif_summary_path = _provided_or_default(dnabert_motif_summary, DEFAULT_DNABERT_MOTIF_SUMMARY)
    if dnabert_tfbs_dir and Path(dnabert_tfbs_dir).exists():
        tfbs_dirs = [Path(dnabert_tfbs_dir)]
    else:
        tfbs_dirs = DEFAULT_DNABERT_TFBS_DIRS
    if motif_summary_path.exists():
        motif_rows = _read_csv(motif_summary_path)
        motif_output_dir = output_dir / "dnabert_tfbs_assets"
        motif_output_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for row in motif_rows[:12]:
            motif_name = row["motif"]
            filename = f"TFBS_{motif_name}.png"
            for source_dir in tfbs_dirs:
                source_path = source_dir / filename
                if source_path.exists():
                    destination = motif_output_dir / filename
                    shutil.copy2(source_path, destination)
                    copied.append(str(destination.relative_to(output_dir)))
                    break
        if copied:
            cast = manifest["generated"]
            assert isinstance(cast, list)
            cast.extend(copied)

    deepseed_scatter_path = _provided_or_default(deepseed_scatter_png, DEFAULT_DEEPSEED_SCATTER)
    if deepseed_scatter_path.exists():
        destination = output_dir / deepseed_scatter_path.name
        shutil.copy2(deepseed_scatter_path, destination)
        cast = manifest["generated"]
        assert isinstance(cast, list)
        cast.append(destination.name)

    for source_dir in [
        _provided_or_default(mpravae_blast_dir, DEFAULT_MPRAVAE_BLAST_DIR),
        _provided_or_default(mpravae_diversity_dir, DEFAULT_MPRAVAE_DIVERSITY_DIR),
    ]:
        if source_dir.exists():
            asset_output_dir = output_dir / source_dir.name
            asset_output_dir.mkdir(parents=True, exist_ok=True)
            copied = []
            for source_path in sorted(source_dir.glob("*.png")):
                destination = asset_output_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied.append(str(destination.relative_to(output_dir)))
            if copied:
                cast = manifest["generated"]
                assert isinstance(cast, list)
                cast.extend(copied)

    if not manifest["generated"]:
        raise FileNotFoundError("No legacy figure sources were found for export.")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
