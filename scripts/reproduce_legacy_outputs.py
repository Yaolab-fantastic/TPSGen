from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "data" / "results" / "reproducible_legacy"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

TISSUES = [
    ("expr_tissue_1", "root"),
    ("expr_tissue_2", "stem"),
    ("expr_tissue_3", "leaf"),
    ("expr_tissue_4", "fruit"),
]


SOURCE_FILES = {
    "training_set": REPO_ROOT / "data" / "raw" / "transvae" / "training_set.csv",
    "designed_promoters_200": REPO_ROOT / "data" / "raw" / "transvae" / "designed_promoters200.csv",
    "designed_promoters_20": REPO_ROOT / "data" / "raw" / "transvae" / "designed_promoters.csv",
    "prediction_reference": REPO_ROOT / "data" / "raw" / "transvae" / "generated_prediction_results.csv",
    "random_promoters": REPO_ROOT / "data" / "raw" / "transvae" / "random_promoters_200.csv",
    "edit_pairs": REPO_ROOT / "data" / "raw" / "transvae" / "designed_promoters200_pairs.csv",
    "deepseed_training_log": REPO_ROOT / "data" / "raw" / "pregan_expression" / "training_log165_mpra_expr_denselstm.csv",
    "dnabert_motif_summary": REPO_ROOT / "data" / "processed" / "dnabert_legacy" / "motif_summary.csv",
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Bold.ttf" if bold else "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(34, True)
FONT_PANEL = _font(24, True)
FONT_LABEL = _font(18, True)
FONT_SMALL = _font(14, False)
FONT_TINY = _font(11, False)

COLORS = {
    "bg": (252, 251, 247),
    "ink": (31, 42, 48),
    "muted": (92, 102, 107),
    "grid": (224, 218, 207),
    "cream": (247, 244, 235),
    "orange": (222, 126, 57),
    "blue": (64, 112, 150),
    "green": (91, 145, 105),
    "red": (195, 88, 82),
    "purple": (121, 96, 157),
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_source(key: str) -> Path:
    source = SOURCE_FILES[key]
    if not source.exists():
        raise FileNotFoundError(f"Required repository source is missing: {_rel(source)}")
    return source


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mx = _mean(x_values)
    my = _mean(y_values)
    numerator = sum((x - mx) * (y - my) for x, y in zip(x_values, y_values))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in y_values))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def _save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300))
    return path


def build_expression_heatmap() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("training_set"))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        try:
            values = [(name, label, float(row[name])) for name, label in TISSUES]
        except (KeyError, ValueError):
            continue
        best = max(values, key=lambda item: item[2])
        others = [value for _, _, value in values if value != best[2]]
        margin = best[2] - max(others) if others else 0.0
        grouped[best[1]].append(
            {
                "sequence_id": f"training_{index:05d}",
                "sequence": row.get("realB", ""),
                "score_root": values[0][2],
                "score_stem": values[1][2],
                "score_leaf": values[2][2],
                "score_fruit": values[3][2],
                "preferred_tissue": best[1],
                "target_margin": margin,
            }
        )

    selected: list[dict[str, object]] = []
    for _, tissue in TISSUES:
        candidates = sorted(grouped[tissue], key=lambda item: float(item["target_margin"]), reverse=True)
        selected.extend(candidates[:25])
    selected.sort(key=lambda item: (["root", "stem", "leaf", "fruit"].index(str(item["preferred_tissue"])), -float(item["target_margin"])))

    table_path = TABLE_DIR / "expression_heatmap_source.csv"
    _write_csv(table_path, selected)

    values = [float(row[key]) for row in selected for key in ("score_root", "score_stem", "score_leaf", "score_fruit")]
    vmin, vmax = min(values), max(values)
    x0, y0 = 340, 210
    cell_w, cell_h = 210, 18
    row_gap = 5
    width = 1500
    height = y0 + len(selected) * (cell_h + row_gap) + 230
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S2", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "Tissue-associated expression landscape across retained tomato promoters", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Balanced rows show 25 promoters per preferred tissue selected from the retained training table.", font=FONT_SMALL, fill=COLORS["muted"])
    for col, (_, tissue) in enumerate(TISSUES):
        draw.text((x0 + col * cell_w + 70, y0 - 34), tissue, font=FONT_LABEL, fill=COLORS["ink"])

    tissue_colors = {
        "root": (98, 131, 86),
        "stem": (157, 132, 68),
        "leaf": (74, 141, 105),
        "fruit": COLORS["orange"],
    }
    for i, row in enumerate(selected):
        y = y0 + i * (cell_h + row_gap)
        draw.text((54, y + 1), str(row["sequence_id"]).replace("training_", "tr_"), font=FONT_TINY, fill=COLORS["ink"])
        pref = str(row["preferred_tissue"])
        draw.rounded_rectangle((228, y - 1, 310, y + cell_h), radius=7, fill=tissue_colors[pref])
        draw.text((244, y + 2), pref, font=FONT_TINY, fill=(255, 255, 255))
        for col, key in enumerate(("score_root", "score_stem", "score_leaf", "score_fruit")):
            x = x0 + col * cell_w
            value = float(row[key])
            t = (value - vmin) / max(vmax - vmin, 1e-9)
            fill = _blend((242, 239, 229), (77, 126, 160), t)
            draw.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h), radius=4, fill=fill, outline=COLORS["grid"])
            draw.text((x + 72, y + 2), f"{value:.2f}", font=FONT_TINY, fill=COLORS["ink"])

    legend_x, legend_y = 1040, y0 + len(selected) * (cell_h + row_gap) + 60
    for k in range(180):
        color = _blend((242, 239, 229), (77, 126, 160), k / 179)
        draw.line((legend_x + k, legend_y, legend_x + k, legend_y + 22), fill=color)
    draw.rectangle((legend_x, legend_y, legend_x + 180, legend_y + 22), outline=COLORS["grid"])
    draw.text((legend_x, legend_y - 28), "expression score", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((legend_x, legend_y + 32), f"{vmin:.2f}", font=FONT_TINY, fill=COLORS["muted"])
    draw.text((legend_x + 142, legend_y + 32), f"{vmax:.2f}", font=FONT_TINY, fill=COLORS["muted"])

    figure_path = _save_image(image, FIGURE_DIR / "figS2_expression_heatmap.png")
    return table_path, figure_path


def build_motif_summary() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("dnabert_motif_summary"))
    parsed: list[dict[str, object]] = []
    tfbs_dirs = [
        REPO_ROOT / "data" / "raw" / "dnabert" / "tfbs_assets",
    ]
    for row in rows:
        try:
            count = int(float(row["num_instances"]))
        except (KeyError, ValueError):
            continue
        motif = row.get("motif", "")
        tfbs_logo = ""
        for directory in tfbs_dirs:
            candidate = directory / f"TFBS_{motif}.png"
            if candidate.exists():
                tfbs_logo = _rel(candidate) if str(candidate).startswith(str(REPO_ROOT)) else str(candidate)
                break
        parsed.append(
            {
                "motif": motif,
                "num_instances": count,
                "has_tfbs_logo": "yes" if tfbs_logo else "no",
                "tfbs_logo": tfbs_logo,
            }
        )
    parsed.sort(key=lambda item: int(item["num_instances"]), reverse=True)
    top = parsed[:20]
    table_path = TABLE_DIR / "dnabert_motif_top20.csv"
    _write_csv(table_path, top)

    width, height = 1250, 980
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S3", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "DNABERT-derived retained motif instances", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Top retained motifs are reconstructed from motif_summary.csv generated by DNABERT-derived post-processing.", font=FONT_SMALL, fill=COLORS["muted"])

    max_count = max(int(row["num_instances"]) for row in top) if top else 1
    x0, y0 = 250, 180
    bar_w, row_h = 820, 34
    for i, row in enumerate(top):
        y = y0 + i * row_h
        count = int(row["num_instances"])
        width_px = int(bar_w * count / max_count)
        draw.text((70, y + 6), str(row["motif"]), font=FONT_SMALL, fill=COLORS["ink"])
        draw.rounded_rectangle((x0, y, x0 + width_px, y + 22), radius=7, fill=COLORS["green"])
        draw.text((x0 + width_px + 12, y + 4), str(count), font=FONT_TINY, fill=COLORS["ink"])
    draw.text((x0, height - 70), "Number of retained motif instances", font=FONT_SMALL, fill=COLORS["muted"])

    figure_path = _save_image(image, FIGURE_DIR / "figS3_dnabert_motif_top20.png")
    return table_path, figure_path


def _kmer_counts(sequences: list[str], k: int = 4, start: int | None = None, end: int | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        seq = sequence.upper()
        region = seq[slice(start, end)] if (start is not None or end is not None) else seq
        for i in range(max(0, len(region) - k + 1)):
            kmer = region[i : i + k]
            if set(kmer) <= {"A", "C", "G", "T"}:
                counts[kmer] += 1
    return counts


def _freq_vector(counts: Counter[str], vocab: list[str]) -> list[float]:
    total = sum(counts.values()) or 1
    return [counts.get(kmer, 0) / total for kmer in vocab]


def build_kmer_comparison() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("designed_promoters_200"))
    orig = [row["orig_sequence"] for row in rows if row.get("orig_sequence")]
    gen = [row["generated_sequence"] for row in rows if row.get("generated_sequence")]
    vocab = ["".join(parts) for parts in product("ACGT", repeat=4)]
    regions = [
        ("all", None, None),
        ("proximal_20bp", 0, 20),
        ("distal_20bp", -20, None),
    ]
    out_rows: list[dict[str, object]] = []
    region_stats: list[tuple[str, list[float], list[float], float]] = []
    for region, start, end in regions:
        orig_freq = _freq_vector(_kmer_counts(orig, 4, start, end), vocab)
        gen_freq = _freq_vector(_kmer_counts(gen, 4, start, end), vocab)
        corr = _pearson(orig_freq, gen_freq)
        region_stats.append((region, orig_freq, gen_freq, corr))
        for kmer, o, g in zip(vocab, orig_freq, gen_freq):
            out_rows.append({"region": region, "kmer": kmer, "original_frequency": o, "generated_frequency": g})
    table_path = TABLE_DIR / "kmer_frequency_comparison.csv"
    _write_csv(table_path, out_rows)

    width, height = 1500, 650
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S4", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "Generated-versus-original 4-mer frequency similarity", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Recreated from the retained 200-promoter design table using the original all/proximal/distal region logic.", font=FONT_SMALL, fill=COLORS["muted"])

    panel_w, panel_h = 360, 330
    for idx, (region, x_values, y_values, corr) in enumerate(region_stats):
        x0 = 90 + idx * 470
        y0 = 205
        draw.text((x0, y0 - 36), region.replace("_", " "), font=FONT_LABEL, fill=COLORS["ink"])
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=COLORS["grid"], width=2)
        maximum = max(x_values + y_values) or 1.0
        draw.line((x0, y0 + panel_h, x0 + panel_w, y0), fill=COLORS["grid"], width=2)
        for xv, yv in zip(x_values, y_values):
            px = x0 + int(xv / maximum * panel_w)
            py = y0 + panel_h - int(yv / maximum * panel_h)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=COLORS["orange"])
        draw.text((x0 + panel_w - 92, y0 + 16), f"r={corr:.3f}", font=FONT_SMALL, fill=COLORS["muted"])
        draw.text((x0 + 86, y0 + panel_h + 28), "original", font=FONT_SMALL, fill=COLORS["muted"])
        draw.text((x0 - 8, y0 - 26), "generated", font=FONT_SMALL, fill=COLORS["muted"])
    figure_path = _save_image(image, FIGURE_DIR / "figS4_kmer_frequency_similarity.png")
    return table_path, figure_path


def build_design_candidate_summary() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("designed_promoters_200"))
    from tpsgen.io.schema import SequenceRecord
    from tpsgen.pipeline.predict_transvae import run_transvae_prediction

    original_records = [
        SequenceRecord(f"original_{row.get('input_index', index)}", row.get("orig_sequence", ""))
        for index, row in enumerate(rows)
        if row.get("orig_sequence", "")
    ]
    original_predictions = {
        record.sequence_id: prediction
        for record, prediction in zip(original_records, run_transvae_prediction(original_records))
    }
    out_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            exprs = [float(row[f"pred_expr{i}"]) for i in range(1, 5)]
        except (KeyError, ValueError):
            continue
        original_id = f"original_{row.get('input_index', index)}"
        original_prediction = original_predictions.get(original_id)
        original_scores = (
            [
                original_prediction.score_root,
                original_prediction.score_stem,
                original_prediction.score_leaf,
                original_prediction.score_fruit,
            ]
            if original_prediction is not None
            else [float("nan")] * 4
        )
        mutations = sum(a != b for a, b in zip(row.get("orig_sequence", ""), row.get("generated_sequence", "")))
        fruit_margin = exprs[3] - max(exprs[:3])
        original_fruit_margin = original_scores[3] - max(original_scores[:3])
        out_rows.append(
            {
                "input_index": row.get("input_index", ""),
                "orig_root": original_scores[0],
                "orig_stem": original_scores[1],
                "orig_leaf": original_scores[2],
                "orig_fruit": original_scores[3],
                "orig_fruit_margin": original_fruit_margin,
                "pred_root": exprs[0],
                "pred_stem": exprs[1],
                "pred_leaf": exprs[2],
                "pred_fruit": exprs[3],
                "fruit_margin": fruit_margin,
                "num_point_differences": mutations,
                "orig_length": len(row.get("orig_sequence", "")),
                "generated_length": len(row.get("generated_sequence", "")),
            }
        )
    table_path = TABLE_DIR / "design_candidate_summary.csv"
    _write_csv(table_path, out_rows)

    fruit_scores = [float(row["pred_fruit"]) for row in out_rows]
    mutations = [float(row["num_point_differences"]) for row in out_rows]
    margins = [float(row["fruit_margin"]) for row in out_rows]
    width, height = 1100, 760
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S5", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "Retained fruit-targeted design candidate summary", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Each point is a retained designed promoter from designed_promoters200.csv.", font=FONT_SMALL, fill=COLORS["muted"])

    x0, y0, plot_w, plot_h = 130, 190, 760, 430
    draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=COLORS["grid"], width=2)
    min_x, max_x = min(fruit_scores), max(fruit_scores)
    min_y, max_y = min(mutations), max(mutations)
    min_m, max_m = min(margins), max(margins)
    for score, mut, margin in zip(fruit_scores, mutations, margins):
        sx = (score - min_x) / max(max_x - min_x, 1e-9)
        sy = (mut - min_y) / max(max_y - min_y, 1e-9)
        sm = (margin - min_m) / max(max_m - min_m, 1e-9)
        px = x0 + int(sx * plot_w)
        py = y0 + plot_h - int(sy * plot_h)
        color = _blend(COLORS["blue"], COLORS["orange"], sm)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline=(255, 255, 255))
    draw.text((x0 + 230, y0 + plot_h + 36), "predicted fruit score", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 - 92, y0 + 190), "point differences", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 + plot_w + 34, y0 + 20), f"n={len(out_rows)}", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 + plot_w + 34, y0 + 48), f"median edits={sorted(mutations)[len(mutations)//2]:.0f}", font=FONT_SMALL, fill=COLORS["muted"])
    figure_path = _save_image(image, FIGURE_DIR / "FigS5_design_candidate_summary.png")
    return table_path, figure_path


def build_prediction_reference() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("prediction_reference"))
    predicted: list[float] = []
    true_values: list[float] = []
    for row in rows:
        try:
            predicted.append(float(row["Predicted_Expr"]))
            true_values.append(float(row["True_Expr"]))
        except (KeyError, ValueError):
            continue
    r = _pearson(predicted, true_values)
    stats_path = TABLE_DIR / "prediction_reference_stats.csv"
    _write_csv(
        stats_path,
        [
            {
                "source_file": _rel(_resolve_source("prediction_reference")),
                "predicted_column": "Predicted_Expr",
                "true_column": "True_Expr",
                "source_rows": len(rows),
                "valid_pairs": len(predicted),
                "excluded_rows": len(rows) - len(predicted),
                "pearson_r": r,
            }
        ],
    )

    width, height = 900, 780
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S6", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "Retained quantitative prediction reference", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Predicted and true scalar-expression pairs retained from the project modeling workflow.", font=FONT_SMALL, fill=COLORS["muted"])
    x0, y0, plot_w, plot_h = 140, 190, 600, 460
    draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=COLORS["grid"], width=2)
    mn, mx = min(predicted + true_values), max(predicted + true_values)
    span = max(mx - mn, 1e-9)
    draw.line((x0, y0 + plot_h, x0 + plot_w, y0), fill=COLORS["grid"], width=2)
    for p, t in zip(predicted, true_values):
        px = x0 + int((p - mn) / span * plot_w)
        py = y0 + plot_h - int((t - mn) / span * plot_h)
        draw.point((px, py), fill=COLORS["red"])
    draw.text((x0 + plot_w - 140, y0 + 22), f"Pearson r={r:.3f}", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 + plot_w - 140, y0 + 48), f"n={len(predicted)}", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 + 200, y0 + plot_h + 38), "predicted expression", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 - 92, y0 + 200), "true expression", font=FONT_SMALL, fill=COLORS["muted"])
    figure_path = _save_image(image, FIGURE_DIR / "figS6_prediction_reference_scatter.png")
    return stats_path, figure_path


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def build_edit_distance_summary() -> tuple[Path, Path]:
    pair_rows = _read_csv(_resolve_source("edit_pairs"))
    random_rows = _read_csv(_resolve_source("random_promoters"))
    originals = [row["orig_sequence"] for row in pair_rows if row.get("orig_sequence")]
    generated = [row["generated_sequence"] for row in pair_rows if row.get("generated_sequence")]
    random_key = next(iter(random_rows[0].keys())) if random_rows else "random_sequence"
    randoms = [row[random_key] for row in random_rows if row.get(random_key)]

    paired = [_levenshtein(a, b) for a, b in zip(originals, generated)]
    natural_random = [_levenshtein(a, b) for a, b in zip(originals[: len(randoms)], randoms)]
    generated_random = [_levenshtein(a, b) for a, b in zip(generated[: len(randoms)], randoms)]
    rows = []
    for label, values in (
        ("original_vs_generated", paired),
        ("original_vs_random", natural_random),
        ("generated_vs_random", generated_random),
    ):
        rows.append(
            {
                "comparison": label,
                "n": len(values),
                "mean_edit_distance": _mean(values),
                "min_edit_distance": min(values) if values else "",
                "max_edit_distance": max(values) if values else "",
            }
        )
    table_path = TABLE_DIR / "edit_distance_summary.csv"
    _write_csv(table_path, rows)

    width, height = 1050, 650
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S7", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "Edit-distance diversity of retained designed promoters", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Distances are recomputed from retained paired design and random-promoter tables.", font=FONT_SMALL, fill=COLORS["muted"])
    x0, y0, plot_w, plot_h = 120, 190, 760, 360
    draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=COLORS["grid"], width=2)
    series = [
        ("original-designed", paired, COLORS["orange"]),
        ("original-random", natural_random, COLORS["blue"]),
        ("generated-random", generated_random, COLORS["green"]),
    ]
    all_values = [value for _, values, _ in series for value in values]
    mn, mx = min(all_values), max(all_values)
    bins = 24
    for label, values, color in series:
        counts = [0] * bins
        for value in values:
            idx = min(bins - 1, int((value - mn) / max(mx - mn, 1e-9) * bins))
            counts[idx] += 1
        max_count = max(counts) or 1
        points = []
        for i, count in enumerate(counts):
            px = x0 + int((i + 0.5) / bins * plot_w)
            py = y0 + plot_h - int(count / max_count * (plot_h - 20))
            points.append((px, py))
        draw.line(points, fill=color, width=3)
    for i, (label, _, color) in enumerate(series):
        y = y0 + 18 + i * 28
        draw.line((x0 + plot_w + 35, y, x0 + plot_w + 75, y), fill=color, width=4)
        draw.text((x0 + plot_w + 86, y - 8), label, font=FONT_TINY, fill=COLORS["ink"])
    draw.text((x0 + 260, y0 + plot_h + 38), "edit distance", font=FONT_SMALL, fill=COLORS["muted"])
    draw.text((x0 - 74, y0 + 150), "relative count", font=FONT_SMALL, fill=COLORS["muted"])
    figure_path = _save_image(image, FIGURE_DIR / "figS7_edit_distance_diversity.png")
    return table_path, figure_path


def build_deepseed_training_summary() -> tuple[Path, Path]:
    rows = _read_csv(_resolve_source("deepseed_training_log"))
    out_path = TABLE_DIR / "deepseed_training_log.csv"
    _write_csv(out_path, rows)
    series = [
        ("train_loss", COLORS["orange"]),
        ("test_loss", COLORS["blue"]),
        ("test_coefs", COLORS["green"]),
    ]
    width, height = 1100, 760
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "Supplementary Figure S8", font=FONT_PANEL, fill=COLORS["ink"])
    draw.text((54, 78), "preGAN expression-constraint retained training trace", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((54, 126), "Curves are redrawn from training_log165_mpra_expr_denselstm.csv.", font=FONT_SMALL, fill=COLORS["muted"])
    x0, y0, plot_w, plot_h = 130, 190, 780, 410
    draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=COLORS["grid"], width=2)
    for label, color in series:
        values = [float(row[label]) for row in rows]
        lo, hi = min(values), max(values)
        pts = []
        for i, value in enumerate(values):
            px = x0 + int(i / max(len(values) - 1, 1) * plot_w)
            py = y0 + plot_h - int((value - lo) / max(hi - lo, 1e-9) * plot_h)
            pts.append((px, py))
        draw.line(pts, fill=color, width=3)
    for i, (label, color) in enumerate(series):
        y = y0 + 24 + i * 28
        draw.line((x0 + plot_w + 35, y, x0 + plot_w + 75, y), fill=color, width=4)
        draw.text((x0 + plot_w + 86, y - 8), label, font=FONT_SMALL, fill=COLORS["ink"])
    draw.text((x0 + 330, y0 + plot_h + 38), "epoch", font=FONT_SMALL, fill=COLORS["muted"])
    figure_path = _save_image(image, FIGURE_DIR / "figS8_deepseed_training_trace.png")
    return out_path, figure_path


def write_manifest(entries: list[dict[str, str]]) -> None:
    _write_csv(OUTPUT_ROOT / "manifest.csv", entries, ["item", "type", "path", "source", "workflow", "note"])


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    jobs = [
        ("expression_heatmap", build_expression_heatmap, "training_set", "transvae", "Balanced 100-promoter tissue-expression heatmap source and figure."),
        ("dnabert_motif_top20", build_motif_summary, "dnabert_motif_summary", "dnabert_find_motifs", "Top retained motifs reconstructed from DNABERT motif_summary.csv."),
        ("kmer_similarity", build_kmer_comparison, "designed_promoters_200", "transvae_verify_kmer", "All/proximal/distal 4-mer comparison using 200 retained designs."),
        ("design_candidate_summary", build_design_candidate_summary, "designed_promoters_200", "transvae_design", "Design-score and modification summary using retained designs with original-sequence baselines."),
        ("prediction_reference", build_prediction_reference, "prediction_reference", "transvae", "Retained predicted/true scalar-expression reference."),
        ("edit_distance", build_edit_distance_summary, "edit_pairs", "transvae_verify_edit_distance", "Edit-distance summary recomputed from retained design pairs."),
        ("pregan_expression_training", build_deepseed_training_summary, "deepseed_training_log", "pregan_expression_constraint_training", "preGAN expression-constraint training trace redrawn from retained CSV log."),
    ]

    for item, builder, source_key, script_name, note in jobs:
        table_path, figure_path = builder()
        entries.append(
            {
                "item": item,
                "type": "source_table",
                "path": _rel(table_path),
                "source": _rel(_resolve_source(source_key)),
                "workflow": script_name,
                "note": note,
            }
        )
        entries.append(
            {
                "item": item,
                "type": "figure",
                "path": _rel(figure_path),
                "source": _rel(table_path),
                "workflow": script_name,
                "note": note,
            }
        )
    write_manifest(entries)
    print(f"Wrote reproducible retained result pack: {_rel(OUTPUT_ROOT)}")
    print(f"Manifest: {_rel(OUTPUT_ROOT / 'manifest.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
