"""Merge observed labels with checkpoint predictions and calculate metrics."""

import csv
import json
from math import fsum
from pathlib import Path

OUT = Path(__file__).resolve().parent
with (OUT / "observed_expression.csv").open(newline="", encoding="utf-8") as handle:
    observed = {row["sequence_id"]: row for row in csv.DictReader(handle)}
with (OUT / "predicted_expression.csv").open(newline="", encoding="utf-8") as handle:
    predicted = {row["sequence_id"]: row for row in csv.DictReader(handle)}

targets = {"expr_tissue_1": "score_root", "expr_tissue_2": "score_stem", "expr_tissue_3": "score_leaf", "expr_tissue_4": "score_fruit"}
merged = []
for sequence_id, source in observed.items():
    if sequence_id not in predicted:
        raise RuntimeError(f"Missing prediction for {sequence_id}")
    row = {"sequence_id": sequence_id, "sequence": source["sequence"]}
    for label, score in targets.items():
        row[f"observed_{label}"] = source[label]
        row[f"predicted_{label}"] = predicted[sequence_id][score]
    merged.append(row)


def pearson(xs, ys):
    mx, my = fsum(xs) / len(xs), fsum(ys) / len(ys)
    numerator = fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = (fsum((x - mx) ** 2 for x in xs) * fsum((y - my) ** 2 for y in ys)) ** 0.5
    return numerator / denominator


metrics = []
for label, score in targets.items():
    xs = [float(row[f"observed_{label}"]) for row in merged]
    ys = [float(row[f"predicted_{label}"]) for row in merged]
    r = pearson(xs, ys)
    metrics.append({"tissue": score.removeprefix("score_"), "observed_column": label, "predicted_column": score, "n": len(xs), "pearson_r": round(r, 6), "r_squared": round(r * r, 6)})

with (OUT / "observed_vs_predicted.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
    writer.writeheader()
    writer.writerows(merged)
with (OUT / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)

manifest = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
manifest.update({"prediction_file": "predicted_expression.csv", "merged_file": "observed_vs_predicted.csv", "metrics_file": "validation_metrics.csv", "score_backend": "tomato_transvae_mlp_checkpoint_scoring", "metrics": metrics})
(OUT / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
