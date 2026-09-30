from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TISSUES = ("root", "stem", "leaf", "fruit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_checkpoint_predictions(
    input_csv: str | Path,
    output_json: str | Path,
    *,
    expected_records: int | None = None,
) -> dict[str, object]:
    """Audit record-level labels and checkpoint predictions.

    Required columns are ``record_id`` plus ``label_<tissue>`` and
    ``prediction_<tissue>`` for each tissue. The historical validation export
    is also accepted with ``sequence_id`` and
    ``observed_expr_tissue_1``/``predicted_expr_tissue_1`` through tissue 4.
    R-squared definitions are kept separate so a squared Pearson correlation
    is never mislabeled as a model coefficient of determination.
    """
    source = Path(input_csv)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "record_id" in fields and all(
            f"label_{tissue}" in fields and f"prediction_{tissue}" in fields for tissue in TISSUES
        ):
            id_column = "record_id"
            columns = {
                tissue: (f"label_{tissue}", f"prediction_{tissue}") for tissue in TISSUES
            }
        elif "sequence_id" in fields and all(
            f"observed_expr_tissue_{index}" in fields
            and f"predicted_expr_tissue_{index}" in fields
            for index in range(1, 5)
        ):
            id_column = "sequence_id"
            columns = {
                tissue: (f"observed_expr_tissue_{index}", f"predicted_expr_tissue_{index}")
                for index, tissue in enumerate(TISSUES, start=1)
            }
        else:
            raise ValueError(
                "CSV must use record_id/label_<tissue>/prediction_<tissue> or "
                "sequence_id/observed_expr_tissue_<1-4>/predicted_expr_tissue_<1-4>."
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Checkpoint evaluation CSV is empty.")
    identifiers = [row[id_column].strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every checkpoint-evaluation row requires a record_id.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate record_id values found in checkpoint evaluation CSV.")
    if expected_records is not None and len(rows) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(rows)}.")

    tissue_metrics: dict[str, dict[str, float | int]] = {}
    for tissue in TISSUES:
        try:
            label_column, prediction_column = columns[tissue]
            labels = np.asarray([float(row[label_column]) for row in rows], dtype=float)
            predictions = np.asarray(
                [float(row[prediction_column]) for row in rows], dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric label or prediction found for {tissue}.") from error
        if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite label or prediction found for {tissue}.")
        if np.std(labels) == 0 or np.std(predictions) == 0:
            raise ValueError(f"Pearson correlation is undefined for constant {tissue} values.")
        pearson_r = float(np.corrcoef(labels, predictions)[0, 1])
        slope, intercept = np.polyfit(labels, predictions, 1)
        fitted = slope * labels + intercept
        residual_sum = float(np.sum((predictions - fitted) ** 2))
        total_sum = float(np.sum((predictions - predictions.mean()) ** 2))
        regression_r2 = 1.0 - residual_sum / total_sum
        tissue_metrics[tissue] = {
            "n": len(rows),
            "pearson_r": pearson_r,
            "pearson_r_squared": pearson_r**2,
            "linear_regression_r_squared": regression_r2,
            "regression_slope_prediction_on_label": float(slope),
            "regression_intercept_prediction_on_label": float(intercept),
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "input_csv": str(source),
        "input_sha256": _sha256(source),
        "num_records": len(rows),
        "record_identity": f"{id_column} column; uniqueness verified",
        "cohort_role": "checkpoint_evaluation; independence not inferred",
        "metrics": tissue_metrics,
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
