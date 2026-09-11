from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tpsgen.io.csv import write_dict_rows
from tpsgen.io.fasta import read_fasta
from tpsgen.pipeline.annotate import run_annotation
from tpsgen.pipeline.annotate_legacy_dnabert import run_legacy_dnabert_motif_annotation
from tpsgen.pipeline.design import run_design
from tpsgen.pipeline.figures import run_figure_export
from tpsgen.pipeline.legacy_figures import run_legacy_figure_export
from tpsgen.pipeline.predict import run_prediction
from tpsgen.pipeline.report import build_report
from tpsgen.visualization.blast_png import render_legacy_blast_figure_pair


RAW_FILE_SPECS = [
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "code" / "transformerresult" / "results1" / "loss_curves" / "loss_history.csv",
        "target": Path("data/raw/transvae/loss_history.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Legacy training-loss table used for paper figure reconstruction.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "results" / "designed_promoters.csv",
        "target": Path("data/raw/transvae/designed_promoters.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Legacy designed promoter table used for k-mer and semantic-space views.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "results" / "designed_promoters200.csv",
        "target": Path("data/raw/transvae/designed_promoters200.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Retained 200-promoter design table used for supplementary k-mer and design-summary reproduction.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "bianjijuli" / "designed_promoters200.csv",
        "target": Path("data/raw/transvae/designed_promoters200_pairs.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Paired original/generated promoter table used for edit-distance reproduction.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "results" / "generated_prediction_results.csv",
        "target": Path("data/raw/transvae/generated_prediction_results.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Legacy prediction verification table.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "bianjijuli" / "mutated_file.csv",
        "target": Path("data/raw/transvae/mutated_file.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Edited promoter pairs used for diversity analysis.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "random_promoters_200.csv",
        "target": Path("data/raw/transvae/random_promoters_200.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Random promoter background for diversity analysis.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "vaedata" / "training_set.csv",
        "target": Path("data/raw/transvae/training_set.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Tomato promoter training set used as natural-sequence reference.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "blast_result.txt",
        "target": Path("data/raw/transvae/blast_result.txt"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Raw BLAST tabular results used to rebuild legacy histogram figures.",
    },
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "data" / "sequences_with_id.csv",
        "target": Path("data/raw/transvae/sequences_with_id.csv"),
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Sequence identifiers used to map BLAST hits back to sequence groups.",
    },
    {
        "source": WORKSPACE_ROOT / "deepseed" / "Predictor" / "results1" / "training_log165_mpra_expr_denselstm.csv",
        "target": Path("data/raw/pregan_expression/training_log165_mpra_expr_denselstm.csv"),
        "stage": "raw",
        "origin": "deepseed",
        "note": "Legacy DenseLSTM training log.",
    },
    {
        "source": WORKSPACE_ROOT / "deepseed" / "Predictor" / "results1" / "scatter_fig" / "scatter_165_mpra_expr_denselstm.png",
        "target": Path("data/raw/pregan_expression/scatter_165_mpra_expr_denselstm.png"),
        "stage": "raw",
        "origin": "deepseed",
        "note": "Original deepseed scatter image.",
    },
    {
        "source": WORKSPACE_ROOT / "DNABERT" / "examples" / "sample_data" / "vision" / "dev.tsv",
        "target": Path("data/raw/dnabert/dev.tsv"),
        "stage": "raw",
        "origin": "DNABERT",
        "note": "DNABERT evaluation sequences for attention-based motif processing.",
    },
    {
        "source": WORKSPACE_ROOT / "DNABERT" / "examples" / "result" / "6" / "atten.npy",
        "target": Path("data/raw/dnabert/atten.npy"),
        "stage": "raw",
        "origin": "DNABERT",
        "note": "DNABERT attention scores aligned with dev.tsv.",
    },
]

RAW_DIR_SPECS = [
    {
        "source": WORKSPACE_ROOT / "MpraVAE" / "results" / "bianjijuli",
        "target": Path("data/raw/transvae/diversity"),
        "pattern": "*.png",
        "stage": "raw",
        "origin": "MpraVAE",
        "note": "Legacy diversity PNG assets.",
    },
    {
        "source": WORKSPACE_ROOT / "DNABERT" / "motif" / "result" / "6-2",
        "target": Path("data/raw/dnabert/tfbs_assets"),
        "pattern": "TFBS_*.png",
        "stage": "raw",
        "origin": "DNABERT",
        "note": "DNABERT TFBS logo assets, primary source directory.",
    },
    {
        "source": WORKSPACE_ROOT / "DNABERT" / "motif" / "result" / "6-3",
        "target": Path("data/raw/dnabert/tfbs_assets"),
        "pattern": "TFBS_*.png",
        "stage": "raw",
        "origin": "DNABERT",
        "note": "DNABERT TFBS logo assets, fallback source directory.",
    },
]

EXTERNAL_RESOURCE_SPECS = [
    {
        "resource": "Tomato reference genome",
        "required_for": "optional promoter extraction and upstream sequence reconstruction",
        "bundled": "no",
        "default_location": "external/tomato_reference_genome.fna",
        "availability_note": "Obtain from the cited genome resource or a project archive; not required for normal package use.",
    },
    {
        "resource": "Synthetic promoter HDF5 corpus",
        "required_for": "full MpraVAE corpus inspection or retraining",
        "bundled": "no",
        "default_location": "external/mpravae_synthetic_sequences.h5",
        "availability_note": "Not required for package-native annotate, predict, design, or report commands.",
    },
    {
        "resource": "BLAST database shards",
        "required_for": "optional generated-sequence similarity checks",
        "bundled": "no",
        "default_location": "external/blast_database/",
        "availability_note": "Not required for normal tool use.",
    },
    {
        "resource": "DNABERT fine-tuned checkpoint",
        "required_for": "rerunning DNABERT inference before motif post-processing",
        "bundled": "no",
        "default_location": "external/dnabert_checkpoint/",
        "availability_note": "The bundled annotate-dnabert command consumes precomputed attention resources and does not require this checkpoint.",
    },
]


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _display_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return _rel(path)
        except ValueError:
            return os.path.relpath(path, REPO_ROOT)
    return str(path)


def _record_inventory(rows: list[dict[str, str]], stage: str, rel_path: str, source: str, note: str) -> None:
    rows.append(
        {
            "stage": stage,
            "path": rel_path,
            "source": source,
            "note": note,
        }
    )


def _normalize_json_paths(path: Path) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        normalized: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, str) and (value.startswith("/") or value.startswith("data/") or value.startswith("outputs/")):
                normalized[key] = _display_path(value)
            else:
                normalized[key] = value
        path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def sync_raw_and_processed_data() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []

    for spec in RAW_FILE_SPECS:
        source = spec["source"]
        destination = REPO_ROOT / spec["target"]
        if not source.exists():
            continue
        _copy_file(source, destination)
        _record_inventory(inventory, spec["stage"], _rel(destination), f"{spec['origin']} retained source", spec["note"])

    tfbs_seen: set[str] = set()
    for spec in RAW_DIR_SPECS:
        source_dir = spec["source"]
        if not source_dir.exists():
            continue
        target_dir = REPO_ROOT / spec["target"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_dir.glob(spec["pattern"])):
            destination = target_dir / source_path.name
            if destination.name in tfbs_seen and "tfbs_assets" in str(target_dir):
                continue
            shutil.copy2(source_path, destination)
            if "tfbs_assets" in str(target_dir):
                tfbs_seen.add(destination.name)
            _record_inventory(inventory, spec["stage"], _rel(destination), f"{spec['origin']} retained source", spec["note"])

    blast_sequence_ids = REPO_ROOT / "data" / "raw" / "transvae" / "sequences_with_id.csv"
    blast_results = REPO_ROOT / "data" / "raw" / "transvae" / "blast_result.txt"
    if blast_sequence_ids.exists() and blast_results.exists():
        blast_paths = render_legacy_blast_figure_pair(
            sequence_id_csv=blast_sequence_ids,
            blast_result_txt=blast_results,
            output_dir=REPO_ROOT / "data" / "raw" / "transvae" / "blast",
        )
        for path in blast_paths:
            _record_inventory(
                inventory,
                "raw",
                _rel(path),
                "data/raw/transvae/blast_result.txt + data/raw/transvae/sequences_with_id.csv",
                "Curated English-only rebuild of legacy BLAST histogram figure.",
            )

    dnabert_processed_dir = Path("data/processed/dnabert_legacy")
    _reset_dir(REPO_ROOT / dnabert_processed_dir)
    dnabert_metadata = run_legacy_dnabert_motif_annotation(
        dev_tsv_path=Path("data/raw/dnabert/dev.tsv"),
        attention_scores_path=Path("data/raw/dnabert/atten.npy"),
        output_dir=dnabert_processed_dir,
    )
    for rel_name, note in (
        ("motif_summary.csv", "Processed motif ranking from migrated DNABERT adapter."),
        ("processed_sequences.csv", "Per-sequence processed motif annotations from migrated DNABERT adapter."),
        ("run_metadata.json", "Run metadata for migrated DNABERT processing."),
    ):
        path = REPO_ROOT / dnabert_processed_dir / rel_name
        if path.name == "run_metadata.json":
            _normalize_json_paths(path)
        _record_inventory(
            inventory,
            "processed",
            _rel(path),
            "data/raw/dnabert/dev.tsv + data/raw/dnabert/atten.npy",
            note,
        )
    return inventory


def generate_demo_results() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    demo_dir = REPO_ROOT / "data" / "results" / "demo"
    _reset_dir(demo_dir)

    records = read_fasta(REPO_ROOT / "examples" / "demo_input.fasta")

    annotate_rows = [hit.to_dict() for hit in run_annotation(records)]
    if not annotate_rows:
        annotate_rows = [
            {"sequence_id": record.sequence_id, "motif": "none", "start": -1, "end": -1, "score": 0.0}
            for record in records
        ]
    annotate_path = demo_dir / "demo_annotate.csv"
    write_dict_rows(annotate_rows, annotate_path)
    _record_inventory(inventory, "results", _rel(annotate_path), "generated_from_examples", "Deterministic demo motif annotation output.")

    predict_rows = [item.to_dict() for item in run_prediction(records)]
    predict_path = demo_dir / "demo_predict.csv"
    write_dict_rows(predict_rows, predict_path)
    _record_inventory(inventory, "results", _rel(predict_path), "generated_from_examples", "Deterministic demo prediction output.")

    design_rows = [item.to_dict() for item in run_design(records, target_tissue="fruit", candidates=3, seed=42)]
    design_path = demo_dir / "demo_design.csv"
    write_dict_rows(design_rows, design_path)
    _record_inventory(inventory, "results", _rel(design_path), "generated_from_examples", "Deterministic demo design output.")

    report_path = demo_dir / "demo_report.json"
    build_report(design_path, report_path)
    _record_inventory(inventory, "results", _rel(report_path), "generated_from_examples", "Deterministic demo report JSON.")

    return inventory


def generate_repository_figures() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []

    figure_jobs = [
        (Path("data/results/demo/demo_predict.csv"), Path("data/results/figures_predict")),
        (Path("data/results/demo/demo_design.csv"), Path("data/results/figures_design")),
        (Path("data/processed/dnabert_legacy/motif_summary.csv"), Path("data/results/figures_motif")),
    ]

    for input_path, output_dir in figure_jobs:
        if not input_path.exists():
            continue
        _reset_dir(REPO_ROOT / output_dir)
        manifest = run_figure_export(input_path, output_dir)
        _normalize_json_paths(REPO_ROOT / output_dir / "manifest.json")
        generated_files = manifest.get("generated", manifest.get("files", []))
        for generated in generated_files:
            path = REPO_ROOT / output_dir / generated
            _record_inventory(inventory, "results", _rel(path), _display_path(input_path), "Generated manuscript-style SVG figure.")
        _record_inventory(inventory, "results", _rel(REPO_ROOT / output_dir / "manifest.json"), _display_path(input_path), "Figure export manifest.")

    legacy_output_dir = Path("data/results/legacy_figures_bundle")
    _reset_dir(REPO_ROOT / legacy_output_dir)
    manifest = run_legacy_figure_export(
        output_dir=legacy_output_dir,
        mpravae_loss_history=Path("data/raw/transvae/loss_history.csv"),
        mpravae_designed_promoters=Path("data/raw/transvae/designed_promoters.csv"),
        mpravae_prediction_results=Path("data/raw/transvae/generated_prediction_results.csv"),
        deepseed_training_log=Path("data/raw/pregan_expression/training_log165_mpra_expr_denselstm.csv"),
        mpravae_mutated_file=Path("data/raw/transvae/mutated_file.csv"),
        mpravae_random_promoters=Path("data/raw/transvae/random_promoters_200.csv"),
        mpravae_training_set=Path("data/raw/transvae/training_set.csv"),
        dnabert_motif_summary=Path("data/processed/dnabert_legacy/motif_summary.csv"),
        dnabert_tfbs_dir=Path("data/raw/dnabert/tfbs_assets"),
        deepseed_scatter_png=Path("data/raw/pregan_expression/scatter_165_mpra_expr_denselstm.png"),
        mpravae_blast_dir=Path("data/raw/transvae/blast"),
        mpravae_diversity_dir=Path("data/raw/transvae/diversity"),
    )
    _normalize_json_paths(REPO_ROOT / legacy_output_dir / "manifest.json")
    for generated in manifest["generated"]:
        path = REPO_ROOT / legacy_output_dir / generated
        _record_inventory(inventory, "results", _rel(path), "generated_from_data/raw_and_processed", "Generated or bundled legacy figure asset.")
    _record_inventory(inventory, "results", _rel(REPO_ROOT / legacy_output_dir / "manifest.json"), "generated_from_data/raw_and_processed", "Legacy figure bundle manifest.")

    return inventory


def write_external_manifest() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    target = REPO_ROOT / "data" / "external" / "external_resources.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["resource", "required_for", "bundled", "default_location", "availability_note"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(EXTERNAL_RESOURCE_SPECS)
    _record_inventory(
        inventory,
        "external",
        _rel(target),
        "external_resource_manifest",
        "Optional large raw/model artifacts intentionally not copied into git.",
    )
    return inventory


def write_inventory(all_rows: list[dict[str, str]]) -> None:
    target = REPO_ROOT / "data" / "inventory.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "path", "source", "note"], delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row["stage"], row["path"])))


def write_summary(all_rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    target = REPO_ROOT / "data" / "summary.json"
    target.write_text(json.dumps({"counts": counts, "total_files": len(all_rows)}, indent=2), encoding="utf-8")


def main() -> int:
    all_rows: list[dict[str, str]] = []
    all_rows.extend(sync_raw_and_processed_data())
    all_rows.extend(generate_demo_results())
    all_rows.extend(generate_repository_figures())
    all_rows.extend(write_external_manifest())
    write_inventory(all_rows)
    write_summary(all_rows)
    print(json.dumps({"copied_or_generated": len(all_rows), "inventory": "data/inventory.tsv"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
