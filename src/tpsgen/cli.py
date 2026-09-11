from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tpsgen.config import AppConfig
from tpsgen.io.csv import write_dict_rows
from tpsgen.io.fasta import read_fasta
from tpsgen.resources import find_example, find_model


def _missing_optional_dependency(parser: argparse.ArgumentParser, command: str, error: ModuleNotFoundError) -> None:
    if error.name == "torch":
        parser.error(
            f"{command} requires the optional PyTorch model dependencies. "
            "Install them with 'pip install tpsgen[models]' "
            "or use package-native commands such as 'predict' and 'design'."
        )
    raise error


def _require_transvae_length(records, parser: argparse.ArgumentParser) -> None:
    canonical_length = AppConfig().canonical_length
    invalid = [record for record in records if len(record.sequence) != canonical_length]
    if invalid:
        details = ", ".join(
            f"{record.sequence_id}={len(record.sequence)} bp" for record in invalid[:5]
        )
        if len(invalid) > 5:
            details += f", ... ({len(invalid)} invalid records total)"
        parser.error(
            "TransVAE commands require unambiguous "
            f"{canonical_length}-bp promoter sequences; found {details}."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpsgen",
        description="Integrated motif-aware tomato promoter analysis and design workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="Annotate motif hits in promoter FASTA input.")
    annotate.add_argument("--input", required=True, help="Input FASTA file.")
    annotate.add_argument("--output", required=True, help="Output CSV path.")

    annotate_dnabert = subparsers.add_parser(
        "annotate-dnabert",
        help="Run the DNABERT attention-to-motif post-processing workflow.",
    )
    annotate_dnabert.add_argument("--dev-tsv", required=True, help="DNABERT dev.tsv file with k-mer sequences and labels.")
    annotate_dnabert.add_argument("--atten-npy", required=True, help="Attention score array exported by DNABERT.")
    annotate_dnabert.add_argument("--output-dir", required=True, help="Output directory for motif summary and labeled sequences.")
    annotate_dnabert.add_argument("--window-size", type=int, default=24)
    annotate_dnabert.add_argument("--min-len", type=int, default=5)
    annotate_dnabert.add_argument("--pval-cutoff", type=float, default=0.005)
    annotate_dnabert.add_argument("--min-n-motif", type=int, default=3)

    validate_dnabert = subparsers.add_parser(
        "validate-dnabert",
        help="Validate row and sequence compatibility of DNABERT result resources.",
    )
    validate_dnabert.add_argument("--dev-tsv", required=True)
    validate_dnabert.add_argument("--atten-npy", required=True)
    validate_dnabert.add_argument("--pred-npy", required=False)

    validate_models = subparsers.add_parser(
        "validate-models",
        help="Audit bundled model resources and report which runtime routes are available.",
    )
    validate_models.add_argument(
        "--json-output", required=False, help="Optional JSON file for the audit report."
    )

    predict_dnabert = subparsers.add_parser(
        "predict-dnabert", help="Run the bundled tomato DNABERT 6-mer inference route."
    )
    predict_dnabert.add_argument("--input", required=True, help="Input 165-bp tomato promoter FASTA.")
    predict_dnabert.add_argument("--model", required=False, help="DNABERT model directory.")
    predict_dnabert.add_argument("--output", required=True, help="Output CSV path.")

    pregan_generate = subparsers.add_parser(
        "pregan-generate", help="Generate masked promoter candidates with a trained preGAN generator."
    )
    pregan_generate.add_argument("--input", required=True, help="FASTA containing 165-bp templates; use M at mutable positions.")
    pregan_generate.add_argument("--checkpoint", required=True)
    pregan_generate.add_argument("--candidates", type=int, default=5)
    pregan_generate.add_argument("--seed", type=int, default=42)
    pregan_generate.add_argument("--output", required=True, help="Output CSV path.")
    pregan_generate.add_argument("--fasta-output", help="Optional FASTA path for generated candidates.")

    run_pregan = subparsers.add_parser(
        "run-pregan", help="Generate masked candidates with preGAN and score them with TransVAE."
    )
    run_pregan.add_argument("--input", required=True, help="165-bp masked tomato template FASTA.")
    run_pregan.add_argument("--checkpoint", required=True)
    run_pregan.add_argument("--candidates", type=int, default=5)
    run_pregan.add_argument("--target", choices=["root", "stem", "leaf", "fruit"], default="fruit")
    run_pregan.add_argument("--seed", type=int, default=42)
    run_pregan.add_argument("--output", required=True, help="Output directory.")

    predict = subparsers.add_parser(
        "predict",
        help="Generate deterministic package-native tissue-associated scores.",
    )
    predict.add_argument("--input", required=True, help="Input FASTA file.")
    predict.add_argument("--output", required=True, help="Output CSV path.")

    predict_transvae = subparsers.add_parser(
        "predict-transvae",
        help="Run the bundled TransVAE tomato four-tissue scorer.",
    )
    predict_transvae.add_argument("--input", required=True, help="Input FASTA file.")
    predict_transvae.add_argument("--output", required=True, help="Output CSV path.")
    predict_transvae.add_argument("--checkpoint", required=False, help="Optional path to a TransVAE tomato checkpoint.")

    design = subparsers.add_parser(
        "design",
        help="Generate deterministic package-native motif-aware candidate promoters.",
    )
    design.add_argument("--input", required=True, help="Input FASTA file.")
    design.add_argument("--target", required=True, choices=["root", "stem", "leaf", "fruit"])
    design.add_argument("--candidates", type=int, default=5, help="Number of ranked candidates to emit per input sequence.")
    design.add_argument("--seed", type=int, default=42, help="Random seed for deterministic design.")
    design.add_argument("--output", required=True, help="Output CSV path.")

    run = subparsers.add_parser(
        "run",
        help="Run motif annotation, candidate design, scoring and reporting in one workflow.",
    )
    run.add_argument("--input", required=True, help="Input FASTA file with one or more promoter records.")
    run.add_argument("--target", required=True, choices=["root", "stem", "leaf", "fruit"])
    run.add_argument("--candidates", type=int, default=5, help="Number of candidates per input sequence.")
    run.add_argument("--seed", type=int, default=42, help="Seed for reproducible candidate generation.")
    run.add_argument("--output", required=True, help="Output directory for all workflow results.")
    run.add_argument("--scoring-backend", choices=["native", "transvae"], default="native")
    run.add_argument("--checkpoint", required=False, help="Optional TransVAE checkpoint for --scoring-backend transvae.")
    run.add_argument("--design-backend", choices=["native", "pregan"], default="native")
    run.add_argument("--motif-backend", choices=["native", "dnabert"], default="native")

    report = subparsers.add_parser("report", help="Build a compact JSON report from a design CSV.")
    report.add_argument("--input", required=True, help="Input design CSV.")
    report.add_argument("--output", required=True, help="Output JSON.")

    figures = subparsers.add_parser("figures", help="Export publication-style SVG figures from result CSV tables.")
    figures.add_argument("--input", required=True, help="Input CSV from predict, design, or motif summary.")
    figures.add_argument("--output-dir", required=True, help="Output directory for SVG figure files.")
    figures.add_argument("--top-n", type=int, default=15, help="Maximum number of motif rows to render for motif summary figures.")

    legacy_figures = subparsers.add_parser(
        "model-figures",
        help="Export model-resource SVG figures from TransVAE, DNABERT and preGAN-related result files.",
    )
    legacy_figures.add_argument("--output-dir", required=True, help="Output directory for legacy SVG figure files.")
    legacy_figures.add_argument("--transvae-loss-history", required=False, help="Optional TransVAE loss_history.csv path.")
    legacy_figures.add_argument("--transvae-designed-promoters", required=False, help="Optional designed_promoters.csv path.")
    legacy_figures.add_argument("--transvae-prediction-results", required=False, help="Optional generated_prediction_results.csv path.")
    legacy_figures.add_argument("--pregan-expression-training-log", required=False, help="Optional preGAN expression-constraint training log CSV path.")
    legacy_figures.add_argument("--transvae-mutated-file", required=False, help="Optional TransVAE mutated_file.csv path for edit-distance diversity.")
    legacy_figures.add_argument("--transvae-random-promoters", required=False, help="Optional TransVAE random_promoters_200.csv path for edit-distance diversity.")
    legacy_figures.add_argument("--transvae-training-set", required=False, help="Optional TransVAE training_set.csv path for semantic-space reconstruction.")
    legacy_figures.add_argument("--dnabert-motif-summary", required=False, help="Optional DNABERT motif_summary.csv path for collecting TFBS assets.")
    legacy_figures.add_argument("--dnabert-tfbs-dir", required=False, help="Optional DNABERT TFBS PNG directory.")
    legacy_figures.add_argument("--pregan-expression-scatter-png", required=False, help="Optional preGAN expression-constraint scatter PNG path to copy into the bundle.")
    legacy_figures.add_argument("--transvae-blast-dir", required=False, help="Optional TransVAE blast PNG directory to collect.")
    legacy_figures.add_argument("--transvae-diversity-dir", required=False, help="Optional TransVAE diversity PNG directory to collect.")

    validate = subparsers.add_parser("validate-input", help="Validate FASTA content and print a small summary.")
    validate.add_argument("--input", required=True, help="Input FASTA file.")

    copy_example = subparsers.add_parser("copy-example", help="Copy the bundled demonstration FASTA to a local path.")
    copy_example.add_argument("--output", required=True, help="Destination FASTA path.")

    extract = subparsers.add_parser(
        "extract-promoters",
        help="Extract complete tomato promoter windows from genome FASTA and GFF3 annotation.",
    )
    extract.add_argument("--genome", required=True, help="Tomato reference genome FASTA.")
    extract.add_argument("--annotation", required=True, help="Tomato GFF3 annotation file.")
    extract.add_argument("--output", required=True, help="Output promoter FASTA path.")
    extract.add_argument("--window-size", type=int, default=165, help="Promoter window length in bp.")
    extract.add_argument(
        "--positive-strand-only",
        action="store_true",
        help="Retain only positive-strand gene records, matching the retained training convention.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "candidates", 1) < 1:
        parser.error("--candidates must be at least 1.")

    if args.command == "copy-example":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(find_example(), output)
        print(json.dumps({"example": str(output)}, indent=2))
        return 0

    if args.command == "extract-promoters":
        if args.window_size != AppConfig().canonical_length:
            parser.error(
                "The released tomato model workflow requires a 165-bp promoter window; "
                "use --window-size 165."
            )
        from tpsgen.preprocessing.extract_promoters import extract_gene_promoters

        summary = extract_gene_promoters(
            args.genome,
            args.annotation,
            args.output,
            window_size=args.window_size,
            include_negative_strand=not args.positive_strand_only,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate-input":
        records = read_fasta(args.input)
        summary = {
            "num_records": len(records),
            "min_length": min(len(record.sequence) for record in records),
            "max_length": max(len(record.sequence) for record in records),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate-dnabert":
        from tpsgen.legacy.dnabert_motif import validate_dnabert_resources

        try:
            summary = validate_dnabert_resources(
                args.dev_tsv, args.atten_npy, prediction_path=args.pred_npy
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate-models":
        from tpsgen.resources import repository_root

        repo_root = repository_root()
        model_root = repo_root / "models"
        transvae = model_root / "transvae" / "best_val_corr_model.pth"
        dnabert = model_root / "dnabert" / "pytorch_model.bin"
        pregan_generator = model_root / "pregan" / "generator_checkpoint.pt"
        pregan = model_root / "pregan_expression" / "165_mpra_expr_denselstm.pth"
        report = {
            "transvae": {
                "checkpoint": str(transvae),
                "available_for_runtime_scoring": transvae.exists(),
                "route": "predict-transvae" if transvae.exists() else None,
            },
            "dnabert": {
                "fine_tuned_checkpoint_present": dnabert.exists(),
                "precomputed_attention_present": (
                    repo_root / "data" / "raw" / "dnabert" / "atten.npy"
                ).exists(),
                "available_for_arbitrary_fasta": dnabert.exists(),
                "route": "predict-dnabert" if dnabert.exists() else None,
                "reason": "The retained attention array is separate historical data; predict-dnabert runs fresh inference when the bundled checkpoint is present.",
            },
            "pregan": {
                "generator_checkpoint_present": pregan_generator.exists(),
                "expression_constraint_checkpoint_present": pregan.exists(),
                "available_for_arbitrary_fasta_generation": pregan_generator.exists(),
                "route": "pregan-generate" if pregan_generator.exists() else None,
                "checkpoint": str(pregan_generator),
                "training_records": 5100 if pregan_generator.exists() else None,
                "training_steps": 1000 if pregan_generator.exists() else None,
                "reason": "Checkpoint is available for computational generation; external biological validation remains required.",
            },
        }
        if args.json_output:
            Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "predict-dnabert":
        from tpsgen.models.dnabert_inference import DNABERTTomatoAdapter

        records = read_fasta(args.input)
        model_dir = args.model or str(find_model("dnabert/pytorch_model.bin").parent)
        try:
            rows = DNABERTTomatoAdapter(model_dir).predict(records)
        except (FileNotFoundError, ValueError, ModuleNotFoundError) as error:
            parser.error(str(error))
        write_dict_rows(rows, args.output)
        return 0

    if args.command == "pregan-generate":
        from tpsgen.training.pregan import generate_pregan_candidates

        records = read_fasta(args.input)
        templates = [record.sequence for record in records]
        try:
            rows = generate_pregan_candidates(templates, args.checkpoint, args.candidates, args.seed)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        for row, record in zip(rows, [record for record in records for _ in range(args.candidates)]):
            row["sequence_id"] = record.sequence_id
            row["candidate_id"] = f"{record.sequence_id}__pregan_{row['candidate_rank']}"
        write_dict_rows(rows, args.output)
        if args.fasta_output:
            from tpsgen.io.schema import SequenceRecord
            from tpsgen.io.fasta import write_fasta

            write_fasta(
                [SequenceRecord(str(row["candidate_id"]), str(row["sequence"])) for row in rows],
                args.fasta_output,
            )
        return 0

    if args.command == "run-pregan":
        from tpsgen.pipeline.pregan_workflow import run_pregan_workflow

        records = read_fasta(args.input)
        try:
            manifest = run_pregan_workflow(records, args.checkpoint, args.candidates, args.seed, args.output, args.target)
        except (FileNotFoundError, ValueError, RuntimeError, ModuleNotFoundError) as error:
            parser.error(str(error))
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "annotate-dnabert":
        from tpsgen.pipeline.annotate_legacy_dnabert import (
            run_legacy_dnabert_motif_annotation,
        )

        metadata = run_legacy_dnabert_motif_annotation(
            dev_tsv_path=args.dev_tsv,
            attention_scores_path=args.atten_npy,
            output_dir=args.output_dir,
            window_size=args.window_size,
            min_len=args.min_len,
            p_value_cutoff=args.pval_cutoff,
            min_n_motif=args.min_n_motif,
        )
        print(json.dumps(metadata, indent=2))
        return 0

    if args.command in {
        "annotate",
        "predict",
        "predict-transvae",
        "design",
        "run",
    }:
        records = read_fasta(args.input)

    if args.command == "annotate":
        from tpsgen.pipeline.annotate import run_annotation

        hits = run_annotation(records)
        rows = [hit.to_dict() for hit in hits]
        if not rows:
            rows = [
                {
                    "sequence_id": record.sequence_id,
                    "motif": "none",
                    "start": -1,
                    "end": -1,
                    "score": 0.0,
                }
                for record in records
            ]
        for row in rows:
            row["backend"] = "package_native_exact_motif_annotation"
            row["evidence_type"] = "exact_configured_motif_hit"
        write_dict_rows(rows, args.output)
        return 0

    if args.command == "predict":
        from tpsgen.pipeline.predict import run_prediction

        predictions = run_prediction(records)
        rows = [item.to_dict() for item in predictions]
        for row in rows:
            row["backend"] = "package_native_deterministic_tissue_associated_scoring"
            row["score_type"] = "tissue_associated_heuristic_score"
        write_dict_rows(rows, args.output)
        return 0

    if args.command == "predict-transvae":
        try:
            from tpsgen.pipeline.predict_transvae import (
                run_transvae_prediction,
            )
        except ModuleNotFoundError as error:
            _missing_optional_dependency(parser, "predict-transvae", error)

        _require_transvae_length(records, parser)
        predictions = run_transvae_prediction(records, checkpoint_path=args.checkpoint)
        rows = [item.to_dict() for item in predictions]
        for row in rows:
            row["backend"] = "tomato_transvae_mlp_checkpoint_scoring"
            row["score_type"] = "transvae_model_score"
        write_dict_rows(rows, args.output)
        return 0

    if args.command == "design":
        from tpsgen.pipeline.design import run_design

        designs = run_design(records, target_tissue=args.target, candidates=args.candidates, seed=args.seed)
        rows = [item.to_dict() for item in designs]
        for row in rows:
            row["backend"] = "package_native_motif_preserving_design"
            row["score_type"] = "tissue_associated_heuristic_score"
        write_dict_rows(rows, args.output)
        return 0

    if args.command == "run":
        from tpsgen.pipeline.integrated import run_integrated_workflow

        if args.design_backend == "pregan":
            if not args.checkpoint:
                parser.error("--design-backend pregan requires --checkpoint for the generator.")
            from tpsgen.pipeline.pregan_workflow import run_pregan_workflow

            try:
                manifest = run_pregan_workflow(
                    records, args.checkpoint, args.candidates, args.seed, args.output, args.target,
                    args.motif_backend
                )
            except (FileNotFoundError, ValueError, RuntimeError, ModuleNotFoundError) as error:
                parser.error(str(error))
            print(json.dumps(manifest, indent=2))
            return 0

        _require_transvae_length(records, parser)
        manifest = run_integrated_workflow(
            records,
            target_tissue=args.target,
            candidates=args.candidates,
            seed=args.seed,
            output_dir=args.output,
            scoring_backend=args.scoring_backend,
            checkpoint_path=args.checkpoint,
            motif_backend=args.motif_backend,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "report":
        from tpsgen.pipeline.report import build_report

        report = build_report(args.input, args.output)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "figures":
        from tpsgen.pipeline.figures import run_figure_export

        manifest = run_figure_export(args.input, args.output_dir, top_n=args.top_n)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "model-figures":
        from tpsgen.pipeline.legacy_figures import run_legacy_figure_export

        manifest = run_legacy_figure_export(
            output_dir=args.output_dir,
            transvae_loss_history=args.transvae_loss_history,
            transvae_designed_promoters=args.transvae_designed_promoters,
            transvae_prediction_results=args.transvae_prediction_results,
            pregan_expression_training_log=args.pregan_expression_training_log,
            transvae_mutated_file=args.transvae_mutated_file,
            transvae_random_promoters=args.transvae_random_promoters,
            transvae_training_set=args.transvae_training_set,
            dnabert_motif_summary=args.dnabert_motif_summary,
            dnabert_tfbs_dir=args.dnabert_tfbs_dir,
            pregan_expression_scatter_png=args.pregan_expression_scatter_png,
            transvae_blast_dir=args.transvae_blast_dir,
            transvae_diversity_dir=args.transvae_diversity_dir,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
