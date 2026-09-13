#!/usr/bin/env python3
"""Standalone fusion detection runner.
Example:
    python fusion_graph.py \
        --bam reads.bam \
        --genedb genes.db \
        --reference genome.fa \
        --output fusion_output
"""

import argparse
import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import types

    pkg_name = os.path.basename(project_dir)
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [project_dir]
        sys.modules[pkg_name] = pkg

    from importlib import import_module

    FusionDetector = import_module(f"{pkg_name}.fusion_detector").FusionDetector
else:
    from .fusion_detector import FusionDetector


logger = logging.getLogger("FusionGraph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run IsoQuant fusion detection directly from a BAM file, a reference FASTA, "
            "and a gffutils gene database."
        )
    )
    parser.add_argument(
        "--bam",
        nargs="+",
        required=True,
        help="One or more BAM files to inspect for fusion candidates.",
    )
    parser.add_argument(
        "--genedb",
        required=True,
        help="Path to a gffutils database (.db) built from a GTF/GFF annotation.",
    )
    parser.add_argument(
        "--reference",
        help="Reference FASTA used for soft-clip realignment and transcript reconstruction.",
    )
    parser.add_argument(
        "--output",
        default="fusion_output",
        help="Directory for fusion TSV reports (default: fusion_output).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum number of supporting reads required for a candidate to remain valid.",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Only emit candidates marked valid by the internal fusion validator.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Minimum confidence threshold used when writing the TSV report.",
    )
    parser.add_argument(
        "--min-al-len-primary",
        type=int,
        default=50,
        help="Minimum aligned length for primary alignments.",
    )
    parser.add_argument(
        "--min-al-len-sa",
        type=int,
        default=40,
        help="Minimum aligned length for SA entries.",
    )
    parser.add_argument(
        "--min-sa-mapq",
        type=int,
        default=10,
        help="Minimum MAPQ required for SA entries.",
    )
    parser.add_argument(
        "--min-softclip-len",
        type=int,
        default=30,
        help="Minimum soft-clip length to trigger soft-clipping realignment.",
    )
    parser.add_argument(
        "--jitter-window",
        type=int,
        default=50,
        help="Local breakpoint jitter window for candidate clustering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose INFO logging.",
    )
    return parser.parse_args(argv)


def _validate_input(path: str, label: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def run_fusion_for_bam(
    bam_path: str,
    gendb_path: str,
    reference_fasta: str | None,
    output_dir: str,
    min_support: int = 2,
    only_valid: bool = False,
    min_confidence: float = 0.30,
    min_al_len_primary: int = 50,
    min_al_len_sa: int = 40,
    min_sa_mapq: int = 10,
    min_softclip_len: int = 30,
    jitter_window: int = 50,
):
    _validate_input(bam_path, "BAM")
    _validate_input(gendb_path, "gffutils database")
    if reference_fasta:
        _validate_input(reference_fasta, "reference FASTA")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing fusion detector for %s", bam_path)
    detector = FusionDetector(bam_path, gendb_path, reference_fasta=reference_fasta)

    detector.detect_fusions(
        min_al_len_primary=min_al_len_primary,
        min_al_len_sa=min_al_len_sa,
        min_sa_mapq=min_sa_mapq,
        min_softclip_len=min_softclip_len,
        jitter_window=jitter_window,
    )

    report_path = out_dir / f"fusion_{Path(bam_path).name}.tsv"
    detector.report(
        output_path=str(report_path),
        min_support=min_support,
        min_confidence=min_confidence,
        only_valid=only_valid,
    )
    logger.info("Fusion report written to %s", report_path)
    return report_path


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.reference:
        logger.warning(
            "No --reference FASTA provided; fusion transcript reconstruction and soft-clip realignment will be limited."
        )

    reports = []
    for bam in args.bam:
        report = run_fusion_for_bam(
            bam_path=bam,
            gendb_path=args.genedb,
            reference_fasta=args.reference,
            output_dir=args.output,
            min_support=args.min_support,
            only_valid=args.only_valid,
            min_confidence=args.min_confidence,
            min_al_len_primary=args.min_al_len_primary,
            min_al_len_sa=args.min_al_len_sa,
            min_sa_mapq=args.min_sa_mapq,
            min_softclip_len=args.min_softclip_len,
            jitter_window=args.jitter_window,
        )
        reports.append(str(report))

    logger.info("Completed fusion detection over %d BAM file(s)", len(reports))
    for report in reports:
        logger.info("Report: %s", report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.exception("Fusion detection failed: %s", exc)
        raise SystemExit(1)
