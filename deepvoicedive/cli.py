"""Command-line interface for DeepVoiceDive."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_analyze(args) -> int:
    from .report import analyze_file

    result = analyze_file(
        args.input, args.output_dir, with_egemaps=not args.no_egemaps
    )
    print(f"Analyse abgeschlossen. Artefakte in: {args.output_dir}")
    print(
        f"  - Voice Embedding : {result['embedding_json']} "
        f"({len(result['embedding'])}-dim)"
    )
    print(f"  - Embedding (npy) : {result['embedding_npy']}")
    print(f"  - Visualisierung  : {result['report_png']}")
    if "egemaps_csv" in result:
        print(f"  - eGeMAPS-Report  : {result['egemaps_csv']} (88 Parameter)")
    return 0


def _cmd_compare(args) -> int:
    from .compare import compare_files

    dist, similarity, match = compare_files(args.file_a, args.file_b)
    print(f"Kosinus-Distanz : {dist:.4f}")
    print(f"Ähnlichkeit     : {similarity:.4f}")
    print(f"Voice Match     : {match:.2f} %")
    return 0


def _cmd_matrix(args) -> int:
    from .batch import write_similarity_report

    result = write_similarity_report(args.inputs, args.output_dir)
    labels = result["labels"]
    matrix = result["matrix"]

    print("Voice-Match-Matrix (%):")
    header = "".join(f"{label[:10]:>12}" for label in labels)
    print(f"{'':>12}{header}")
    for i, label in enumerate(labels):
        row = "".join(f"{matrix[i, j]:>12.2f}" for j in range(len(labels)))
        print(f"{label[:10]:>12}{row}")

    print(f"\nTabelle  : {result['csv']}")
    print(f"Heatmap  : {result['png']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepvoicedive",
        description="Deep, mathematical voice analysis.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyse a single WAV recording.")
    analyze.add_argument("input", help="Path to the input WAV file.")
    analyze.add_argument(
        "--output-dir",
        default="results",
        help="Directory for the report artefacts (default: results).",
    )
    analyze.add_argument(
        "--no-egemaps",
        action="store_true",
        help="Skip the openSMILE eGeMAPS report.",
    )
    analyze.set_defaults(func=_cmd_analyze)

    compare = sub.add_parser(
        "compare", help="Compare two WAV recordings biometrically."
    )
    compare.add_argument("file_a", help="First WAV file.")
    compare.add_argument("file_b", help="Second WAV file.")
    compare.set_defaults(func=_cmd_compare)

    matrix = sub.add_parser(
        "matrix",
        help="Compare many WAV recordings at once (all-pairs Voice Match).",
    )
    matrix.add_argument(
        "inputs", nargs="+", help="Two or more WAV files to compare."
    )
    matrix.add_argument(
        "--output-dir",
        default="results",
        help="Directory for the matrix table and heatmap (default: results).",
    )
    matrix.set_defaults(func=_cmd_matrix)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
