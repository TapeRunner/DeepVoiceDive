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


def _resolve_method(method: str) -> str:
    """Return ``method``, falling back from 'neural' to 'mfcc' if unavailable."""
    if method != "neural":
        return method
    try:
        from .neural import _get_encoder

        _get_encoder()
        return "neural"
    except Exception as exc:  # noqa: BLE001
        print(
            f"Hinweis: Neuronales Modell nicht verfügbar ({exc}). "
            "Fallback auf die MFCC-Methode.",
            file=sys.stderr,
        )
        return "mfcc"


def _cmd_profile(args) -> int:
    from .profile import build_profile, save_profile

    method = _resolve_method(args.method)
    profile = build_profile(args.clips, method=method)
    save_profile(profile, args.output)

    print(
        f"Stimmprofil erstellt ({method}, {len(profile['embedding'])}-dim) "
        f"aus {len(args.clips)} Clip(s)."
    )
    if profile["f0_median"] is not None:
        print(
            f"  Tonlage : {profile['f0_low']:.0f}-{profile['f0_high']:.0f} Hz "
            f"(Median {profile['f0_median']:.0f} Hz)"
        )
    print(f"  Gespeichert: {args.output}")
    return 0


def _cmd_screen(args) -> int:
    from .profile import build_profile, load_profile
    from .screen import write_screening_report

    if args.profile:
        profile = load_profile(args.profile)
        method = profile.get("method", "mfcc")
        if method == "neural" and _resolve_method("neural") != "neural":
            print(
                "Fehler: Profil wurde mit dem neuronalen Modell erstellt, das hier "
                'nicht verfügbar ist. Bitte ".[neural]" installieren oder ein '
                "MFCC-Profil verwenden.",
                file=sys.stderr,
            )
            return 1
    elif args.reference_clips:
        method = _resolve_method(args.method)
        profile = build_profile(args.reference_clips, method=method)
    else:
        print(
            "Fehler: Bitte --profile ODER --reference-clips angeben.",
            file=sys.stderr,
        )
        return 1

    result = write_screening_report(
        profile, args.stems, args.output_dir,
        method=method, threshold=args.threshold,
    )

    print(f"Screening ({method}) - sortiert nach Eignung:\n")
    print(f"{'Stem':<24}{'Match':>8}{'Tonlage':>9}{'Eignung':>9}")
    for r in result["results"]:
        mark = "✅" if r["verdict"] else "❌"
        print(
            f"{r['name'][:24]:<24}{r['match']:>7.0f}%{r['pitch']:>8.0f}%"
            f"{r['suitability']:>8.0f}%  {mark}"
        )
    print(f"\nTabelle  : {result['csv']}")
    print(f"Diagramm : {result['png']}")
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

    profile = sub.add_parser(
        "profile",
        help="Build a reusable voice profile from several clips of your voice.",
    )
    profile.add_argument(
        "clips", nargs="+", help="One or more clean clips of your own voice."
    )
    profile.add_argument(
        "-o", "--output", default="voice_profile.json",
        help="Where to save the profile JSON (default: voice_profile.json).",
    )
    profile.add_argument(
        "--method", choices=["neural", "mfcc"], default="neural",
        help="Embedding method (default: neural, falls back to mfcc).",
    )
    profile.set_defaults(func=_cmd_profile)

    screen = sub.add_parser(
        "screen",
        help="Screen candidate vocal stems against your voice for swap suitability.",
    )
    screen.add_argument(
        "stems", nargs="+", help="Candidate vocal stems (WAV) to screen."
    )
    screen.add_argument(
        "--profile", help="Path to a saved voice profile JSON (preferred)."
    )
    screen.add_argument(
        "--reference-clips", nargs="+",
        help="Build a profile on the fly from these clips instead of --profile.",
    )
    screen.add_argument(
        "--threshold", type=float, default=75.0,
        help="Suitability %% needed for a pass (default: 75).",
    )
    screen.add_argument(
        "--output-dir", default="results",
        help="Directory for the screening table and chart (default: results).",
    )
    screen.add_argument(
        "--method", choices=["neural", "mfcc"], default="neural",
        help="Embedding method when using --reference-clips (default: neural).",
    )
    screen.set_defaults(func=_cmd_screen)

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
