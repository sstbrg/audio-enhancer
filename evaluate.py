#!/usr/bin/env python3
"""Evaluate audio quality using no-reference and reference-based metrics.

Usage:
    # No-reference (just score the quality of enhanced audio):
    python evaluate.py enhanced.wav

    # Reference-based (compare enhanced vs ground truth):
    python evaluate.py enhanced.wav --reference original_hires.wav

    # Full evaluation (all metrics):
    python evaluate.py enhanced.wav --reference original_hires.wav --full

    # Compare before/after enhancement:
    python evaluate.py --before input.mp3 --after enhanced.wav --reference original.wav

    # Batch evaluate a directory:
    python evaluate.py /path/to/enhanced/ --reference-dir /path/to/originals/

    # Select specific metrics:
    python evaluate.py enhanced.wav --metrics pam,audiobox,si-snr
"""

import argparse
import json
import sys
from pathlib import Path

from metrics.evaluate import AudioMetrics


def main():
    parser = argparse.ArgumentParser(description="Audio quality evaluation")
    parser.add_argument("input", nargs="?", help="Audio file or directory to evaluate")
    parser.add_argument("--reference", "-r", help="Reference (ground truth) audio file")
    parser.add_argument("--reference-dir", help="Directory of reference files (for batch)")
    parser.add_argument("--before", help="Pre-enhancement file (for before/after comparison)")
    parser.add_argument("--after", help="Post-enhancement file (for before/after comparison)")
    parser.add_argument("--full", action="store_true", help="Run all metrics")
    parser.add_argument("--metrics", help="Comma-separated list of metrics: pam,audiobox,muq,visqol,si-snr,sdr,cdpam")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    evaluator = AudioMetrics(device=args.device)

    # Before/after comparison mode
    if args.before and args.after:
        print("=== BEFORE enhancement ===")
        report_before = evaluator.evaluate_no_reference(args.before)
        print(report_before.summary())

        print("\n=== AFTER enhancement ===")
        if args.reference:
            report_after = evaluator.evaluate_full(args.reference, args.after)
        else:
            report_after = evaluator.evaluate_no_reference(args.after)
        print(report_after.summary())
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff"}

    files = []
    if input_path.is_dir():
        files = sorted(f for f in input_path.rglob("*") if f.suffix.lower() in audio_exts)
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    all_results = []

    for f in files:
        ref = None
        if args.reference:
            ref = args.reference
        elif args.reference_dir:
            # Match by filename stem
            ref_dir = Path(args.reference_dir)
            for ext in audio_exts:
                candidate = ref_dir / f"{f.stem}{ext}"
                if candidate.exists():
                    ref = str(candidate)
                    break
            # Also try without _enhanced suffix
            if not ref:
                stem = f.stem.replace("_enhanced", "")
                for ext in audio_exts:
                    candidate = ref_dir / f"{stem}{ext}"
                    if candidate.exists():
                        ref = str(candidate)
                        break

        if args.metrics:
            # Run selected metrics only
            selected = set(args.metrics.lower().split(","))
            report = _run_selected(evaluator, str(f), ref, selected)
        elif ref and args.full:
            report = evaluator.evaluate_full(ref, str(f))
        elif ref:
            report = evaluator.evaluate_with_reference(ref, str(f))
        else:
            report = evaluator.evaluate_no_reference(str(f))

        if args.json:
            result = {
                "file": str(f),
                "reference": ref,
                "metrics": {
                    m.name: m.score if not isinstance(m.score, float) or not m.score != m.score
                    else None
                    for m in report.metrics
                },
            }
            all_results.append(result)
        else:
            print(report.summary())

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))


def _run_selected(evaluator, audio_path, reference, selected):
    """Run only the selected metrics."""
    from metrics.evaluate import EvaluationReport

    report = EvaluationReport(
        input_file=audio_path,
        enhanced_file=audio_path,
        reference_file=reference,
    )

    metric_map = {
        "pam": lambda: evaluator.pam_score(audio_path),
        "audiobox": lambda: evaluator.audiobox_aesthetics(audio_path),
        "muq": lambda: evaluator.muq_eval(audio_path),
    }
    if reference:
        metric_map.update({
            "visqol": lambda: evaluator.visqol(reference, audio_path),
            "si-snr": lambda: evaluator.si_snr(reference, audio_path),
            "sdr": lambda: evaluator.sdr(reference, audio_path),
            "cdpam": lambda: evaluator.cdpam_score(reference, audio_path),
        })

    for name in selected:
        name = name.strip()
        if name in metric_map:
            report.metrics.append(metric_map[name]())
        else:
            print(f"Warning: unknown metric '{name}'. Available: {', '.join(metric_map.keys())}")

    return report


if __name__ == "__main__":
    main()
