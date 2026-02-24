"""
Command-line interface for the Longspan CRISPR guide RNA design tool.

Usage examples
--------------
Find guides in a sequence::

    longspan design --sequence ATCGATCG... --pam NGG

Find off-targets for a specific guide::

    longspan offtarget --guide GCACTGAGCTACGATCGACT --reference ATCG...

Predict editing outcomes::

    longspan outcomes --guide GCACTGAGCTACGATCGACT --pam GGG --context ATCG...
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from longspan.guide_rna import GuideRNA, find_guides, score_guide
from longspan.offtarget import find_offtargets
from longspan.analysis import predict_editing_outcome, calculate_efficiency_score


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_design(args: argparse.Namespace) -> int:
    """Handle the ``design`` sub-command."""
    sequence = _read_sequence(args.sequence, args.fasta)
    if sequence is None:
        print("Error: provide --sequence or --fasta", file=sys.stderr)
        return 1

    guides = find_guides(
        sequence,
        pam=args.pam,
        guide_length=args.guide_length,
        min_score=args.min_score,
        strands=args.strands,
    )

    if not guides:
        print("No guide RNAs found matching the criteria.")
        return 0

    print(f"Found {len(guides)} guide RNA(s):\n")
    _print_guides(guides, top_n=args.top)
    return 0


def _cmd_offtarget(args: argparse.Namespace) -> int:
    """Handle the ``offtarget`` sub-command."""
    reference = _read_sequence(args.reference, args.fasta)
    if reference is None:
        print("Error: provide --reference or --fasta", file=sys.stderr)
        return 1

    spacer = args.guide.upper()
    guide = GuideRNA(
        spacer=spacer,
        pam=args.pam,
        start=0,
        end=len(spacer),
        strand="+",
    )

    sites = find_offtargets(
        guide,
        reference,
        max_mismatches=args.max_mismatches,
        pam=args.pam,
        strands=args.strands,
    )

    if not sites:
        print("No off-target sites found within the mismatch threshold.")
        return 0

    print(f"Found {len(sites)} potential off-target site(s):\n")
    header = f"{'#':<4} {'Sequence':<22} {'Strand':<8} {'Start':<8} {'MM':<5} {'Risk':<6}"
    print(header)
    print("-" * len(header))
    for i, site in enumerate(sites, 1):
        print(
            f"{i:<4} {site.sequence:<22} {site.strand:<8} {site.start:<8} "
            f"{site.mismatches:<5} {site.score:.1f}"
        )
    return 0


def _cmd_outcomes(args: argparse.Namespace) -> int:
    """Handle the ``outcomes`` sub-command."""
    context = _read_sequence(args.context, args.fasta)
    if context is None:
        print("Error: provide --context or --fasta", file=sys.stderr)
        return 1

    spacer = args.guide.upper()
    # Find the guide position within the context sequence
    guides = find_guides(context, pam=args.pam, guide_length=len(spacer))
    target_guide: Optional[GuideRNA] = None
    for g in guides:
        if g.spacer == spacer:
            target_guide = g
            break

    if target_guide is None:
        # Create a synthetic guide at position 0 if exact match not found
        print(
            "Warning: guide not found in context; using position 0 as cut reference.",
            file=sys.stderr,
        )
        target_guide = GuideRNA(
            spacer=spacer, pam=args.pam, start=0, end=len(spacer), strand="+"
        )

    efficiency = calculate_efficiency_score(target_guide)
    outcomes = predict_editing_outcome(target_guide, context)

    print(f"Guide: {spacer}")
    print(f"Estimated efficiency score: {efficiency:.1f}/100\n")
    print(f"{'Outcome':<15} {'Indel':>6} {'Frequency':>10}  Description")
    print("-" * 70)
    for o in outcomes:
        indel_str = f"{o.indel_size:+d}" if o.outcome_type != "unmodified" else "—"
        print(
            f"{o.outcome_type:<15} {indel_str:>6} {o.frequency:>10.2%}  {o.description}"
        )
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_sequence(sequence_arg: Optional[str], fasta_arg: Optional[str]) -> Optional[str]:
    """Return the DNA sequence from CLI arguments or a FASTA file."""
    if sequence_arg:
        return sequence_arg.strip().upper()
    if fasta_arg:
        try:
            return _read_fasta(fasta_arg)
        except OSError as exc:
            print(f"Error reading FASTA file: {exc}", file=sys.stderr)
            return None
    return None


def _read_fasta(path: str) -> str:
    """Read the first sequence from a FASTA file (header lines ignored)."""
    bases: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if bases:  # stop after first record
                    break
                continue
            bases.append(line.upper())
    return "".join(bases)


def _print_guides(guides: List[GuideRNA], top_n: int = 10) -> None:
    subset = guides[:top_n]
    header = (
        f"{'#':<4} {'Spacer':<22} {'Strand':<8} {'Start':<8} "
        f"{'GC%':>5} {'Score':>6}"
    )
    print(header)
    print("-" * len(header))
    for i, g in enumerate(subset, 1):
        print(
            f"{i:<4} {g.spacer:<22} {g.strand:<8} {g.start:<8} "
            f"{g.gc_content:>5.0%} {g.score:>6.1f}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longspan",
        description=(
            "Longspan: CRISPR-Cas9 guide RNA design and analysis tool "
            "for longevity research."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- design ------------------------------------------------------------
    p_design = sub.add_parser("design", help="Design guide RNAs for a target sequence.")
    _add_sequence_args(p_design, dest="sequence", ref_name="sequence")
    p_design.add_argument("--pam", default="NGG", help="PAM sequence (default: NGG)")
    p_design.add_argument(
        "--guide-length", type=int, default=20, dest="guide_length",
        help="Spacer length in nt (default: 20)"
    )
    p_design.add_argument(
        "--min-score", type=float, default=0.0, dest="min_score",
        help="Minimum guide score threshold (default: 0)"
    )
    p_design.add_argument(
        "--strands", choices=["+", "-", "both"], default="both",
        help="Which strands to search (default: both)"
    )
    p_design.add_argument(
        "--top", type=int, default=10,
        help="Display top N guides (default: 10)"
    )

    # ---- offtarget ---------------------------------------------------------
    p_off = sub.add_parser("offtarget", help="Predict off-target sites for a guide.")
    _add_sequence_args(p_off, dest="reference", ref_name="reference")
    p_off.add_argument("--guide", required=True, help="Guide spacer sequence (20 nt)")
    p_off.add_argument("--pam", default="NGG", help="PAM sequence (default: NGG)")
    p_off.add_argument(
        "--max-mismatches", type=int, default=3, dest="max_mismatches",
        help="Maximum allowed mismatches (default: 3)"
    )
    p_off.add_argument(
        "--strands", choices=["+", "-", "both"], default="both",
        help="Strands to search (default: both)"
    )

    # ---- outcomes ----------------------------------------------------------
    p_out = sub.add_parser("outcomes", help="Predict editing outcomes at the cut site.")
    _add_sequence_args(p_out, dest="context", ref_name="context")
    p_out.add_argument("--guide", required=True, help="Guide spacer sequence (20 nt)")
    p_out.add_argument("--pam", default="NGG", help="PAM sequence (default: NGG)")

    return parser


def _add_sequence_args(
    parser: argparse.ArgumentParser, dest: str, ref_name: str
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{ref_name}",
        dest=dest,
        default=None,
        help=f"DNA sequence string for the {ref_name}",
    )
    group.add_argument(
        "--fasta",
        default=None,
        help=f"Path to a FASTA file containing the {ref_name}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the ``longspan`` CLI.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "design": _cmd_design,
        "offtarget": _cmd_offtarget,
        "outcomes": _cmd_outcomes,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
