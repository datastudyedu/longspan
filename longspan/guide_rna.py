"""
Guide RNA design module for CRISPR-Cas9 genome editing.

This module provides functions to identify candidate guide RNA sequences
within a target DNA region, evaluate their quality, and return scored
:class:`GuideRNA` objects ready for downstream analysis.

Supported PAM sequences
-----------------------
- ``NGG``  – *S. pyogenes* SpCas9 (default)
- ``NNGRRT`` – *S. aureus* SaCas9
- ``NNNRRT``  – SaCas9 variants
- Custom PAM strings using IUPAC ambiguity codes

IUPAC ambiguity codes used internally:
  N = A/C/G/T
  R = A/G
  Y = C/T
  S = G/C
  W = A/T
  K = G/T
  M = A/C
  B = C/G/T
  D = A/G/T
  H = A/C/T
  V = A/C/G
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# IUPAC helpers
# ---------------------------------------------------------------------------

_IUPAC_RE: dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "N": "[ACGT]",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
}


def _pam_to_regex(pam: str) -> str:
    """Convert an IUPAC PAM string to a regular-expression pattern.

    The pattern is wrapped in a lookahead so that :func:`re.finditer` returns
    overlapping matches (important when consecutive PAM occurrences share
    bases, e.g. ``GGGG`` contains two overlapping ``NGG`` PAMs).
    """
    inner = "".join(_IUPAC_RE.get(base.upper(), base.upper()) for base in pam)
    return f"(?=({inner}))"


def _complement(base: str) -> str:
    return {"A": "T", "T": "A", "G": "C", "C": "G"}.get(base.upper(), "N")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return "".join(_complement(b) for b in reversed(seq.upper()))


# ---------------------------------------------------------------------------
# GuideRNA dataclass
# ---------------------------------------------------------------------------


@dataclass
class GuideRNA:
    """Represents a candidate CRISPR-Cas9 guide RNA.

    Attributes
    ----------
    spacer:
        The 20-nt (or custom-length) spacer sequence (5'→3', DNA).
    pam:
        The PAM sequence immediately 3' of the spacer in the genomic strand.
    start:
        0-based start position of the spacer in the *input* sequence.
    end:
        0-based exclusive end position of the spacer (``start + len(spacer)``).
    strand:
        ``'+'`` if the guide targets the forward strand, ``'-'`` for the
        reverse complement strand.
    gc_content:
        Fraction of G/C bases in the spacer (0.0–1.0).
    score:
        Composite on-target quality score (0–100).  Higher is better.
    off_target_count:
        Number of predicted off-target sites (populated externally).
    """

    spacer: str
    pam: str
    start: int
    end: int
    strand: str
    gc_content: float = field(default=0.0, compare=False)
    score: float = field(default=0.0, compare=False)
    off_target_count: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if not self.gc_content:
            self.gc_content = _calc_gc(self.spacer)
        if not self.score:
            self.score = score_guide(self.spacer)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GuideRNA(spacer={self.spacer!r}, strand={self.strand!r}, "
            f"start={self.start}, score={self.score:.1f})"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Weights used in composite score (must sum to 1.0)
_GC_WEIGHT = 0.40
_SEED_WEIGHT = 0.30
_POLY_WEIGHT = 0.20
_POSITION_WEIGHT = 0.10


def _calc_gc(seq: str) -> float:
    """Return the GC fraction of *seq*."""
    if not seq:
        return 0.0
    upper = seq.upper()
    return (upper.count("G") + upper.count("C")) / len(upper)


def _gc_score(gc: float) -> float:
    """Return 0–100 score favouring 40–70 % GC content."""
    if 0.40 <= gc <= 0.70:
        # Peak at 55 %
        return 100.0 - abs(gc - 0.55) * 200.0
    if gc < 0.40:
        return max(0.0, gc / 0.40 * 60.0)
    return max(0.0, (1.0 - gc) / 0.30 * 60.0)


def _seed_score(spacer: str, seed_length: int = 12) -> float:
    """Score the seed region (3' end of spacer, adjacent to PAM).

    Penalises homopolymers or extreme GC in the seed.
    """
    seed = spacer[-seed_length:].upper()
    gc = _calc_gc(seed)
    base_score = 100.0 - abs(gc - 0.50) * 150.0
    # Penalise runs of ≥4 identical bases
    for base in "ACGT":
        if base * 4 in seed:
            base_score -= 30.0
    return max(0.0, base_score)


def _poly_t_score(spacer: str) -> float:
    """Penalise poly-T stretches that terminate RNA Pol III transcription."""
    upper = spacer.upper()
    if "TTTT" in upper:
        return 0.0
    if "TTT" in upper:
        return 50.0
    return 100.0


def _position_score(spacer: str) -> float:
    """Prefer G at position 1 and avoid G at position 20 (1-based)."""
    upper = spacer.upper()
    score = 50.0
    if upper[0] == "G":
        score += 25.0
    if upper[-1] != "G":
        score += 25.0
    return score


def score_guide(spacer: str) -> float:
    """Compute a composite on-target quality score for *spacer* (0–100).

    Parameters
    ----------
    spacer:
        DNA spacer sequence (5'→3').

    Returns
    -------
    float
        Score between 0 and 100 where higher values indicate a better guide.
    """
    gc = _calc_gc(spacer)
    composite = (
        _GC_WEIGHT * _gc_score(gc)
        + _SEED_WEIGHT * _seed_score(spacer)
        + _POLY_WEIGHT * _poly_t_score(spacer)
        + _POSITION_WEIGHT * _position_score(spacer)
    )
    return round(min(100.0, max(0.0, composite)), 2)


# ---------------------------------------------------------------------------
# Guide finding
# ---------------------------------------------------------------------------


def find_guides(
    sequence: str,
    pam: str = "NGG",
    guide_length: int = 20,
    min_score: float = 0.0,
    strands: str = "both",
) -> List[GuideRNA]:
    """Find all candidate guide RNAs in *sequence*.

    The function searches both the forward and reverse-complement strands by
    default and returns :class:`GuideRNA` objects sorted by descending score.

    Parameters
    ----------
    sequence:
        Target DNA sequence (A/C/G/T, case-insensitive).
    pam:
        PAM sequence using IUPAC ambiguity codes.  Defaults to ``'NGG'``
        (SpCas9).
    guide_length:
        Length of the spacer (default 20 nt).
    min_score:
        Minimum composite score threshold; guides below this value are
        excluded.
    strands:
        ``'+'`` to search only the forward strand, ``'-'`` for only the
        reverse strand, or ``'both'`` for both (default).

    Returns
    -------
    list[GuideRNA]
        Candidate guides sorted by descending :attr:`GuideRNA.score`.

    Raises
    ------
    ValueError
        If *sequence* is too short, *guide_length* is non-positive, or
        *strands* is not one of ``'+'``, ``'-'``, ``'both'``.
    """
    sequence = sequence.upper().strip()
    _validate_inputs(sequence, guide_length, strands)

    pam_re = re.compile(_pam_to_regex(pam), re.IGNORECASE)
    guides: List[GuideRNA] = []

    if strands in ("+", "both"):
        guides.extend(_scan_strand(sequence, pam_re, guide_length, "+"))
    if strands in ("-", "both"):
        rc = reverse_complement(sequence)
        for g in _scan_strand(rc, pam_re, guide_length, "-"):
            # Map coordinates back to the forward strand
            seq_len = len(sequence)
            fwd_end = seq_len - g.start
            fwd_start = seq_len - g.end
            guides.append(
                GuideRNA(
                    spacer=g.spacer,
                    pam=g.pam,
                    start=fwd_start,
                    end=fwd_end,
                    strand="-",
                    gc_content=g.gc_content,
                    score=g.score,
                )
            )

    guides = [g for g in guides if g.score >= min_score]
    guides.sort(key=lambda g: g.score, reverse=True)
    return guides


def _validate_inputs(sequence: str, guide_length: int, strands: str) -> None:
    if guide_length <= 0:
        raise ValueError(f"guide_length must be positive, got {guide_length}")
    if strands not in ("+", "-", "both"):
        raise ValueError(f"strands must be '+', '-', or 'both', got {strands!r}")
    if len(sequence) < guide_length:
        raise ValueError(
            f"Sequence length ({len(sequence)}) is shorter than guide_length ({guide_length})"
        )


def _scan_strand(
    sequence: str, pam_re: re.Pattern, guide_length: int, strand: str
) -> List[GuideRNA]:
    """Scan *sequence* for PAM occurrences and yield guide candidates."""
    guides: List[GuideRNA] = []
    for match in pam_re.finditer(sequence):
        pam_start = match.start()
        spacer_start = pam_start - guide_length
        if spacer_start < 0:
            continue
        spacer = sequence[spacer_start:pam_start]
        # Ensure spacer contains only valid DNA bases
        if not re.fullmatch(r"[ACGT]+", spacer):
            continue
        gc = _calc_gc(spacer)
        sc = score_guide(spacer)
        guides.append(
            GuideRNA(
                spacer=spacer,
                pam=match.group(1),
                start=spacer_start,
                end=pam_start,
                strand=strand,
                gc_content=gc,
                score=sc,
            )
        )
    return guides
