"""
Off-target prediction module for CRISPR-Cas9 guide RNAs.

This module identifies potential off-target cleavage sites within a reference
sequence by allowing a configurable number of mismatches between the guide
spacer and genomic subsequences adjacent to a PAM motif.

The approach is a heuristic alignment scan rather than a full Smith–Waterman
alignment.  It is appropriate for small-to-medium reference sequences and
educational / research use.  For whole-genome off-target analysis, dedicated
tools such as Cas-OFFinder or CRISPRitz should be used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from longspan.guide_rna import GuideRNA, _pam_to_regex, reverse_complement

# ---------------------------------------------------------------------------
# OfftargetSite dataclass
# ---------------------------------------------------------------------------


@dataclass
class OfftargetSite:
    """Represents a potential off-target cleavage site.

    Attributes
    ----------
    sequence:
        Genomic sequence at the off-target locus (same length as the guide
        spacer).
    pam:
        PAM sequence found at this locus.
    start:
        0-based start position in the *reference* sequence.
    end:
        0-based exclusive end position (``start + len(sequence)``).
    strand:
        ``'+'`` for the forward strand, ``'-'`` for the reverse complement.
    mismatches:
        Number of mismatching positions between the guide and this locus.
    mismatch_positions:
        List of 0-based positions within the spacer where mismatches occur.
    score:
        Off-target risk score (0–100).  Higher values indicate a *higher*
        probability of unintended cleavage.
    """

    sequence: str
    pam: str
    start: int
    end: int
    strand: str
    mismatches: int
    mismatch_positions: List[int] = field(default_factory=list)
    score: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if not self.score:
            self.score = _offtarget_risk_score(
                self.mismatches, self.mismatch_positions, len(self.sequence)
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OfftargetSite(sequence={self.sequence!r}, strand={self.strand!r}, "
            f"start={self.start}, mismatches={self.mismatches}, score={self.score:.1f})"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Seed region: last N bases of the spacer (adjacent to PAM).
_SEED_LENGTH = 12


def _offtarget_risk_score(
    mismatches: int, mismatch_positions: List[int], guide_length: int
) -> float:
    """Estimate the risk that this locus is cleaved off-target (0–100).

    The score decreases with each mismatch and decreases further when
    mismatches fall in the seed region (3' end, adjacent to PAM).

    Parameters
    ----------
    mismatches:
        Total number of mismatches.
    mismatch_positions:
        0-based positions of mismatches within the spacer.
    guide_length:
        Length of the spacer.
    """
    if mismatches == 0:
        return 100.0

    seed_start = guide_length - _SEED_LENGTH
    seed_mismatches = sum(1 for p in mismatch_positions if p >= seed_start)
    non_seed_mismatches = mismatches - seed_mismatches

    # Each mismatch reduces the score; seed-region mismatches penalised more
    penalty = (seed_mismatches * 25.0) + (non_seed_mismatches * 12.0)
    return round(max(0.0, 100.0 - penalty), 2)


# ---------------------------------------------------------------------------
# Off-target search
# ---------------------------------------------------------------------------


def find_offtargets(
    guide: GuideRNA,
    reference: str,
    max_mismatches: int = 3,
    pam: str = "NGG",
    strands: str = "both",
) -> List[OfftargetSite]:
    """Find potential off-target sites for *guide* in *reference*.

    Scans every PAM-adjacent window in *reference* and counts mismatches
    against the guide spacer.  Sites with ≤ *max_mismatches* are returned,
    sorted by ascending mismatch count (most dangerous first).

    Parameters
    ----------
    guide:
        The :class:`~longspan.guide_rna.GuideRNA` whose spacer is used as the
        query.
    reference:
        Reference DNA sequence to search (A/C/G/T, case-insensitive).
    max_mismatches:
        Maximum number of allowed mismatches (default 3).
    pam:
        PAM sequence to require at candidate off-target sites (default
        ``'NGG'``).
    strands:
        ``'+'``, ``'-'``, or ``'both'`` (default).

    Returns
    -------
    list[OfftargetSite]
        Off-target sites sorted by ascending mismatch count (i.e. by
        descending risk).

    Raises
    ------
    ValueError
        If *max_mismatches* is negative.
    """
    if max_mismatches < 0:
        raise ValueError(f"max_mismatches must be ≥ 0, got {max_mismatches}")

    reference = reference.upper().strip()
    pam_re = re.compile(_pam_to_regex(pam), re.IGNORECASE)
    spacer = guide.spacer.upper()
    guide_length = len(spacer)
    sites: List[OfftargetSite] = []

    if strands in ("+", "both"):
        sites.extend(
            _scan_offtargets(
                spacer, reference, pam_re, guide_length, max_mismatches, "+"
            )
        )
    if strands in ("-", "both"):
        rc = reverse_complement(reference)
        for site in _scan_offtargets(
            spacer, rc, pam_re, guide_length, max_mismatches, "-"
        ):
            seq_len = len(reference)
            fwd_end = seq_len - site.start
            fwd_start = seq_len - site.end
            sites.append(
                OfftargetSite(
                    sequence=site.sequence,
                    pam=site.pam,
                    start=fwd_start,
                    end=fwd_end,
                    strand="-",
                    mismatches=site.mismatches,
                    mismatch_positions=site.mismatch_positions,
                    score=site.score,
                )
            )

    sites.sort(key=lambda s: (s.mismatches, -s.score))
    return sites


def _scan_offtargets(
    spacer: str,
    sequence: str,
    pam_re: re.Pattern,
    guide_length: int,
    max_mismatches: int,
    strand: str,
) -> List[OfftargetSite]:
    """Scan one strand for off-target sites."""
    sites: List[OfftargetSite] = []
    for match in pam_re.finditer(sequence):
        pam_start = match.start()
        window_start = pam_start - guide_length
        if window_start < 0:
            continue
        window = sequence[window_start:pam_start]
        if not re.fullmatch(r"[ACGT]+", window):
            continue
        mm_positions = [i for i, (a, b) in enumerate(zip(spacer, window)) if a != b]
        mm_count = len(mm_positions)
        if mm_count <= max_mismatches:
            risk = _offtarget_risk_score(mm_count, mm_positions, guide_length)
            sites.append(
                OfftargetSite(
                    sequence=window,
                    pam=match.group(1),
                    start=window_start,
                    end=pam_start,
                    strand=strand,
                    mismatches=mm_count,
                    mismatch_positions=mm_positions,
                    score=risk,
                )
            )
    return sites
