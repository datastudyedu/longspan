"""
Editing outcome analysis module for CRISPR-Cas9.

After Cas9 creates a double-strand break (DSB), the cell repairs the cut
predominantly via non-homologous end-joining (NHEJ), which introduces
insertions and deletions (indels) that disrupt the target sequence.  This
module provides heuristic predictions of:

- The most likely indel patterns at the cut site.
- An efficiency score based on guide sequence features correlated with
  high editing rates in published datasets.

References
----------
- Doench et al. (2016) *Optimized sgRNA design to maximize activity and
  minimize off-target effects of CRISPR-Cas9*. Nature Biotechnology.
- Shen et al. (2018) *Predictable and precise template-free NHEJ repair of
  both 1- and 2-cut CRISPR/Cas9 substrates*. eLife.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from longspan.guide_rna import GuideRNA, _calc_gc


# ---------------------------------------------------------------------------
# EditingOutcome dataclass
# ---------------------------------------------------------------------------


@dataclass
class EditingOutcome:
    """Predicted repair outcome at a CRISPR cut site.

    Attributes
    ----------
    outcome_type:
        Category of outcome: ``'deletion'``, ``'insertion'``, or
        ``'unmodified'``.
    sequence:
        Predicted resulting sequence around the cut site.
    frequency:
        Estimated frequency of this outcome (0.0–1.0).
    indel_size:
        Length of the insertion or deletion (0 for unmodified).
    description:
        Human-readable description of the outcome.
    """

    outcome_type: str
    sequence: str
    frequency: float
    indel_size: int = 0
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EditingOutcome(type={self.outcome_type!r}, indel={self.indel_size:+d}, "
            f"freq={self.frequency:.2%})"
        )


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------

# Feature weights derived from the Doench 2016 Rule Set 2 heuristic
_DOENCH_WEIGHTS: Dict[str, float] = {
    "gc_optimal": 0.35,
    "no_poly_t": 0.20,
    "seed_gc": 0.20,
    "g_at_position1": 0.10,
    "no_tt_dinuc": 0.15,
}


def _efficiency_gc_score(gc: float) -> float:
    """Score GC content: peak at 50–65 %."""
    if 0.50 <= gc <= 0.65:
        return 1.0
    if 0.40 <= gc < 0.50:
        return (gc - 0.40) / 0.10
    if 0.65 < gc <= 0.75:
        return (0.75 - gc) / 0.10
    return 0.0


def _efficiency_seed_gc(spacer: str, seed_len: int = 12) -> float:
    """Score GC content in seed region (last *seed_len* bases)."""
    seed = spacer[-seed_len:]
    gc = _calc_gc(seed)
    return 1.0 if 0.40 <= gc <= 0.65 else max(0.0, 1.0 - abs(gc - 0.525) * 4)


def calculate_efficiency_score(guide: GuideRNA) -> float:
    """Predict on-target editing efficiency for *guide* (0–100).

    The score is a weighted combination of guide sequence features associated
    with high editing rates.

    Parameters
    ----------
    guide:
        A :class:`~longspan.guide_rna.GuideRNA` object.

    Returns
    -------
    float
        Efficiency score between 0 and 100.
    """
    spacer = guide.spacer.upper()
    gc = guide.gc_content

    gc_s = _efficiency_gc_score(gc)
    poly_t_s = 0.0 if "TTTT" in spacer else (0.5 if "TTT" in spacer else 1.0)
    seed_s = _efficiency_seed_gc(spacer)
    g1_s = 1.0 if spacer[0] == "G" else 0.5
    tt_s = 0.0 if "TT" in spacer[-8:] else 1.0

    score = (
        _DOENCH_WEIGHTS["gc_optimal"] * gc_s
        + _DOENCH_WEIGHTS["no_poly_t"] * poly_t_s
        + _DOENCH_WEIGHTS["seed_gc"] * seed_s
        + _DOENCH_WEIGHTS["g_at_position1"] * g1_s
        + _DOENCH_WEIGHTS["no_tt_dinuc"] * tt_s
    )
    return round(min(100.0, score * 100.0), 2)


# ---------------------------------------------------------------------------
# Indel prediction
# ---------------------------------------------------------------------------

# SpCas9 cuts 3 bp upstream of the PAM (between positions 17 and 18 of the
# spacer when 1-indexed from the 5' end, i.e. index 17 in 0-based).
_CUT_OFFSET = 17


def predict_editing_outcome(
    guide: GuideRNA, context_sequence: str, context_start: int = 0
) -> List[EditingOutcome]:
    """Predict likely NHEJ repair outcomes at the guide's cut site.

    The function uses the local sequence context around the predicted DSB to
    estimate the most common indel outcomes according to simple microhomology
    and sequence-context heuristics.

    Parameters
    ----------
    guide:
        The :class:`~longspan.guide_rna.GuideRNA` whose cut site is analysed.
    context_sequence:
        A DNA sequence that includes the guide target.  Should span at least
        20 nt up- and downstream of the cut site for reliable predictions.
    context_start:
        0-based position in *context_sequence* corresponding to position 0 of
        the *guide's* ``start`` coordinate.  Defaults to 0.

    Returns
    -------
    list[EditingOutcome]
        Predicted outcomes sorted by descending frequency.

    Raises
    ------
    ValueError
        If the cut site falls outside *context_sequence*.
    """
    context = context_sequence.upper()
    # Absolute cut position in context_sequence (between cut_pos-1 and cut_pos)
    cut_pos = guide.start - context_start + _CUT_OFFSET

    if cut_pos < 0 or cut_pos >= len(context):
        raise ValueError(
            f"Cut position {cut_pos} is outside context_sequence "
            f"(length {len(context)}). Provide a wider context."
        )

    outcomes: List[EditingOutcome] = []

    # ---- 1-bp deletion (most frequent NHEJ outcome) -----------------------
    if cut_pos < len(context):
        del_seq = context[: cut_pos] + context[cut_pos + 1 :]
        outcomes.append(
            EditingOutcome(
                outcome_type="deletion",
                sequence=del_seq,
                frequency=0.35,
                indel_size=1,
                description="1-bp deletion at cut site (most common NHEJ outcome)",
            )
        )

    # ---- 1-bp insertion (template-free +1 insertion) ----------------------
    # Duplicates the base just upstream of the cut (position cut_pos - 1)
    if cut_pos > 0:
        ins_base = context[cut_pos - 1]
        ins_seq = context[:cut_pos] + ins_base + context[cut_pos:]
        outcomes.append(
            EditingOutcome(
                outcome_type="insertion",
                sequence=ins_seq,
                frequency=0.25,
                indel_size=1,
                description=f"1-bp insertion (+{ins_base}) at cut site",
            )
        )

    # ---- Microhomology-mediated deletion ----------------------------------
    mh_outcomes = _microhomology_deletions(context, cut_pos)
    outcomes.extend(mh_outcomes)

    # ---- Unmodified (NHEJ with perfect ligation) --------------------------
    total_freq = sum(o.frequency for o in outcomes)
    unmod_freq = max(0.0, 1.0 - total_freq)
    outcomes.append(
        EditingOutcome(
            outcome_type="unmodified",
            sequence=context,
            frequency=round(unmod_freq, 4),
            indel_size=0,
            description="Unmodified (no editing or perfect repair)",
        )
    )

    outcomes.sort(key=lambda o: o.frequency, reverse=True)
    return outcomes


def _microhomology_deletions(
    context: str, cut_pos: int, max_mh_len: int = 5, max_del_len: int = 20
) -> List[EditingOutcome]:
    """Find microhomology-mediated deletions around *cut_pos*.

    Returns up to 3 predicted MH-deletion outcomes with estimated frequencies.
    """
    outcomes: List[EditingOutcome] = []
    # Search for microhomologies: identical sequences flanking the cut
    for del_len in range(2, max_del_len + 1):
        left_start = max(0, cut_pos - max_mh_len)
        left_end = cut_pos
        right_start = cut_pos
        right_end = min(len(context), cut_pos + del_len)

        left_flank = context[left_start:left_end]
        right_flank = context[right_start:right_end]

        # Find shared prefix length (microhomology)
        mh_len = 0
        for a, b in zip(reversed(left_flank), right_flank):
            if a == b:
                mh_len += 1
            else:
                break

        if mh_len >= 2:
            del_seq = context[:cut_pos] + context[cut_pos + del_len :]
            freq = round(0.05 * mh_len / del_len, 4)
            outcomes.append(
                EditingOutcome(
                    outcome_type="deletion",
                    sequence=del_seq,
                    frequency=freq,
                    indel_size=del_len,
                    description=(
                        f"{del_len}-bp microhomology-mediated deletion "
                        f"(MH length={mh_len})"
                    ),
                )
            )
            if len(outcomes) >= 3:
                break

    return outcomes
