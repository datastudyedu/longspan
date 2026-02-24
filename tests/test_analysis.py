"""Tests for the editing outcome analysis module."""

import pytest

from longspan.guide_rna import GuideRNA
from longspan.analysis import (
    EditingOutcome,
    predict_editing_outcome,
    calculate_efficiency_score,
    _efficiency_gc_score,
    _efficiency_seed_gc,
    _microhomology_deletions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPACER = "GCACTGAGCTACGATCGACT"

# Build a 80-nt context with the spacer + PAM embedded at position 10
_CONTEXT = "ATCGATCGAT" + _SPACER + "GGG" + "ATCGATCGATCGATCGATCGATCGATCGATCGAT"


def _make_guide(start: int = 10) -> GuideRNA:
    return GuideRNA(
        spacer=_SPACER,
        pam="GGG",
        start=start,
        end=start + len(_SPACER),
        strand="+",
    )


# ---------------------------------------------------------------------------
# calculate_efficiency_score
# ---------------------------------------------------------------------------

class TestCalculateEfficiencyScore:
    def test_returns_float(self):
        guide = _make_guide()
        result = calculate_efficiency_score(guide)
        assert isinstance(result, float)

    def test_range_0_to_100(self):
        for spacer in [
            "GCACTGAGCTACGATCGACT",
            "TTTTTTTTTTTTTTTTTTTT",
            "GGGGGGGGGGGGGGGGGGGG",
            "GCGCGCGCGCGCGCGCGCGC",
        ]:
            g = GuideRNA(spacer=spacer, pam="GGG", start=0, end=20, strand="+")
            s = calculate_efficiency_score(g)
            assert 0.0 <= s <= 100.0, f"Efficiency {s} out of range for {spacer}"

    def test_poly_t_penalised(self):
        good = GuideRNA(spacer="GCACTGAGCTACGATCGACT", pam="GGG", start=0, end=20, strand="+")
        bad = GuideRNA(spacer="GCACTGAGCTTTTTGATCAC", pam="GGG", start=0, end=20, strand="+")
        assert calculate_efficiency_score(good) > calculate_efficiency_score(bad)


class TestEfficiencyGcScore:
    def test_optimal_range(self):
        assert _efficiency_gc_score(0.55) == pytest.approx(1.0)

    def test_zero_gc(self):
        assert _efficiency_gc_score(0.0) == pytest.approx(0.0)

    def test_full_gc(self):
        assert _efficiency_gc_score(1.0) == pytest.approx(0.0)


class TestEfficiencySeedGC:
    def test_good_seed(self):
        spacer = "ATCGATCGATCGATCGATCG"  # last 12: ATCGATCGATCG -> 50% GC
        result = _efficiency_seed_gc(spacer)
        assert 0.0 <= result <= 1.0

    def test_all_at_seed(self):
        spacer = "GCGCGCGCAAAAAAAAAAAT"  # seed: AAAAAAAAAAT = 0% GC
        result = _efficiency_seed_gc(spacer)
        assert result < 1.0


# ---------------------------------------------------------------------------
# predict_editing_outcome
# ---------------------------------------------------------------------------

class TestPredictEditingOutcome:
    def test_returns_list(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        assert isinstance(outcomes, list)

    def test_outcomes_are_editing_outcome_instances(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        for o in outcomes:
            assert isinstance(o, EditingOutcome)

    def test_contains_unmodified(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        types = [o.outcome_type for o in outcomes]
        assert "unmodified" in types

    def test_contains_deletion(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        types = [o.outcome_type for o in outcomes]
        assert "deletion" in types

    def test_contains_insertion(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        types = [o.outcome_type for o in outcomes]
        assert "insertion" in types

    def test_frequencies_sum_to_one(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        total = sum(o.frequency for o in outcomes)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_sorted_by_descending_frequency(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        freqs = [o.frequency for o in outcomes]
        assert freqs == sorted(freqs, reverse=True)

    def test_out_of_range_cut_raises(self):
        # Place guide start very close to end so cut site is outside context
        guide = GuideRNA(spacer=_SPACER, pam="GGG", start=200, end=220, strand="+")
        with pytest.raises(ValueError, match="(?i)cut"):
            predict_editing_outcome(guide, _CONTEXT)

    def test_context_start_offset(self):
        # Same as default but with explicit context_start=0
        guide = _make_guide()
        outcomes1 = predict_editing_outcome(guide, _CONTEXT, context_start=0)
        outcomes2 = predict_editing_outcome(guide, _CONTEXT)
        assert len(outcomes1) == len(outcomes2)

    def test_one_bp_deletion_length(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        one_bp_del = [o for o in outcomes if o.outcome_type == "deletion" and o.indel_size == 1]
        assert one_bp_del
        # Deletion of 1 bp shortens sequence by 1
        for o in one_bp_del:
            assert len(o.sequence) == len(_CONTEXT) - 1

    def test_one_bp_insertion_length(self):
        guide = _make_guide()
        outcomes = predict_editing_outcome(guide, _CONTEXT)
        one_bp_ins = [o for o in outcomes if o.outcome_type == "insertion" and o.indel_size == 1]
        assert one_bp_ins
        for o in one_bp_ins:
            assert len(o.sequence) == len(_CONTEXT) + 1


# ---------------------------------------------------------------------------
# _microhomology_deletions
# ---------------------------------------------------------------------------

class TestMicrohomologyDeletions:
    def test_returns_list(self):
        result = _microhomology_deletions(_CONTEXT, 25)
        assert isinstance(result, list)

    def test_all_are_deletions(self):
        result = _microhomology_deletions(_CONTEXT, 25)
        for o in result:
            assert o.outcome_type == "deletion"
            assert o.indel_size >= 2

    def test_max_three_results(self):
        result = _microhomology_deletions(_CONTEXT, 25)
        assert len(result) <= 3
