"""Tests for the off-target prediction module."""

import pytest

from longspan.guide_rna import GuideRNA
from longspan.offtarget import (
    OfftargetSite,
    find_offtargets,
    _offtarget_risk_score,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A 60-nt reference with an exact copy of the guide + PAM embedded
_SPACER = "GCACTGAGCTACGATCGACT"
_REF = (
    "ATCGATCG"
    + _SPACER + "GGG"
    + "ATCGATCGATCGATCGATCGATCGATCGAT"
)


def _make_guide(spacer: str = _SPACER, start: int = 8) -> GuideRNA:
    return GuideRNA(
        spacer=spacer,
        pam="GGG",
        start=start,
        end=start + len(spacer),
        strand="+",
    )


# ---------------------------------------------------------------------------
# _offtarget_risk_score
# ---------------------------------------------------------------------------

class TestOfftargetRiskScore:
    def test_perfect_match_is_100(self):
        assert _offtarget_risk_score(0, [], 20) == pytest.approx(100.0)

    def test_seed_mismatch_penalised_more(self):
        # Mismatch in seed (positions 8-19 in 0-based for a 20-nt guide)
        seed_score = _offtarget_risk_score(1, [15], 20)
        non_seed_score = _offtarget_risk_score(1, [2], 20)
        assert seed_score < non_seed_score

    def test_multiple_mismatches_lower_score(self):
        s0 = _offtarget_risk_score(0, [], 20)
        s1 = _offtarget_risk_score(1, [5], 20)
        s3 = _offtarget_risk_score(3, [3, 8, 15], 20)
        assert s0 > s1 > s3

    def test_score_non_negative(self):
        # Even many mismatches should not produce negative scores
        assert _offtarget_risk_score(10, list(range(10)), 20) >= 0.0


# ---------------------------------------------------------------------------
# find_offtargets
# ---------------------------------------------------------------------------

class TestFindOfftargets:
    def test_exact_match_found(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, max_mismatches=0)
        # Should find the exact embedded site
        assert any(s.mismatches == 0 for s in sites)

    def test_returns_list_of_offtarget_sites(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF)
        assert isinstance(sites, list)
        for s in sites:
            assert isinstance(s, OfftargetSite)

    def test_sorted_by_mismatches(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, max_mismatches=3)
        mms = [s.mismatches for s in sites]
        assert mms == sorted(mms)

    def test_max_mismatches_respected(self):
        guide = _make_guide()
        for max_mm in (0, 1, 2, 3):
            sites = find_offtargets(guide, _REF, max_mismatches=max_mm)
            for s in sites:
                assert s.mismatches <= max_mm

    def test_negative_max_mismatches_raises(self):
        guide = _make_guide()
        with pytest.raises(ValueError):
            find_offtargets(guide, _REF, max_mismatches=-1)

    def test_strand_plus_only(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, strands="+")
        for s in sites:
            assert s.strand == "+"

    def test_strand_minus_only(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, strands="-")
        for s in sites:
            assert s.strand == "-"

    def test_mismatch_positions_length(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, max_mismatches=3)
        for s in sites:
            assert len(s.mismatch_positions) == s.mismatches

    def test_score_range(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, max_mismatches=3)
        for s in sites:
            assert 0.0 <= s.score <= 100.0

    def test_no_sites_returned_when_no_match(self):
        # All-N sequence won't have valid ACGT windows
        guide = _make_guide()
        sites = find_offtargets(guide, "ATCGATCGATCG", max_mismatches=0)
        # This tiny reference is shorter than guide + PAM; should return nothing
        assert sites == []

    def test_perfect_match_has_score_100(self):
        guide = _make_guide()
        sites = find_offtargets(guide, _REF, max_mismatches=0, strands="+")
        perfect = [s for s in sites if s.mismatches == 0]
        for s in perfect:
            assert s.score == pytest.approx(100.0)
