"""Tests for the guide RNA design module."""

import pytest

from longspan.guide_rna import (
    GuideRNA,
    find_guides,
    score_guide,
    reverse_complement,
    _calc_gc,
    _gc_score,
    _poly_t_score,
    _seed_score,
)


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------

class TestReverseComplement:
    def test_simple(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_all_bases(self):
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("CCCC") == "GGGG"

    def test_lowercase(self):
        assert reverse_complement("atcg") == "CGAT"

    def test_empty(self):
        assert reverse_complement("") == ""


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------

class TestCalcGC:
    def test_all_gc(self):
        assert _calc_gc("GCGC") == 1.0

    def test_all_at(self):
        assert _calc_gc("ATAT") == 0.0

    def test_mixed(self):
        assert _calc_gc("ATGC") == pytest.approx(0.5)

    def test_empty(self):
        assert _calc_gc("") == 0.0

    def test_lowercase(self):
        assert _calc_gc("gcgc") == 1.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

class TestGCScore:
    def test_optimal_gc(self):
        # 55% GC should give maximum score (100)
        assert _gc_score(0.55) == pytest.approx(100.0)

    def test_low_gc(self):
        assert _gc_score(0.0) == pytest.approx(0.0)

    def test_high_gc(self):
        assert _gc_score(1.0) == pytest.approx(0.0)

    def test_boundary_lower(self):
        # 40% is acceptable
        assert _gc_score(0.40) > 0

    def test_boundary_upper(self):
        # 70% is acceptable
        assert _gc_score(0.70) > 0


class TestPolyTScore:
    def test_no_poly_t(self):
        assert _poly_t_score("ACGACGACGACGACGACGAC") == 100.0

    def test_three_t(self):
        assert _poly_t_score("ACGTTTAGCAGCAGCAGCAG") == 50.0

    def test_four_t(self):
        assert _poly_t_score("ACGTTTTGCAGCAGCAGCAG") == 0.0


class TestSeedScore:
    def test_good_seed(self):
        spacer = "AAAAAAAAACGCGCGCGCGCGG"  # seed region has balanced GC
        score = _seed_score(spacer)
        assert 0.0 <= score <= 100.0

    def test_homopolymer_penalty(self):
        # seed contains 4×G → penalty applied
        spacer = "AAAAAAAAAAAAAAAAGGGG"
        score_bad = _seed_score(spacer)
        spacer_good = "AAAAAAAAAAAAAAGCGCGC"
        score_good = _seed_score(spacer_good)
        assert score_good > score_bad


# ---------------------------------------------------------------------------
# score_guide
# ---------------------------------------------------------------------------

class TestScoreGuide:
    def test_returns_float(self):
        result = score_guide("GCACTGAGCTACGATCGACT")
        assert isinstance(result, float)

    def test_range(self):
        # Score must always be in [0, 100]
        for seq in [
            "GCACTGAGCTACGATCGACT",
            "TTTTTTTTTTTTTTTTTTTT",
            "GGGGGGGGGGGGGGGGGGGG",
            "AAAAAAAAAAAAAAAAAAAA",
        ]:
            s = score_guide(seq)
            assert 0.0 <= s <= 100.0, f"Score {s} out of range for {seq}"

    def test_poly_t_penalised(self):
        good = score_guide("GCACTGAGCTACGATCGACT")
        bad = score_guide("GCACTGAGCTACGTTTTACT")
        assert good > bad

    def test_high_gc_penalised(self):
        # Very high GC (100%) should score lower than ~55% GC
        high_gc = score_guide("GCGCGCGCGCGCGCGCGCGC")
        moderate_gc = score_guide("GCACTGAGCTACGATCGACT")
        # moderate GC should score better
        assert moderate_gc > high_gc


# ---------------------------------------------------------------------------
# find_guides
# ---------------------------------------------------------------------------

# A 60-nt sequence with two embedded NGG PAMs
_SEQ = "ATCGATCGATCGCGACTGAGCTACGATCGACTGGTATCGATCGATCGCGACTGAGCTACG"
#                   |--- 20-nt spacer ---|NGG
#                   CGACTGAGCTACGATCGACT GG  (positions 13–32, PAM at 33)


class TestFindGuides:
    def test_returns_list(self):
        guides = find_guides(_SEQ)
        assert isinstance(guides, list)

    def test_all_guide_rna_instances(self):
        guides = find_guides(_SEQ)
        for g in guides:
            assert isinstance(g, GuideRNA)

    def test_sorted_by_score(self):
        guides = find_guides(_SEQ)
        scores = [g.score for g in guides]
        assert scores == sorted(scores, reverse=True)

    def test_spacer_length(self):
        guides = find_guides(_SEQ, guide_length=20)
        for g in guides:
            assert len(g.spacer) == 20

    def test_custom_pam(self):
        # NAG is an alternative PAM recognised at lower efficiency
        guides = find_guides(_SEQ, pam="NAG")
        for g in guides:
            assert g.pam.upper().endswith("AG")

    def test_min_score_filter(self):
        all_guides = find_guides(_SEQ)
        filtered = find_guides(_SEQ, min_score=50.0)
        assert len(filtered) <= len(all_guides)
        for g in filtered:
            assert g.score >= 50.0

    def test_strand_plus_only(self):
        guides = find_guides(_SEQ, strands="+")
        for g in guides:
            assert g.strand == "+"

    def test_strand_minus_only(self):
        guides = find_guides(_SEQ, strands="-")
        for g in guides:
            assert g.strand == "-"

    def test_both_strands(self):
        guides = find_guides(_SEQ, strands="both")
        strands = {g.strand for g in guides}
        # May include both + and - guides if PAMs exist on both strands
        assert strands.issubset({"+", "-"})

    def test_invalid_strands_raises(self):
        with pytest.raises(ValueError, match="strands"):
            find_guides(_SEQ, strands="x")

    def test_invalid_guide_length_raises(self):
        with pytest.raises(ValueError):
            find_guides(_SEQ, guide_length=0)

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError):
            find_guides("ATCG", guide_length=20)

    def test_gc_content_populated(self):
        guides = find_guides(_SEQ)
        for g in guides:
            assert 0.0 <= g.gc_content <= 1.0

    def test_coordinates_within_sequence(self):
        guides = find_guides(_SEQ, strands="+")
        for g in guides:
            assert g.start >= 0
            assert g.end <= len(_SEQ)
            assert _SEQ[g.start:g.end] == g.spacer


# ---------------------------------------------------------------------------
# GuideRNA dataclass
# ---------------------------------------------------------------------------

class TestGuideRNA:
    def test_post_init_fills_gc(self):
        g = GuideRNA(spacer="GCGCGCGCGCGCGCGCGCGC", pam="GGG", start=0, end=20, strand="+")
        assert g.gc_content == pytest.approx(1.0)

    def test_post_init_fills_score(self):
        g = GuideRNA(spacer="GCACTGAGCTACGATCGACT", pam="GGG", start=0, end=20, strand="+")
        assert 0.0 <= g.score <= 100.0
