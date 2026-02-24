"""Tests for the CLI module."""

import pytest

from longspan.cli import main, _read_fasta
import os
import tempfile


# ---------------------------------------------------------------------------
# Helper sequences
# ---------------------------------------------------------------------------

_SEQ = "ATCGATCGATCGCGACTGAGCTACGATCGACTGGTATCGATCGATCGCGACTGAGCTACG"


# ---------------------------------------------------------------------------
# design sub-command
# ---------------------------------------------------------------------------

class TestDesignCommand:
    def test_basic_design(self, capsys):
        ret = main(["design", "--sequence", _SEQ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "guide RNA" in out.lower() or "found" in out.lower()

    def test_no_guides_found(self, capsys):
        # A sequence without any NGG PAM
        ret = main(["design", "--sequence", "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"])
        assert ret == 0

    def test_min_score_filter(self, capsys):
        ret = main(["design", "--sequence", _SEQ, "--min-score", "90"])
        assert ret == 0

    def test_strand_plus(self, capsys):
        ret = main(["design", "--sequence", _SEQ, "--strands", "+"])
        assert ret == 0

    def test_fasta_input(self, capsys, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(f">seq1\n{_SEQ}\n")
        ret = main(["design", "--fasta", str(fasta)])
        assert ret == 0

    def test_no_sequence_returns_error(self, capsys):
        ret = main(["design"])
        assert ret == 1

    def test_top_n(self, capsys):
        ret = main(["design", "--sequence", _SEQ, "--top", "3"])
        assert ret == 0


# ---------------------------------------------------------------------------
# offtarget sub-command
# ---------------------------------------------------------------------------

_SPACER = "CGACTGAGCTACGATCGACT"
_REF = "ATCGATCG" + _SPACER + "GGG" + "ATCGATCGATCGATCGATCGATCGATCGATCGAT"


class TestOfftargetCommand:
    def test_basic_offtarget(self, capsys):
        ret = main(["offtarget", "--guide", _SPACER, "--reference", _REF])
        assert ret == 0

    def test_no_sites_found(self, capsys):
        ret = main([
            "offtarget", "--guide", _SPACER,
            "--reference", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "--max-mismatches", "0",
        ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "no off-target" in out.lower()

    def test_no_reference_returns_error(self, capsys):
        ret = main(["offtarget", "--guide", _SPACER])
        assert ret == 1


# ---------------------------------------------------------------------------
# outcomes sub-command
# ---------------------------------------------------------------------------

_CONTEXT = "ATCGATCGAT" + _SPACER + "GGG" + "ATCGATCGATCGATCGATCGATCGATCGATCGAT"


class TestOutcomesCommand:
    def test_basic_outcomes(self, capsys):
        ret = main(["outcomes", "--guide", _SPACER, "--context", _CONTEXT])
        assert ret == 0
        out = capsys.readouterr().out
        assert "efficiency" in out.lower()

    def test_no_context_returns_error(self, capsys):
        ret = main(["outcomes", "--guide", _SPACER])
        assert ret == 1

    def test_guide_not_in_context_warns(self, capsys):
        # Guide is not present in this context; should warn but not crash
        ret = main([
            "outcomes", "--guide", _SPACER,
            "--context", "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        ])
        # Should succeed (returns 0) or exit gracefully
        assert ret in (0, 1)


# ---------------------------------------------------------------------------
# --version flag
# ---------------------------------------------------------------------------

class TestVersionFlag:
    def test_version_exits(self):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# _read_fasta helper
# ---------------------------------------------------------------------------

class TestReadFasta:
    def test_single_record(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1\nATCGATCG\nATCGATCG\n")
        result = _read_fasta(str(fasta))
        assert result == "ATCGATCGATCGATCG"

    def test_multi_record_reads_first(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">seq1\nATCG\n>seq2\nGCTA\n")
        result = _read_fasta(str(fasta))
        assert result == "ATCG"
