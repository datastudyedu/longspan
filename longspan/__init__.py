"""
Longspan: A Python tool for designing and analyzing CRISPR-Cas9 guide RNAs
for efficient and precise genome editing with a focus on longevity research.
"""

from longspan.guide_rna import GuideRNA, find_guides, score_guide
from longspan.offtarget import OfftargetSite, find_offtargets
from longspan.analysis import EditingOutcome, predict_editing_outcome, calculate_efficiency_score

__version__ = "0.1.0"
__all__ = [
    "GuideRNA",
    "find_guides",
    "score_guide",
    "OfftargetSite",
    "find_offtargets",
    "EditingOutcome",
    "predict_editing_outcome",
    "calculate_efficiency_score",
]
