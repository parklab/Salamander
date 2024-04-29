"""
A collection of NMF algorithms
"""

from .corrnmf_det import CorrNMFDet
from .klnmf import KLNMF
from .mvnmf import MvNMF

__all__ = [
    "CorrNMFDet",
    "KLNMF",
    "MvNMF",
]
