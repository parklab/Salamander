"""
A collection of NMF algorithms
"""

from .corrnmf_det import CorrNMFDet
from .klnmf import KLNMF
from .mmcorrnmf import MultimodalCorrNMF
from .mvnmf import MvNMF

__all__ = [
    "CorrNMFDet",
    "KLNMF",
    "MultimodalCorrNMF",
    "MvNMF",
]
