"""
Salamander: a non-negative matrix factorization framework for signature analysis
================================================================================
"""
from .nmf_framework.corrnmf_det import CorrNMFDet
from .nmf_framework.klnmf import KLNMF
from .nmf_framework.multimodal_corrnmf import MultimodalCorrNMF
from .nmf_framework.mvnmf import MvNMF
from .plot import set_salamander_style

__version__ = "0.3.1"
__all__ = ["CorrNMFDet", "KLNMF", "MvNMF", "MultimodalCorrNMF"]


set_salamander_style()
