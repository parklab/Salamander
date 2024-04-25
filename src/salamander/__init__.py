"""
Salamander: a non-negative matrix factorization framework for signature analysis
================================================================================
"""

from .models.corrnmf_det import CorrNMFDet
from .models.klnmf import KLNMF
from .models.mvnmf import MvNMF
from .plot import set_salamander_style

__version__ = "0.3.2"
__all__ = ["CorrNMFDet", "KLNMF", "MvNMF"]


set_salamander_style()
