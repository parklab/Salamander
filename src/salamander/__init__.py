"""
Salamander: a non-negative matrix factorization framework for signature analysis
================================================================================
"""

from . import plot as pl
from . import tools as tl
from .models.corrnmf_det import CorrNMFDet
from .models.klnmf import KLNMF
from .models.mvnmf import MvNMF

__version__ = "0.3.2"
__all__ = ["CorrNMFDet", "KLNMF", "MvNMF"]


pl.set_salamander_style()
