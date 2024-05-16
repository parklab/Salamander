"""
Salamander: a non-negative matrix factorization framework for signature analysis
================================================================================
"""

from . import models
from . import plot as pl
from . import tools as tl

__version__ = "0.4.1"

pl.set_salamander_style()

__all__ = [
    "__version__",
    "models",
    "pl",
    "tl",
]
