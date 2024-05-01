from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable

import matplotlib.pyplot as plt

from .. import plot as pl
from .. import tools as tl
from ..initialization.initialize import initialize_standard_nmf
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .signature_nmf import _Dim_reduction_methods


class StandardNMF(SignatureNMF):
    """
    The abstract class StandardNMF unifies the structure of NMF algorithms
    with a signature and exposure matrix.

    Examples of these algorithms include the NMF algorithms from
    (Lee and Seung, 1999), minimum volume NMF (mvNMF) or any NMF variants
    with regularizations on the entries of W or H.
    All of these NMF algorithms have the same parameters. Therefore,
    their initializations are identical, and the lower-dimensional
    representations are the sample exposures.
    """

    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the signatures and exposures.
        A subset of the signatures can be given by the user. They will
        not be overwritten during fitting.

        Input:
        ------
        given_parameters : dict, default=None
            Optinally given 'asignatures' AnnData signatures object.

        init_kwargs: dict
            Any further keywords arguments to be passed to the initialization method.
            This includes, for example, an optional 'seed' for all stochastic methods.
        """
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures = initialize_standard_nmf(
            self.adata,
            self.n_signatures,
            self.init_method,
            given_parameters,
            **init_kwargs,
        )

    def plot_embeddings(
        self,
        method: _Dim_reduction_methods = "umap",
        n_components: int = 2,
        dimensions: tuple[int, int] = (0, 1),
        color: str | None = None,
        zorder: str | None = None,
        annotations: Iterable[str] | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes:
        tl.reduce_dimension(
            self.adata,
            basis="exposures",
            method=method,
            n_components=n_components,
        )
        if self.n_signatures <= 2:
            warnings.warn(
                f"There are only {self.n_signatures} many signatures. "
                "The exposures are plotted directly.",
                UserWarning,
            )
            basis = "exposures"
        else:
            basis = method

        ax = pl.embedding(
            adata=self.adata,
            basis=basis,
            dimensions=dimensions,
            color=color,
            zorder=zorder,
            annotations=annotations,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax
