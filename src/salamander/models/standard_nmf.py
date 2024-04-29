from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import anndata as ad

from .. import tools as tl
from ..initialization.initialize import initialize_standard_nmf
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
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

    def reduce_dimension_embeddings(
        self, method: _Dim_reduction_methods = "umap", n_components: int = 2, **kwargs
    ) -> None:
        tl.reduce_dimension(
            self.adata,
            basis="exposures",
            method=method,
            n_components=n_components,
            **kwargs,
        )

    def _get_embedding_plot_adata(
        self, method: _Dim_reduction_methods = "umap"
    ) -> tuple[ad.AnnData, str]:
        """
        Plot the exposures directly if the number of signatures is at most 2.
        """
        if self.n_signatures <= 2:
            warnings.warn(
                f"There are only {self.n_signatures} many signatures. "
                "The exposures are plotted directly.",
                UserWarning,
            )
            return self.adata, "exposures"

        return self.adata, method

    def _get_default_embedding_plot_annotations(self) -> None:
        """
        The embedding plot defaults to no annotations.
        """
        return
