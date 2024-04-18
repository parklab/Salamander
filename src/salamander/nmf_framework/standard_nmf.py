from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import anndata as ad

from .. import tools as tl
from ._utils_klnmf import check_given_parameters
from .initialization import initialize
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
    from typing import Any

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
    ):
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
        given_parameters = check_given_parameters(
            given_parameters=given_parameters,
            mutation_types_data=self.mutation_types,
            n_signatures_model=self.n_signatures,
        )
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()

        if "asignatures" in given_parameters:
            given_asignatures = given_parameters["asignatures"]
            given_signatures = given_asignatures.to_df().T
        else:
            given_signatures = None

        # initialize takes counts X of shape (n_features, n_samples),
        # and given_signatures of shape (n_features, n_given_signatures)
        W, H, signature_names = initialize(
            self.adata.X.T,
            self.n_signatures,
            self.init_method,
            given_signatures,
            **init_kwargs,
        )
        self.asignatures = ad.AnnData(W.T)
        self.asignatures.obs_names = signature_names
        self.asignatures.var_names = self.mutation_types

        # keep signature annotations
        if "asignatures" in given_parameters:
            n_given_signatures = given_asignatures.n_obs
            asignatures_new = self.asignatures[n_given_signatures:, :]
            self.asignatures = ad.concat(
                [given_asignatures, asignatures_new], join="outer"
            )

        self.adata.obsm["exposures"] = H.T
        return given_parameters

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
