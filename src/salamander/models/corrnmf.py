from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
from scipy.spatial.distance import squareform

from ..initialization.initialize import initialize_corrnmf
from ..tools import reduce_dimension_multiple
from ..utils import value_checker
from . import _utils_corrnmf
from ._utils_klnmf import samplewise_kl_divergence
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
    from ..initialization.methods import _Init_methods
    from .signature_nmf import _Dim_reduction_methods


class CorrNMF(SignatureNMF):
    """
    The abstract class CorrNMF unifies the structure of NMF algorithms
    with a signature matrix and an exposure matrix refactored into
    signature and sample scalings and embeddings.
    """

    def __init__(
        self,
        n_signatures: int = 1,
        init_method: _Init_methods = "nndsvd",
        dim_embeddings: int | None = None,
        min_iterations: int = 500,
        max_iterations: int = 10000,
        conv_test_freq: int = 10,
        tol: float = 1e-7,
    ):
        """
        Input:
        ------
        dim_embeddings: int
            The assumed dimension of the signature and sample embeddings.
            Should be smaller or equal to the number of signatures as a dimension
            equal to the number of signatures covers the case of independent
            signatures. The smaller the embedding dimension, the stronger the
            enforced correlation structure on both signatures and samples.
        """
        super().__init__(
            n_signatures,
            init_method,
            min_iterations,
            max_iterations,
            conv_test_freq,
            tol,
        )
        if dim_embeddings is None:
            dim_embeddings = n_signatures

        self.dim_embeddings = dim_embeddings
        self.variance = 1.0

    def compute_exposures(self) -> None:
        """
        In contrast to the classical NMF framework, the exposure matrix is
        restructured and determined by the signature & sample biases and
        embeddings.
        """
        self.adata.obsm["exposures"] = _utils_corrnmf.compute_exposures(
            self.asignatures.obs["scalings"].values,
            self.adata.obs["scalings"].values,
            self.asignatures.obsm["embeddings"],
            self.adata.obsm["embeddings"],
        )

    def compute_reconstruction_errors(self):
        self.compute_exposures()
        errors = samplewise_kl_divergence(
            self.adata.X.T, self.asignatures.X.T, self.adata.obsm["exposures"].T
        )
        self.adata.obs["reconstruction_error"] = errors

    def objective_function(self, penalize_sample_embeddings: bool = True) -> float:
        """
        The evidence lower bound (ELBO)
        """
        return _utils_corrnmf.elbo_corrnmf(
            self.adata.X,
            self.asignatures.X,
            self.adata.obsm["exposures"],
            self.asignatures.obsm["embeddings"],
            self.adata.obsm["embeddings"],
            self.variance,
            penalize_sample_embeddings=penalize_sample_embeddings,
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the signature matrix, the signature and sample scalings,
        the signature and sample embeddings, and the variance.

        Parameters
        ----------
        given_parameters: dict, default=None
            A priori known parameters / parameters to fix during model training.
            Allowed keys: 'asignatures', 'signature_scalings', 'sample_scalings',
            'signature_embeddings', 'sample_embeddings'. The values have to
            have the appropriate shape. If 'asignatures' is not None, it is
            expected to be an AnnData object.

        init_kwargs : dict
            Any further keyword arguments to pass to the initialization method
            of the signatures. This includes, for example, a possible 'seed'
            keyword argument for all stochastic initialization methods.
        """
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures, self.variance = initialize_corrnmf(
            self.adata,
            self.n_signatures,
            self.dim_embeddings,
            self.init_method,
            given_parameters,
            **init_kwargs,
        )
        self.compute_exposures()

    def _setup_fitting_parameters(
        self, fitting_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        No additional fitting parameters implemented so far.
        """
        return

    def compute_correlation_scaled(
        self, data: Literal["samples", "signatures"] = "signatures"
    ) -> None:
        """
        Compute the signature or sample correlation based on the
        scaled exposures and store it in the respective anndata object.
        """
        value_checker("data", data, ["samples", "signatures"])
        assert "embeddings" in self.adata.obsm, (
            "Computing the sample or signature correlation "
            "requires fitting the CorrNMF model."
        )

        if data == "samples":
            vectors = self.adata.obsm["embeddings"]
        else:
            vectors = self.asignatures.obsm["embeddings"]

        norms = np.sqrt(np.sum(vectors**2, axis=1))
        n_vectors = len(norms)
        corr_vector = np.array(
            [
                np.dot(v1, v2) / (norms[i1] * norms[i1 + i2 + 1])
                for i1, v1 in enumerate(vectors)
                for i2, v2 in enumerate(vectors[(i1 + 1) :, :])
            ]
        )
        correlation = squareform(corr_vector) + np.identity(n_vectors)

        if data == "samples":
            self.adata.obsp["X_correlation"] = correlation
        else:
            self.asignatures.obsp["correlation"] = correlation

    def reduce_dimension_embeddings(
        self, method: _Dim_reduction_methods = "umap", n_components: int = 2, **kwargs
    ) -> None:
        reduce_dimension_multiple(
            adatas=[self.asignatures, self.adata],
            basis="embeddings",
            method=method,
            n_components=n_components,
            **kwargs,
        )

    def _get_embedding_plot_adata(
        self, method: _Dim_reduction_methods = "umap"
    ) -> tuple[ad.AnnData, str]:
        """
        Plot the embeddings directly if the embedding dimension is at most 2.
        """
        plot_adata = ad.concat([self.asignatures, self.adata])

        if self.dim_embeddings <= 2:
            warnings.warn(
                f"The embedding dimension is {self.dim_embeddings}. "
                "The embeddings are plotted without an additional "
                "dimensionality reduction.",
                UserWarning,
            )
            basis = "embeddings"
        else:
            basis = method

        return plot_adata, basis

    def _get_default_embedding_plot_annotations(self) -> list[str]:
        """
        The embedding plot defaults to annotating the signature embeddings.
        """
        return self.signature_names
