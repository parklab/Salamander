from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..utils import shape_checker, type_checker
from . import _utils_klnmf
from .standard_nmf import StandardNMF

if TYPE_CHECKING:
    from ..initialization.methods import _Init_methods

_FITTING_KWARGS = ["weights_kl", "weights_lhalf"]
_DEFAULT_FITTING_KWARGS = {kwarg: None for kwarg in _FITTING_KWARGS}


class KLNMF(StandardNMF):
    """
    Decompose a mutation count matrix X into the product of a signature
    matrix W and an exposure matrix H by minimizing the weighted
    generalized Kullback-Leibler (KL) divergence under the constraint of
    having normalized signatures.
    The implementation supports a sparstiy-inducing l_half penalty of the
    exposures.

    Reference
    ---------
    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """

    def __init__(
        self,
        n_signatures: int = 1,
        init_method: _Init_methods = "nndsvd",
        min_iterations: int = 500,
        max_iterations: int = 10000,
        conv_test_freq: int = 10,
        tol: float = 1e-7,
    ):
        super().__init__(
            n_signatures,
            init_method,
            min_iterations,
            max_iterations,
            conv_test_freq,
            tol,
        )
        self.weights_kl = None
        self.weights_l_half = None

    def compute_reconstruction_errors(self) -> None:
        """
        Add the unweighted samplewise Kullback-Leibler divergences
        as observation annotations to the AnnData count data.
        """
        errors = _utils_klnmf.samplewise_kl_divergence(
            self.adata.X.T, self.asignatures.X.T, self.adata.obsm["exposures"].T
        )
        self.adata.obs["reconstruction_error"] = errors

    def objective_function(self) -> float:
        """
        The sum of the (weighted) Kullback-Leibler divergence and the sparsity
        penalty.
        """
        of_value = _utils_klnmf.kl_divergence(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            self.weights_kl,
        )
        if self.weights_l_half is not None:
            of_value += np.dot(
                self.weights_l_half,
                np.sum(np.sqrt(self.adata.obsm["exposures"].T), axis=0),
            )
        return of_value

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"

    def _update_parameters(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "asignatures" in given_parameters:
            n_given_signatures = given_parameters["asignatures"].n_obs
        else:
            n_given_signatures = 0

        W, H = _utils_klnmf.update_WH(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            self.weights_kl,
            self.weights_l_half,
            n_given_signatures,
        )
        self.asignatures.X = W.T
        self.adata.obsm["exposures"] = H.T

    def _check_weights(self, weights: np.ndarray, name: str = "weights") -> None:
        """
        Check if the given sample-specific loss function or l-1/2 penalty weights
        are compatible with the input data.

        weights : np.ndarray of shape (n_obs,)
            Sample-specific KL-divergence or sparsity penalty loss weights

        name : str, default='weights'
            Name to be displayed in a potential error message
        """
        type_checker(name, weights, np.ndarray)
        shape_checker(name, weights, (self.adata.n_obs,))

        if not all(weights >= 0):
            raise ValueError(
                "Only non-negative KL-divergence and sparsity penalty weights "
                "are allowed."
            )

    def _setup_fitting_parameters(
        self,
        fitting_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if fitting_kwargs is None:
            fitting_kwargs = _DEFAULT_FITTING_KWARGS

        for kwarg in fitting_kwargs:
            if kwarg not in _FITTING_KWARGS:
                raise ValueError(
                    "The given fitting keyword arguments include parameters "
                    f"outside of {_FITTING_KWARGS}."
                )

        for name, weights in fitting_kwargs.items():
            if weights is not None:
                type_checker(name, weights, [float, int, list, np.ndarray])
                if type(weights) in [float, int]:
                    weights *= np.ones(self.adata.n_obs)

                if type(weights) is list:
                    weights = np.array(weights)

                self._check_weights(weights, name)

            setattr(self, name, weights)
