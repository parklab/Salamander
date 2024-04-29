from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numba import njit

from ..initialization.initialize import EPSILON
from ..utils import normalize_WH
from ._utils_klnmf import kl_divergence, samplewise_kl_divergence, update_H
from .standard_nmf import StandardNMF

if TYPE_CHECKING:
    from ..initialization.methods import _Init_methods

_DEFAULT_FITTING_KWARGS = None


@njit
def volume_logdet(W: np.ndarray, delta: float) -> float:
    n_signatures = W.shape[1]
    diag = np.diag(np.full(n_signatures, delta))
    volume = np.log(np.linalg.det(W.T @ W + diag))
    return volume


@njit
def kl_divergence_penalized(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, lam: float, delta: float
) -> float:
    reconstruction_error = kl_divergence(X, W, H)
    volume = volume_logdet(W, delta)
    loss = reconstruction_error + lam * volume
    return loss


@njit
def update_W_unconstrained(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    lam: float,
    delta: float,
    n_given_signatures: int = 0,
) -> np.ndarray:
    n_signatures = W.shape[1]
    diag = np.diag(np.full(n_signatures, delta))
    Y = np.linalg.inv(W.T @ W + diag)
    Y_minus = np.maximum(0, -Y)
    Y_abs = np.abs(Y)
    WY_minus = W @ Y_minus
    WY_abs = W @ Y_abs

    rowsums_H = np.sum(H, axis=1)
    discriminant_s1 = (rowsums_H - 4 * lam * WY_minus) ** 2
    discriminant_s2 = 8 * lam * WY_abs * ((X / (W @ H)) @ H.T)
    numerator_s1 = np.sqrt(discriminant_s1 + discriminant_s2)
    numerator_s2 = -rowsums_H + 4 * lam * WY_minus
    numerator = numerator_s1 + numerator_s2
    denominator = 4 * lam * WY_abs
    W_unconstrained = W * numerator / denominator
    W_unconstrained[:, :n_given_signatures] = W[:, :n_given_signatures].copy()
    W_unconstrained[:, n_given_signatures:] = W_unconstrained[
        :, n_given_signatures:
    ].clip(EPSILON)
    return W_unconstrained


@njit
def line_search(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    lam: float,
    delta: float,
    gamma: float,
    W_unconstrained: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    prev_of_value = kl_divergence_penalized(X, W, H, lam, delta)
    W_new, H_new = normalize_WH(W_unconstrained, H)
    W_new, H_new = W_new.clip(EPSILON), H_new.clip(EPSILON)
    of_value = kl_divergence_penalized(X, W_new, H_new, lam, delta)

    while of_value > prev_of_value and gamma > 1e-16:
        gamma *= 0.8
        W_new = (1 - gamma) * W + gamma * W_unconstrained
        W_new, H_new = normalize_WH(W_new, H)
        W_new, H_new = W_new.clip(EPSILON), H_new.clip(EPSILON)
        of_value = kl_divergence_penalized(X, W_new, H_new, lam, delta)

    gamma = min(1.0, 1.2 * gamma)
    return W_new, H_new, gamma


class MvNMF(StandardNMF):
    """
    Min-volume non-negative matrix factorization. This algorithms is a volume-
    regularized version of NMF with the generalized Kullback-Leibler (KL)
    divergence.

    Parameters
    ----------
    lam : float, default=1.0
        Objective function volume penalty weight.

    delta : float, default=1.0
        Objective function hyperparameter, see equation (4) in reference.

    Reference
    ---------
    Leplat, V., Gillis, N. and Ang, A.M., 2020.
    Blind audio source separation with minimum-volume beta-divergence NMF.
    IEEE Transactions on Signal Processing, 68, pp.3400-3410.
    """

    def __init__(
        self,
        n_signatures: int = 1,
        init_method: _Init_methods = "nndsvd",
        lam: float = 1.0,
        delta: float = 1.0,
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
        self.lam = lam
        self.delta = delta
        self._gamma = 1.0

    def compute_reconstruction_errors(self) -> None:
        """
        Add the samplewise Kullback-Leibler divergences
        as observation annotations to the AnnData count data.
        """
        errors = samplewise_kl_divergence(
            self.adata.X.T, self.asignatures.X.T, self.adata.obsm["exposures"].T
        )
        self.adata.obs["reconstruction_error"] = errors

    def objective_function(self) -> float:
        return kl_divergence_penalized(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            self.lam,
            self.delta,
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"

    def _update_H(self):
        self.adata.obsm["exposures"] = update_H(
            self.adata.X.T, self.asignatures.X.T, self.adata.obsm["exposures"].T
        ).T

    def _update_W_unconstrained(self, n_given_signatures: int = 0) -> np.ndarray:
        return update_W_unconstrained(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            self.lam,
            self.delta,
            n_given_signatures,
        )

    def _line_search(self, W_unconstrained: np.ndarray) -> None:
        W, H, self._gamma = line_search(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            self.lam,
            self.delta,
            self._gamma,
            W_unconstrained,
        )
        self.asignatures.X = W.T
        self.adata.obsm["exposures"] = H.T

    def _update_W(self, n_given_signatures: int = 0) -> None:
        if n_given_signatures == self.n_signatures:
            return

        W_unconstrained = self._update_W_unconstrained(n_given_signatures)
        self._line_search(W_unconstrained)

    def _update_parameters(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        self._update_H()

        if "asignatures" in given_parameters:
            n_given_signatures = given_parameters["asignatures"].n_obs
        else:
            n_given_signatures = 0

        self._update_W(n_given_signatures)

    def _setup_fitting_parameters(
        self, fitting_kwargs: dict[str, Any] | None = None
    ) -> None:
        if fitting_kwargs is None:
            fitting_kwargs = _DEFAULT_FITTING_KWARGS  # still None

        self._gamma = 1.0
