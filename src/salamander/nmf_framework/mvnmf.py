import numpy as np
import pandas as pd
from numba import njit

from ..utils import normalize_WH
from ._utils_klnmf import kl_divergence, poisson_llh, samplewise_kl_divergence, update_H
from .nmf import NMF

EPSILON = np.finfo(np.float32).eps


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


class MvNMF(NMF):
    """
    Min-volume non-negative matrix factorization. This algorithms is a volume-
    regularized version of NMF with the generalized Kullback-Leibler (KL)
    divergence.

    Parameters
    ----------
    n_signatures: int
        Number of signatures to decipher.

    init_method : str, default=nndsvd
        One of "custom", "flat", "hierarchical_cluster", "nndsvd",
        "nndsvda", "nndsvdar" "random" and "separableNMF". Please see the initialization
        module for further details on each method.

    lam : float, default=1.0
        Objective function volume penalty weight.

    delta : float, default=1.0
        Objective function hyperparameter.

    min_iterations : int, default=500
        Minimum number of iterations.

    max_iterations : int, default=10000
        Maximum number of iterations.

    tol : float, default=1e-7
        Tolerance of the stopping condition.

    Reference
    ---------
    Leplat, V., Gillis, N. and Ang, A.M., 2020.
    Blind audio source separation with minimum-volume beta-divergence NMF.
    IEEE Transactions on Signal Processing, 68, pp.3400-3410.
    """

    def __init__(
        self,
        n_signatures=1,
        init_method="nndsvd",
        lam=1.0,
        delta=1.0,
        min_iterations=500,
        max_iterations=10000,
        tol=1e-7,
    ):
        super().__init__(n_signatures, init_method, min_iterations, max_iterations, tol)
        self.lam = lam
        self.delta = delta
        self._gamma = None

    @property
    def reconstruction_error(self):
        return kl_divergence(self.X, self.W, self.H)

    @property
    def samplewise_reconstruction_error(self):
        return samplewise_kl_divergence(self.X, self.W, self.H)

    def objective_function(self):
        return kl_divergence_penalized(self.X, self.W, self.H, self.lam, self.delta)

    @property
    def objective(self) -> str:
        return "minimize"

    def loglikelihood(self) -> float:
        return poisson_llh(self.X, self.W, self.H)

    def _update_H(self):
        self.H = update_H(self.X, self.W, self.H)

    def _update_W_unconstrained(self):
        return update_W_unconstrained(
            self.X, self.W, self.H, self.lam, self.delta, self.n_given_signatures
        )

    def _line_search(self, W_unconstrained):
        self.W, self.H, self._gamma = line_search(
            self.X,
            self.W,
            self.H,
            self.lam,
            self.delta,
            self._gamma,
            W_unconstrained,
        )

    def _update_W(self):
        W_unconstrained = self._update_W_unconstrained()
        self._line_search(W_unconstrained)

    def fit(
        self,
        data: pd.DataFrame,
        given_signatures=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Input count data.

        given_signatures : pd.DataFrame, default=None
            Signatures to fix during the inference.

        init_kwargs: dict
            Any further keyword arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.

        history : bool, default=True
            If true, the objective function value of each iteration is saved.

        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._setup_data_parameters(data)
        self._initialize(given_signatures, init_kwargs)
        self._gamma = 1.0
        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 1000 == 0:
                print(f"iteration: {n_iteration}; objective: {of_values[-1]:.2f}")

            self._update_H()

            if self.n_given_signatures < self.n_signatures:
                self._update_W()

            prev_of_value = of_values[-1]
            of_values.append(self.objective_function())
            rel_change = (prev_of_value - of_values[-1]) / prev_of_value
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]

        return self
