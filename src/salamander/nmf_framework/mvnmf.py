import numpy as np
import pandas as pd

from ..utils import kl_divergence, normalize_WH, poisson_llh, samplewise_kl_divergence
from .nmf import NMF

EPSILON = np.finfo(np.float32).eps


class MvNMF(NMF):
    """
    Min-volume non-negative matrix factorization. Based on Algorithm 1 of

    Leplat, V., Gillis, N. and Ang, A.M., 2020.
    Blind audio source separation with minimum-volume beta-divergence NMF.
    IEEE Transactions on Signal Processing, 68, pp.3400-3410.

    Input:
    ------
    n_signatures: int
        Number of signatures to decipher.

    init_method: str
        One of "custom", "flat", "hierarchical_cluster", "nndsvd",
        "nndsvda", "nndsvdar" "random" and "separableNMF". Please see the initialization
        module for further details on each method.

    lambda_tilde : float
        Objective function hyperparameter.

    delta : float
        Objective function hyperparameter.

    min_iterations : int, default=200
        Minimum number of iterations.

    max_iterations : int, default=400
        Maximum number of iterations.

    tol : float, default=1e-7
        Tolerance of the stopping condition.

    Note:
    -----
        The algorithm should work better when the initial guesses are better.
        One reason lies in lambda and lambda_tilde.
        Lambda is calculated in a way such that the two terms
        in the objective function are comparable. Ideally, lambda should be set to
        kl_divergence(X, W_true @ H_true)/abs(volume(W_true)) * lambda_tilde.
        In our code, the true W and H are replaced by the initial guesses.
        So if the initial guesses are good, then indeed the two terms will be
        comparable. If the initial guesses are far off, then the kl_divergence
        part will be far over-estimated. As a result, the two terms are
        not comparable anymore. One potential improvement is to first run
        a small number of NMF iterations, and then use the NMF results
        as hot starts for the mvNMF algorithm.
    """

    def __init__(
        self,
        n_signatures=1,
        init_method="nndsvd",
        lambda_tilde=1e-5,
        delta=1.0,
        min_iterations=500,
        max_iterations=10000,
        tol=1e-7,
    ):
        super().__init__(n_signatures, init_method, min_iterations, max_iterations, tol)
        self.lambda_tilde = lambda_tilde
        self.lam = lambda_tilde
        self.delta = delta
        self.gamma = 1.0

    @property
    def reconstruction_error(self):
        return kl_divergence(self.X, self.W, self.H)

    @property
    def samplewise_reconstruction_error(self):
        return samplewise_kl_divergence(self.X, self.W, self.H)

    @staticmethod
    def _volume_logdet(W, delta) -> float:
        n_signatures = W.shape[1]
        diag = delta * np.identity(n_signatures)
        volume = np.log(np.linalg.det(W.T @ W + diag))

        return volume

    @staticmethod
    def _objective_function(
        X: np.ndarray, W: np.ndarray, H: np.ndarray, lam: float, delta: float
    ) -> float:
        reconstruction_error = kl_divergence(X, W, H)
        volume = MvNMF._volume_logdet(W, delta)
        loss = reconstruction_error + lam * volume

        return loss

    def objective_function(self):
        return self._objective_function(self.X, self.W, self.H, self.lam, self.delta)

    @property
    def objective(self) -> str:
        return "minimize"

    def loglikelihood(self) -> float:
        return poisson_llh(self.X, self.W, self.H)

    def _update_H(self):
        self.H *= self.W.T @ (self.X / (self.W @ self.H))
        self.H /= np.sum(self.W, axis=0)[:, np.newaxis]
        self.H = self.H.clip(EPSILON)

    def _update_W_unconstrained(self):
        diag = np.diag(np.full(self.n_signatures, self.delta))
        Y = np.linalg.inv(self.W.T @ self.W + diag)

        Y_minus = np.maximum(0, -Y)
        Y_abs = np.abs(Y)

        WY_minus = self.W @ Y_minus
        WY_abs = self.W @ Y_abs

        rowsums_H = np.sum(self.H, axis=1)

        discriminant_s1 = (rowsums_H - 4 * self.lam * WY_minus) ** 2
        discriminant_s2 = (
            8 * self.lam * WY_abs * ((self.X / (self.W @ self.H)) @ self.H.T)
        )

        numerator_s1 = np.sqrt(discriminant_s1 + discriminant_s2)
        numerator_s2 = -rowsums_H + 4 * self.lam * WY_minus
        numerator = numerator_s1 + numerator_s2

        denominator = 4 * self.lam * WY_abs

        W_uc = self.W * numerator / denominator
        W_uc = W_uc.clip(EPSILON)

        return W_uc

    def _line_search(self, W_uc, loss_prev):
        W_new = self.W + self.gamma * (W_uc - self.W)
        W_new, H_new = normalize_WH(W_new, self.H)
        W_new, H_new = W_new.clip(EPSILON), H_new.clip(EPSILON)

        loss = self._objective_function(self.X, W_new, H_new, self.lam, self.delta)

        while loss > loss_prev and self.gamma > 1e-16:
            self.gamma *= 0.8

            W_new = self.W + self.gamma * (W_uc - self.W)
            W_new, H_new = normalize_WH(W_new, self.H)
            W_new, H_new = W_new.clip(EPSILON), H_new.clip(EPSILON)

            loss = self._objective_function(self.X, W_new, H_new, self.lam, self.delta)

        self.gamma = min(1.0, 2 * self.gamma)
        self.W, self.H = W_new, H_new

    # pylint: disable-next=W0221
    def _update_W(self, loss_prev):
        W_uc = self._update_W_unconstrained()
        self._line_search(W_uc, loss_prev)

    def _initialize_mvnmf_parameters(self):
        # lambda is chosen s.t. both loss summands
        # approximately contribute equally for lambda_tilde = 1
        init_reconstruction_error = self.reconstruction_error
        init_volume = self._volume_logdet(self.W, self.delta)
        self.lam = self.lambda_tilde * init_reconstruction_error / abs(init_volume)
        self.gamma = 1.0

    def fit(
        self,
        data: pd.DataFrame,
        given_signatures=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        """
        Input:
        ------
        data : array-like of shape (n_features, n_samples)
            The mutation count data.

        init_kwargs: dict
            Any further keyword arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.

        verbose : int, default=0
            Verbosity level.
        """
        self._setup_data_parameters(data)
        self._initialize(given_signatures, init_kwargs)
        self._initialize_mvnmf_parameters()

        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 100 == 0:
                print(f"iteration {n_iteration}")

            self._update_H()
            prev_of_value = of_values[-1]

            if given_signatures is None:
                self._update_W(prev_of_value)

            of_values.append(self.objective_function())
            rel_change = (prev_of_value - of_values[-1]) / prev_of_value
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]

        return self
