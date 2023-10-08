import multiprocessing
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

from ..utils import (
    differential_tail_test,
    kl_divergence,
    normalize_WH,
    poisson_llh,
    samplewise_kl_divergence,
)
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


class MvNMFHyperparameterSelector:
    """
    The volume-regularization weight is the only hyperparameter of mvNMF.
    This class implements methods to select the "optimal" volume-regularization weight.
    The framework of hyperparameter selectors allow to implement
    a denovo signature analysis in an NMF algorithm agnostic manner: A dictionary can
    be used to set all hyperparameters, irrespective of the NMF algorithm
    and its arbitrary number of hyperparameters.

    The best model is defined as the model with the strongest volume regularization
    such that the samplewise reconstruction errors are still (approximately) identically
    distributed to the model with the lowest volume regularization.
    The distributions are compared with the Mann-Whitney-U test.
    """

    # fmt: off
    default_lambda_tildes = (
        1e-10, 2e-10, 5e-10, 1e-9, 2e-9, 5e-9,
        1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7,
        1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
        1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3,
        1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1,
        1.0, 2.0,
    )
    # fmt: on

    def __init__(self, lambda_tildes=default_lambda_tildes, pthresh=0.05):
        """
        Inputs:
        ------
        lambda_tildes: tuple
            An ordered list of possible volume-regularization parameters.

        pthresh: float
            The distribution of samplewise reconstruction errors between two
            mvNMF fitted models is considered different if the Mann-Whitney-U
            test pvalue of comparing their, and their tail, distribution is lower
            than pthresh.
        """
        self.lambda_tildes = lambda_tildes
        self.pthresh = pthresh

        # initialize selection dependent attributes
        self.mvnmf_algorithm = None
        self.data = None
        self.given_signatures = None
        self.init_kwargs = None
        self.verbose = 0

    def _job(self, lambda_tilde):
        """
        Apply mvNMF for a single lambda_tilde volume regularization.
        """
        model = deepcopy(self.mvnmf_algorithm)
        model.lambda_tilde = lambda_tilde
        model.fit(
            data=self.data,
            given_signatures=self.given_signatures,
            init_kwargs=self.init_kwargs,
            verbose=0,
        )

        if self.verbose:
            print(f"mvNMF with lambda_tilde = {lambda_tilde:.2E} finished.")

        return model

    def _indicate_good_models(self, rerrors_base, rerrors_rest):
        """
        Compare the distributions of the samplewise baseline reconstruction errors
        and the samplewise model reconstruction errors.

        Output:
        -------
        indicators: np.ndarray
            One-dimensional boolean array indicating all models having
            samplewise reconstruction errors similar to the baseline errors.
        """
        n_models_rest = len(rerrors_rest)

        pvalue_indicators = np.empty(n_models_rest, dtype=bool)
        pvalue_tail_indicators = np.empty(n_models_rest, dtype=bool)

        for i, rerrors in enumerate(rerrors_rest):
            # Turn everything non-negative for the differential tail test.
            # Note: The Mann-Whitney U test statistic is shift-invariant
            shift = np.min([rerrors_base, rerrors])
            re_base = rerrors_base - shift
            re = rerrors - shift

            pvalue = stats.mannwhitneyu(re_base, re, alternative="less")[1]
            pvalue_indicators[i] = pvalue > self.pthresh

            pvalue_tail = differential_tail_test(
                re_base, re, percentile=90, alternative="less"
            )[1]
            pvalue_tail_indicators[i] = pvalue_tail > self.pthresh

        indicators = pvalue_indicators & pvalue_tail_indicators

        return indicators

    def _get_best_lambda_tilde(self, indicators):
        # np.argmin returns the first "bad" model index
        # Note: self.lambda_tildes[index] will be a "good" model because the number of
        # possible volume regularizations and the length of indicators differs by one.
        index = np.argmin(indicators)

        if all(indicators):
            index = len(indicators)
            warnings.warn(
                "For all lambda_tilde, the sample-wise reconstruction errors are "
                "comparable to the reconstruction errors with no regularization. "
                "The model with the strongest volume regularization is selected.",
                UserWarning,
            )

        if index == 0:
            warnings.warn(
                "The smallest lambda_tilde is selected. The optimal lambda_tilde "
                "might be smaller. We suggest to extend the grid to smaller "
                "lambda_tilde values to validate.",
                UserWarning,
            )

        best_lambda_tilde = self.lambda_tildes[index]

        return best_lambda_tilde

    def select(
        self,
        mvnmf_algorithm,
        data: pd.DataFrame,
        given_signatures=None,
        init_kwargs=None,
        ncpu=1,
        verbose=0,
    ):
        self.mvnmf_algorithm = mvnmf_algorithm
        self.data = data
        self.given_signatures = given_signatures
        self.init_kwargs = init_kwargs
        self.verbose = verbose

        if ncpu is None:
            ncpu = os.cpu_count()

        workers = multiprocessing.Pool(ncpu)
        models = workers.map(self._job, self.lambda_tildes)
        workers.close()
        workers.join()

        samplewise_rerrors_all = np.array(
            [model.samplewise_reconstruction_error for model in models]
        )
        rerrors_base, rerrors_rest = (
            samplewise_rerrors_all[0],
            samplewise_rerrors_all[1:],
        )

        indicators = self._indicate_good_models(rerrors_base, rerrors_rest)
        best_lambda_tilde = self._get_best_lambda_tilde(indicators)
        hyperparameters = {"lambda_tilde": best_lambda_tilde}

        return hyperparameters
