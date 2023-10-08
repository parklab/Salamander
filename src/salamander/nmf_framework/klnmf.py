import numpy as np
import pandas as pd

from ..utils import kl_divergence, normalize_WH, poisson_llh, samplewise_kl_divergence
from .nmf import NMF

EPSILON = np.finfo(np.float32).eps


class KLNMF(NMF):
    """
    Decompose a mutation count matrix X into the product of a signature
    matrix W and an exposure matrix H by using the generalized Kullback-Leibler (KL)
    loss induced multiplicative update rules derived by Lee and Seung
    in "Algorithms for non-negative matrix factorization".

    The class KLNMF is implemented as a child class of NMF to inherit
    its unified mutational signature analysis structure.
    """

    @property
    def reconstruction_error(self) -> float:
        return kl_divergence(self.X, self.W, self.H)

    @property
    def samplewise_reconstruction_error(self) -> np.ndarray:
        return samplewise_kl_divergence(self.X, self.W, self.H)

    def objective_function(self) -> float:
        return self.reconstruction_error

    @property
    def objective(self) -> str:
        return "minimize"

    def loglikelihood(self) -> float:
        return poisson_llh(self.X, self.W, self.H)

    def _update_W(self):
        """
        The multiplicative update rule of the signature matrix W
        derived by Lee and Seung. See Theorem 2 in
        "Algorithms for non-negative matrix factorization".

        Clipping the matrix avoids floating point errors.
        """
        self.W *= (self.X / (self.W @ self.H)) @ self.H.T
        self.W /= np.sum(self.H, axis=1)
        self.W = self.W.clip(EPSILON)

    def _update_H(self):
        """
        The multiplicative update rule of the exposure matrix H
        derived by Lee and Seung. See Theorem 2 in
        "Algorithms for non-negative matrix factorization".

        Clipping the matrix avoids floating point errors.
        """
        self.H *= self.W.T @ (self.X / (self.W @ self.H))
        self.H /= np.sum(self.W, axis=0)[:, np.newaxis]
        self.H = self.H.clip(EPSILON)

    def fit(
        self,
        data: pd.DataFrame,
        given_signatures=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        """
        Minimize the generalized Kullback-Leibler divergence D_KL(X || WH) between
        the mutation count matrix X and product of the signature matrix W and
        exposure matrix H by altering the multiplicative update steps for W and H.

        Input:
        ------
        data: pd.DataFrame
            The mutation count data

        given_signatures: pd.DataFrame, default=None
            In the case of refitting, a priori known signatures have to be provided. The
            number of signatures has to match to the NMF object and the mutation type
            names have to match to the mutation count matrix

        init_kwargs: dict
            Any further keyword arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.

        history: bool
            When set to true, the history of the objective function
            will be stored in a dictionary.

        verbose: int
            Every 100th iteration number will be printed when set unequal to zero.
        """
        self._setup_data_parameters(data)
        self._initialize(given_signatures, init_kwargs)
        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 100 == 0:
                print(f"iteration {n_iteration}")

            self._update_H()

            if given_signatures is None:
                self._update_W()

            self.W, self.H = normalize_WH(self.W, self.H)
            self.W, self.H = self.W.clip(EPSILON), self.H.clip(EPSILON)

            prev_of_value = of_values[-1]
            of_values.append(self.objective_function())
            rel_change = (prev_of_value - of_values[-1]) / prev_of_value
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]

        return self
