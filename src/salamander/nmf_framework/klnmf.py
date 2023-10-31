import numpy as np
import pandas as pd

from ..utils import shape_checker, type_checker, value_checker
from . import _utils_klnmf
from .nmf import NMF

EPSILON = np.finfo(np.float32).eps


class KLNMF(NMF):
    """
    Decompose a mutation count matrix X into the product of a signature
    matrix W and an exposure matrix H by minimizing the generalized
    Kullback-Leibler (KL) divergence under the constraint of having
    normalized signatures.

    Parameters
    ----------
    n_signatures: int
        Number of signatures to decipher.

    init_method : str, default=nndsvd
        One of "custom", "flat", "hierarchical_cluster", "nndsvd",
        "nndsvda", "nndsvdar" "random" and "separableNMF". Please see the initialization
        module for further details on each method.

    update_method : str, default=mu-joint
        One of "mu-standard" and "mu-joint". The standard multiplicative
        update rules alternates between optimizing the signatures and exposures.
        The joint multiplicative update rule updates both matrices at once.
        It requires one matrix multiplication less and is faster.

    min_iterations : int, default=500
        Minimum number of iterations.

    max_iterations : int, default=10000
        Maximum number of iterations.

    tol : float, default=1e-7
        Tolerance of the stopping condition.

    Reference
    ---------
    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """

    def __init__(
        self,
        n_signatures=1,
        init_method="nndsvd",
        update_method="mu-joint",
        min_iterations=500,
        max_iterations=10000,
        tol=1e-7,
    ):
        super().__init__(n_signatures, init_method, min_iterations, max_iterations, tol)
        value_checker("update method", update_method, ["mu-standard", "mu-joint"])
        self.update_method = update_method
        self.weights_kl = None
        self.weights_l_half = None

    @property
    def reconstruction_error(self) -> float:
        """
        The unweighted Kullback-Leibler divergence.
        """
        return _utils_klnmf.kl_divergence(self.X, self.W, self.H)

    @property
    def samplewise_reconstruction_error(self) -> np.ndarray:
        """
        The unweighted samplewise Kullback-Leibler divergence.
        """
        return _utils_klnmf.samplewise_kl_divergence(self.X, self.W, self.H)

    def objective_function(self) -> float:
        """
        The sum of the (weighted) Kullback-Leibler divergence and the sparsity
        penalty.
        """
        of_value = _utils_klnmf.kl_divergence(self.X, self.W, self.H, self.weights_kl)

        if self.weights_l_half is not None:
            of_value += np.dot(self.weights_l_half, np.sum(np.sqrt(self.H), axis=0))

        return of_value

    @property
    def objective(self) -> str:
        return "minimize"

    def loglikelihood(self) -> float:
        return _utils_klnmf.poisson_llh(self.X, self.W, self.H)

    def _update_W(self):
        self.W = _utils_klnmf.update_W(
            self.X, self.W, self.H, self.weights_kl, self.n_given_signatures
        )

    def _update_H(self):
        self.H = _utils_klnmf.update_H(
            self.X, self.W, self.H, self.weights_kl, self.weights_l_half
        )

    def _update_WH(self):
        if self.update_method == "mu-standard":
            self._update_H()
            if self.n_given_signatures < self.n_signatures:
                self._update_W()
        else:
            self.W, self.H = _utils_klnmf.update_WH(
                self.X,
                self.W,
                self.H,
                self.weights_kl,
                self.weights_l_half,
                self.n_given_signatures,
            )

    def _check_weights(self, weights: np.ndarray, name: str = "weights"):
        """
        Check if the given sample-specific loss function or l-1/2 penalty weights
        are compatible with the input data.

        weights : np.ndarray of shape (n_samples,)
            Sample-specific KL-divergence or sparsity penalty loss weights

        name : str, default='weights'
            Name to be displayed in a potential error message
        """
        type_checker(name, weights, np.ndarray)
        shape_checker(name, weights, (self.n_samples,))

        if not all(weights >= 0):
            raise ValueError(
                "Only non-negative KL-divergence and sparsity penalty weights "
                "are allowed."
            )

    def _setup_weight_parameters(
        self, weights_kl: np.ndarray, weights_l_half: np.ndarray
    ):
        for weights, name in zip(
            [weights_kl, weights_l_half], ["weights_kl", "weights_l_half"]
        ):
            if type(weights) in [float, int]:
                weights *= np.ones(self.n_samples)

            if type(weights) is list:
                weights = np.array(weights)

            if weights is not None:
                self._check_weights(weights, name)

            setattr(self, name, weights)

    def fit(
        self,
        data: pd.DataFrame,
        given_signatures=None,
        weights_kl=None,
        weights_l_half=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        """
        Minimize the generalized Kullback-Leibler divergence D_KL(X || WH) between
        the mutation count matrix X and product of the signature matrix W and
        exposure matrix H under the constraint of normalized signatures.

        Parameters
        ----------
        data : pd.DataFrame
            The mutation count data

        given_signatures : pd.DataFrame, default=None
            Known signatures that should be fixed by the algorithm.
            The number of known signatures can be less or equal to the
            number of signatures specified in the algorithm instance.

        weights_kl : np.ndarray, default=None
            Per sample KL-divergence loss weights of shape (n_samples,).

        weights_l_half : np.ndarray, default=None
            Per sample l_half penalty weights of shape (n_samples,).
            They can be used to induce sparse exposures.

        init_kwargs : dict, default=None
            Any further keyword arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.

        history : bool, default=False
            If True, the objective function value will be stored after every
            iteration.

        verbose : int, default=0
            verbosity level

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._setup_data_parameters(data)
        self._setup_weight_parameters(weights_kl, weights_l_half)
        self._initialize(given_signatures, init_kwargs)
        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 1000 == 0:
                print(f"iteration: {n_iteration}; objective: {of_values[-1]:.2f}")

            self._update_WH()
            prev_of_value = of_values[-1]
            of_values.append(self.objective_function())
            rel_change = (prev_of_value - of_values[-1]) / prev_of_value
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]

        return self
