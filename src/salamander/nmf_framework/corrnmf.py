from abc import abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.special import gammaln

from ..utils import match_signatures_pair, shape_checker, type_checker
from ._utils_klnmf import kl_divergence, poisson_llh, samplewise_kl_divergence
from .initialization import initialize
from .signature_nmf import SignatureNMF

EPSILON = np.finfo(np.float32).eps


class CorrNMF(SignatureNMF):
    r"""
    The abstract class CorrNMF unifies the structure of deterministic and
    stochastic correlated NMF (CorrNMF) with and without given signatures.
    Both variants of CorrNMF have an identical generative model and objective function.
    The model parameters are the sample biases \alpha, variance \sigma^2,
    signature matrix W and the auxiliary parameters p.
    The latent variables are the signature embeddings L and the sample embeddings U.

    Overview:

    Every child class has to implement the following methods:

        - _update_alpha:
            update the sample exposure biases \alpha

        - _update_beta:
            update the signature exposure biases \beta

        - _update_sigma_sq:
            update the embedding distribution variance \sigma^2

        - _update_W:
            update the signature matrix W

        - _update_p:
            update the auxiliary parameters p

        - _update_l:
            update a single signature embedding l

        - _update_u:
            update a single sample embedding u

        - fit:
            Run CorrNMF for a given mutation count data. Every
            fit method should also implement a version that allows fixing
            arbitrary many a priori known signatures.


    The following attributes are implemented in the abstract class CNMF:

        - signatures: pd.DataFrame
            The signature matrix including mutation type names and signature names

        - exposures: pd.DataFrame
            The exposure matrix including the signature names and sample names

        - reconstruction_error: float
            The reconstruction error between the count matrix
            and the reconstructed count matrix.

        - samplewise_reconstruction_error: np.ndarray
            The samplewise reconstruction error between the sample counts
            and the reconstructed sample counts.

        - _n_parameters:
            The number of parameters fitted in CorrNMF

        - objective: str
            "minimize" or "maximize". Whether the NMF algorithm maximizes
            or minimizes the objective function. Some algorithms maximize a likelihood,
            others minimize a distance. The distinction is useful for filtering NMF
            runs based on the fitted objective function value.

        - corr_signatures: pd.DataFrame
            The signature correlation matrix induced by the signature embeddings

        - corr_samples: pd.DataFrame
            The sample correlation matrix induced by the sample embeddings


    The following methods are implemented in the abstract class CorrNMF:

        - objective_function:
            The evidence lower bound (ELBO) of the log-likelihood.
            Note: The ELBO is sometimes called the variational lower bound.

        - _surrogate_objective_function:
            A surrogate lower bound of the ELBO after introducing the
            auxiliary parameters p. In contrast to the original objective_function,
            the surrogate is strictly convex in the signature and sample embeddings

        - loglikelihood:
            The loglikelihood of the underyling generative model

        - _initialize:
            Initialize all model parameters and latent variables depending on the
            initialization method chosen

        - _get_embedding_annotations:
            A helper function to concatenate signature and sample names

        - plot_embeddings:
            Plot signature or sample embeddings in 2D using PCA, tSNE or UMAP.
            The respective plotting functions are implemented in the plot.py module

    More specific docstrings are written for the respective attributes and methods.
    """

    def __init__(
        self,
        n_signatures=1,
        dim_embeddings=None,
        init_method="nndsvd",
        min_iterations=500,
        max_iterations=10000,
        tol=1e-7,
    ):
        """
        Input:
        ------
        n_signatures: int
            The number of underlying signatures that are assumed to
            have generated the mutation count data

        dim_embeddings: int
            The assumed dimension of the signature and sample embeddings.
            Should be smaller or equal to the number of signatures as a dimension
            equal to the number of signatures covers the case of independent
            signatures. The smaller the embedding dimension, the stronger the
            enforced correlation structure  on both signatures and samples.

        init_method: str
            One of "custom", "flat", "hierarchical_cluster", "nndsvd",
            "nndsvda", "nndsvdar" "random" and "separableNMF".
            See the initialization module for further details.

        min_iterations: int
            The minimum number of iterations to perform during inference

        max_iterations: int
            The maximum number of iterations to perform during inference

        tol: float
            The CorrNMF algorithm is converged when the relative change of the
            surrogate objective function of one iteration is smaller
            than the tolerance 'tol'.
        """
        super().__init__(n_signatures, init_method, min_iterations, max_iterations, tol)

        if dim_embeddings is None:
            dim_embeddings = n_signatures

        self.dim_embeddings = dim_embeddings

        # initialize data/fitting-dependent attributes
        self.W = None
        self.alpha = None
        self.beta = None
        self.L = None
        self.U = None
        self.sigma_sq = None

    @property
    def signatures(self) -> pd.DataFrame:
        signatures = pd.DataFrame(
            self.W, index=self.mutation_types, columns=self.signature_names
        )
        return signatures

    @property
    def exposures(self) -> pd.DataFrame:
        """
        In contrast to the classical NMF framework, the exposure matrix is
        restructured and determined by the signature & sample biases and
        embeddings.
        """
        exposures = pd.DataFrame(
            np.exp(self.alpha + self.beta[:, np.newaxis] + self.L.T @ self.U),
            index=self.signature_names,
            columns=self.sample_names,
        )
        return exposures

    @property
    def _n_parameters(self):
        """
        There are n_features * n_signatures parameters corresponding to
        the signature matrix, each embedding corresponds to dim_embeddings parameters,
        and each signature & sample has a bias.
        Finally, the model variance is a single positive real number.

        Note: We do not include the number of auxiliary parameters p.
        """
        n_parameters_signatures = self.n_features * self.n_signatures
        n_parameters_embeddings = self.dim_embeddings * (
            self.n_signatures + self.n_samples
        )
        n_parameters_biases = self.n_samples + self.n_signatures
        n_parameters_exposures = n_parameters_embeddings + n_parameters_biases
        n_parameters = n_parameters_signatures + n_parameters_exposures + 1

        return n_parameters

    @property
    def reconstruction_error(self):
        return kl_divergence(self.X, self.W, self.exposures.values)

    @property
    def samplewise_reconstruction_error(self):
        return samplewise_kl_divergence(self.X, self.W, self.exposures.values)

    def objective_function(self, penalize_sample_embeddings=True) -> float:
        """
        The evidence lower bound (ELBO)
        """
        elbo = poisson_llh(self.X, self.signatures.values, self.exposures.values)
        elbo -= (
            0.5
            * self.dim_embeddings
            * self.n_signatures
            * np.log(2 * np.pi * self.sigma_sq)
        )
        elbo -= np.sum(self.L**2) / (2 * self.sigma_sq)

        if penalize_sample_embeddings:
            elbo -= (
                0.5
                * self.dim_embeddings
                * self.n_samples
                * np.log(2 * np.pi * self.sigma_sq)
            )
            elbo -= np.sum(self.U**2) / (2 * self.sigma_sq)

        return elbo

    @property
    def objective(self) -> str:
        return "maximize"

    def _surrogate_objective_function(self, penalize_sample_embeddings=True) -> float:
        """
        The surrogate lower bound of the ELBO.
        """
        p = self._update_p()
        exposures = self.exposures.values
        aux = np.log(self.W)[:, :, None] + np.log(exposures)[None, :, :] - np.log(p)
        sof_value = np.einsum("VD,VKD,VKD->", self.X, p, aux, optimize="greedy").item()
        sof_value -= np.sum(gammaln(1 + self.X))
        sof_value -= np.sum(exposures)
        sof_value -= (
            0.5
            * self.dim_embeddings
            * self.n_signatures
            * np.log(2 * np.pi * self.sigma_sq)
        )
        sof_value -= np.sum(self.L**2) / (2 * self.sigma_sq)

        if penalize_sample_embeddings:
            sof_value -= (
                0.5
                * self.dim_embeddings
                * self.n_samples
                * np.log(2 * np.pi * self.sigma_sq)
            )
            sof_value -= np.sum(self.U**2) / (2 * self.sigma_sq)

        return sof_value

    def loglikelihood(self):
        return self.objective_function()

    @abstractmethod
    def _update_alpha(self):
        pass

    @abstractmethod
    def _update_sigma_sq(self):
        pass

    @abstractmethod
    def _update_W(self):
        pass

    @abstractmethod
    def _update_p(self):
        pass

    @abstractmethod
    def _update_l(self, index, aux_row, outer_prods_U):
        r"""
        Input:
        ------
        index: int
            The index of the signature whose embedding is updated

        aux_row: nd.ndarray
            Row of the following matrix:
            aux_kd = \sum_v X_vd * p_vkd.
            This auxiliary matrix is used for updating the signatures
            and the sample embeddidngs. The aux_row argument
            is the k-th row of aux, where k is equal to 'index'.

        outer_prods_U: np.ndarray
            All outer products of the sample embeddings.
            shape: (n_samples, dim_embeddings, dim_embeddings)
        """

    @abstractmethod
    def _update_u(self, index, aux_col, outer_prods_L):
        r"""
        Input:
        ------
        index: int
            The index of the sample whose embedding is updated

        aux_col: nd.ndarray
            Column of the following matrix:
            aux_kd = \sum_v X_vd * p_vkd.
            This auxiliary matrix is used for updating the signatures
            and the sample embeddidngs. The aux_col argument
            is the d-th row of aux, where d is equal to 'index'.

        outer_prods_L: np.ndarray
            All outer products of the signature embeddings.
            shape: (n_signatures, dim_embeddings, dim_embeddings)
        """

    def _check_given_biases(self, given_biases, expected_n_biases, name):
        type_checker(name, given_biases, np.ndarray)
        shape_checker(name, given_biases, (expected_n_biases,))

    def _check_given_embeddings(self, given_embeddings, expected_n_embeddings, name):
        type_checker(name, given_embeddings, np.ndarray)
        shape_checker(
            name, given_embeddings, (self.dim_embeddings, expected_n_embeddings)
        )

    def _check_given_parameters(
        self,
        given_signatures,
        given_signature_biases,
        given_signature_embeddings,
        given_sample_biases,
        given_sample_embeddings,
    ):
        if given_signatures is not None:
            self._check_given_signatures(given_signatures)

        if given_signature_biases is not None:
            self._check_given_biases(
                given_signature_biases, self.n_signatures, "given_signature_biases"
            )

        if given_signature_embeddings is not None:
            self._check_given_embeddings(
                given_signature_embeddings,
                self.n_signatures,
                "given_signature_embeddings",
            )

        if given_sample_biases is not None:
            self._check_given_biases(
                given_sample_biases, self.n_samples, "given_sample_biases"
            )

        if given_sample_embeddings is not None:
            self._check_given_embeddings(
                given_sample_embeddings, self.n_samples, "given_sample_embeddings"
            )

    def _initialize(
        self,
        given_signatures=None,
        given_signature_biases=None,
        given_signature_embeddings=None,
        given_sample_biases=None,
        given_sample_embeddings=None,
        init_kwargs=None,
    ):
        """
        Initialize the signature matrix W, sample biases alpha, signature biases beta,
        the squared variance, and the signature and sample embeddings.
        The signatures or signature embeddings can also be provided by the user.

        Parameters
        ----------
        given_signatures: pd.DataFrame, default=None
            A priori known signatures. The number of given signatures has
            to be less or equal to the number of signatures of NMF
            algorithm instance, and the mutation type names have to match
            the mutation types of the count data.

        given_signature_biases : np.ndarray, default=None
            Known signature biases of shape (n_signatures,) that will be fixed
            during model fitting.

        given_signature_embeddings : np.ndarray, default=None
            A priori known signature embeddings of shape (dim_embeddings, n_signatures).

        given_sample_biases : np.ndarray, default=None
            Known sample biases of shape (n_samples,) that will be fixed
            during model fitting.

        given_sample_embeddings : np.ndarray, default=None
            A priori known sample embeddings of shape (dim_embeddings, n_samples).

        init_kwargs : dict
            Any further keyword arguments to pass to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.
        """
        self._check_given_parameters(
            given_signatures,
            given_signature_biases,
            given_signature_embeddings,
            given_sample_biases,
            given_sample_embeddings,
        )

        if given_signatures is not None:
            self.n_given_signatures = len(given_signatures.columns)
        else:
            self.n_given_signatures = 0

        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.W, _, self.signature_names = initialize(
            self.X, self.n_signatures, self.init_method, given_signatures, **init_kwargs
        )
        self.sigma_sq = 1.0

        if given_signature_biases is None:
            self.beta = np.zeros(self.n_signatures)
        else:
            self.beta = given_signature_biases

        if given_signature_embeddings is None:
            self.L = np.random.multivariate_normal(
                np.zeros(self.dim_embeddings),
                np.identity(self.dim_embeddings),
                size=self.n_signatures,
            ).T
        else:
            self.L = given_signature_embeddings

        if given_sample_biases is None:
            self.alpha = np.zeros(self.n_samples)
        else:
            self.alpha = given_sample_biases

        if given_sample_embeddings is None:
            self.U = np.random.multivariate_normal(
                np.zeros(self.dim_embeddings),
                np.identity(self.dim_embeddings),
                size=self.n_samples,
            ).T
        else:
            self.U = given_sample_embeddings

    @property
    def corr_signatures(self) -> pd.DataFrame:
        norms = np.sqrt(np.sum(self.L**2, axis=0))

        corr_vector = np.array(
            [
                np.dot(l1, l2) / (norms[k1] * norms[k1 + k2 + 1])
                for k1, l1 in enumerate(self.L.T)
                for k2, l2 in enumerate(self.L[:, (k1 + 1) :].T)
            ]
        )
        corr_matrix = squareform(corr_vector) + np.identity(self.n_signatures)
        corr = pd.DataFrame(
            corr_matrix, index=self.signature_names, columns=self.signature_names
        )

        return corr

    @property
    def corr_samples(self) -> pd.DataFrame:
        norms = np.sqrt(np.sum(self.U**2, axis=0))

        corr_vector = np.array(
            [
                np.dot(u1, u2) / (norms[d1] * norms[d1 + d2 + 1])
                for d1, u1 in enumerate(self.U.T)
                for d2, u2 in enumerate(self.U[:, (d1 + 1) :].T)
            ]
        )
        corr_matrix = squareform(corr_vector) + np.identity(self.n_samples)
        corr = pd.DataFrame(
            corr_matrix, index=self.sample_names, columns=self.sample_names
        )

        return corr

    def reorder(self, other_signatures, metric="cosine", keep_names=False):
        reordered_indices = match_signatures_pair(
            other_signatures, self.signatures, metric=metric
        )
        self.W = self.W[:, reordered_indices]
        self.beta = self.beta[reordered_indices]
        self.L = self.L[:, reordered_indices]

        if keep_names:
            self.signature_names = self.signature_names[reordered_indices]

        return reordered_indices

    def _get_embedding_data(self) -> np.ndarray:
        """
        In CorrNMF, the data for the embedding plot are the (transpoed) signature and
        sample embeddings.
        """
        return np.concatenate([self.L, self.U], axis=1).T.copy()

    def _get_default_embedding_annotations(self) -> np.ndarray:
        """
        The embedding plot defaults to annotating the signature embeddings.
        """
        # Only annotate with the first 20 characters of names
        annotations = np.empty(self.n_signatures + self.n_samples, dtype="U20")
        annotations[: self.n_signatures] = self.signature_names

        return annotations
