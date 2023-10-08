import multiprocessing
import os
import warnings
from abc import abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.special import gammaln

from ..plot import pca_2d, scatter_1d, scatter_2d, tsne_2d, umap_2d
from ..utils import (
    kl_divergence,
    match_signatures_pair,
    poisson_llh,
    samplewise_kl_divergence,
    shape_checker,
    type_checker,
    value_checker,
)
from .initialization import (
    init_custom,
    init_flat,
    init_nndsvd,
    init_random,
    init_separableNMF,
)
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
            fit method should also implement a "refitting version", where the signatures
            W are known in advance and fixed


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
        update_W="1999-Lee",
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

        update_W: str, "1999-Lee" or "surrogate"
            The signature matrix W can be inferred by either using the Lee and Seung
            multiplicative update rules to optimize the objective function or by
            maximizing the surrogate objective function.

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
        value_checker("update_W", update_W, ["1999-Lee", "surrogate"])
        self.update_W = update_W

        # initialize data/fitting dependent attributes
        self.W = None
        self.alpha = None
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
        restructured and determined by the signature and sample embeddings.
        """
        exposures = pd.DataFrame(
            np.exp(np.tile(self.alpha, (self.n_signatures, 1)) + self.L.T @ self.U),
            index=self.signature_names,
            columns=self.sample_names,
        )

        return exposures

    @property
    def _n_parameters(self):
        """
        There are n_features * n_signatures parameters corresponding to
        the signature matrix, each embedding corresponds to dim_embeddings parameters,
        and each sample has a bias parameter.
        Finally, the model variance is a single positive real number.

        Note: We do not include the number of auxiliary parameters p.
        """
        n_parameters_signatures = self.n_features * self.n_signatures
        n_parameters_embeddings = self.dim_embeddings * (
            self.n_signatures + self.n_samples
        )
        n_parameters_biases = self.n_samples
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

    def _surrogate_objective_function(
        self, p, penalize_sample_embeddings=True
    ) -> float:
        """
        The surrogate lower bound of the ELBO after
        introducing the auxiliary parameters p.
        """
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
    def _update_W(self, p):
        """
        Input:
        ------
        p: np.ndarray
            The auxiliary parameters of CorrNMF
        """

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

    def _check_given_signature_embeddings(self, given_signature_embeddings: np.ndarray):
        type_checker("signature embeddings", given_signature_embeddings, np.ndarray)
        shape_checker(
            "given_signature_embeddings",
            given_signature_embeddings,
            (self.dim_embeddings, self.n_signatures),
        )

    def _check_given_sample_embeddings(self, given_sample_embeddings: np.ndarray):
        type_checker("sample embeddings", given_sample_embeddings, np.ndarray)
        shape_checker(
            "given_sample_embeddings",
            given_sample_embeddings,
            (self.dim_embeddings, self.n_samples),
        )

    def _initialize(
        self,
        given_signatures=None,
        given_signature_embeddings=None,
        given_sample_embeddings=True,
        init_kwargs=None,
    ):
        """
        Initialize the signature matrix W, sample biases alpha, the squared variance,
        and the signature and sample embeddings.
        The signatures or signature embeddings can also be provided by the user.

        Input:
        ------
        init_kwargs: dict
            Any further arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.
        """
        if given_signatures is not None:
            self._check_given_signatures(given_signatures)

        if given_signature_embeddings is not None:
            self._check_given_signature_embeddings(given_signature_embeddings)

        if given_sample_embeddings is not None:
            self._check_given_sample_embeddings(given_sample_embeddings)

        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()

        if self.init_method == "custom":
            self.W, _ = init_custom(self.X, self.n_signatures, **init_kwargs)

        elif self.init_method == "flat":
            self.W, _ = init_flat(self.X, self.n_signatures)

        elif self.init_method in ["nndsvd", "nndsvda", "nndsvdar"]:
            self.W, _ = init_nndsvd(
                self.X, self.n_signatures, init=self.init_method, **init_kwargs
            )

        elif self.init_method == "random":
            self.W, _ = init_random(self.X, self.n_signatures, **init_kwargs)

        else:
            self.W = init_separableNMF(self.X, self.n_signatures)

        if given_signatures is not None:
            self.W = given_signatures.copy().values
            self.signature_names = given_signatures.columns.to_numpy(dtype=str)

        self.W /= np.sum(self.W, axis=0)
        self.W = self.W.clip(EPSILON)
        self.alpha = np.zeros(self.n_samples, dtype=float)
        self.sigma_sq = 1.0
        self.L = np.random.multivariate_normal(
            np.zeros(self.dim_embeddings),
            np.identity(self.dim_embeddings),
            size=self.n_signatures,
        ).T
        self.U = np.random.multivariate_normal(
            np.zeros(self.dim_embeddings),
            np.identity(self.dim_embeddings),
            size=self.n_samples,
        ).T

        if given_signature_embeddings is not None:
            self.L = given_signature_embeddings

        if given_sample_embeddings is not None:
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
        self.L = self.L[:, reordered_indices]

        if keep_names:
            self.signature_names = self.signature_names[reordered_indices]

        return reordered_indices

    def _get_embedding_annotations(self, annotate_signatures, annotate_samples):
        # Only annotate with the first 20 characters of names
        annotations = np.empty(self.n_signatures + self.n_samples, dtype="U20")

        if annotate_signatures:
            annotations[: self.n_signatures] = self.signature_names

        if annotate_samples:
            annotations[-self.n_samples :] = self.sample_names

        return annotations

    def plot_embeddings(
        self,
        method="umap",
        annotate_signatures=True,
        annotate_samples=False,
        annotation_kwargs=None,
        normalize=False,
        ax=None,
        outfile=None,
        **kwargs,
    ):
        """
        Plot the signature and sample embeddings. If the embedding dimension is two,
        the embeddings will be plotted directly, ignoring the chosen method.
        See plot.py for the implementation of scatter_2d, tsne_2d, pca_2d, umap_2d.

        Input:
        ------
        methdod: str
            Either 'tsne', 'pca' or 'umap'. The respective dimensionality reduction
            will be applied to plot the signature and sample embeddings in 2D.

        annotate_signatures: bool

        annotate_samples: bool

        normalize: bool
            Normalize the embeddings before applying the dimensionality reduction.

        *args, **kwargs:
            arguments to be passed to scatter_2d, tsne_2d, pca_2d or umap_2d
        """
        value_checker("method", method, ["pca", "tsne", "umap"])
        annotations = self._get_embedding_annotations(
            annotate_signatures, annotate_samples
        )

        data = np.concatenate([self.L, self.U], axis=1).T

        if normalize:
            data /= np.sum(data, axis=0)

        if self.dim_embeddings in [1, 2]:
            warnings.warn(
                f"The embedding dimension is {self.dim_embeddings}. "
                f"The method argument '{method}' will be ignored "
                "and the embeddings are plotted directly.",
                UserWarning,
            )

        if self.dim_embeddings == 1:
            ax = scatter_1d(
                data[:, 0],
                annotations=annotations,
                annotation_kwargs=annotation_kwargs,
                ax=ax,
                **kwargs,
            )

        elif self.dim_embeddings == 2:
            ax = scatter_2d(
                data,
                annotations=annotations,
                annotation_kwargs=annotation_kwargs,
                ax=ax,
                **kwargs,
            )

        elif method == "tsne":
            ax = tsne_2d(
                data,
                annotations=annotations,
                annotation_kwargs=annotation_kwargs,
                ax=ax,
                **kwargs,
            )

        elif method == "pca":
            ax = pca_2d(
                data,
                annotations=annotations,
                annotation_kwargs=annotation_kwargs,
                ax=ax,
                **kwargs,
            )

        else:
            ax = umap_2d(
                data,
                annotations=annotations,
                annotation_kwargs=annotation_kwargs,
                ax=ax,
                **kwargs,
            )

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax


class CorrNMFHyperparameterSelector:
    """
    The embedding dimension of samples and signatures is
    the only hyperparameter of correlated NMF.
    This class implements methods to select the "optimal" embedding dimension.
    The framework of hyperparameter selectors allows to implement
    a denovo signature analysis pipeline in an NMF algorithm agnostic manner:
    A dictionary can be used to set all hyperparameters,
    irrespective of the NMF algorithm and its arbitrary number of hyperparameters.
    """

    def __init__(self, method="unbiased", method_kwargs=None):
        value_checker("method", method, ["BIC", "proportional", "unbiased"])
        self.method = method
        self.method_kwargs = {} if method_kwargs is None else method_kwargs.copy()

        # initialize selection dependent attributes
        self.corrnmf_algorithm = None
        self.dims_embeddings = np.empty(0, dtype=int)
        self.data = None
        self.given_signatures = None
        self.init_kwargs = None
        self.verbose = 0
        self.models = []

    def _job_select_bic(self, dim_embeddings):
        """
        Apply CorrNMF for a single embedding dimension.
        """
        model = deepcopy(self.corrnmf_algorithm)
        model.dim_embeddings = dim_embeddings
        model.fit(
            data=self.data,
            given_signatures=self.given_signatures,
            init_kwargs=self.init_kwargs,
            verbose=0,
        )

        if self.verbose:
            print(f"CorrNMF with dim_embeddings = {dim_embeddings} finished.")

        return model

    def select_bic(self, ncpu=None):
        """
        Select the best embedding dimension based
        on the Bayesian Information Criterion (BIC).
        """
        if ncpu is None:
            ncpu = os.cpu_count()

        if ncpu > 1:
            workers = multiprocessing.Pool(ncpu)
            models = workers.map(self._job_select_bic, self.dims_embeddings)
            workers.close()
            workers.join()

        else:
            models = [
                self._job_select_bic(dim_embeddings)
                for dim_embeddings in self.dims_embeddings
            ]

        self.models = models
        bics = np.array([model.bic for model in models])
        best_index = np.argmin(bics)
        best_model = models[best_index]

        return best_model.dim_embeddings

    def select_proportional(self, proportion=0.75):
        """
        The embedding dimension is set to a proportion of the number of signatures.
        """
        n_signatures = self.corrnmf_algorithm.n_signatures
        dim_embeddings = int(proportion * n_signatures) if n_signatures > 1 else 1

        return dim_embeddings

    def select_unbiased(self, normalized=True):
        """
        The embedding dimension is set to the number of signatures
        if 'normalized' is false. If 'normalized' is true, the embedding
        dimension is set to the number of signatures minus one.

        Input:
        ------
        normalized: bool
            If the input count matrix will be normalized, the number of free
            parameters for each sample exposure is 'n_signatures - 1'.
            Without the normalization, there are 'n_signatures' many free parameters.
        """
        n_signatures = self.corrnmf_algorithm.n_signatures

        if not normalized:
            return n_signatures

        return max(1, n_signatures - 1)

    def select(
        self,
        corrnmf_algorithm,
        data: pd.DataFrame,
        given_signatures=None,
        init_kwargs=None,
        ncpu=None,
        verbose=0,
    ):
        self.corrnmf_algorithm = corrnmf_algorithm
        self.dims_embeddings = np.arange(1, corrnmf_algorithm.n_signatures + 1)
        self.data = data
        self.given_signatures = given_signatures
        self.init_kwargs = init_kwargs
        self.verbose = verbose

        if self.method == "BIC":
            dim_embeddings = self.select_bic(ncpu=ncpu, **self.method_kwargs)

        elif self.method == "proportional":
            dim_embeddings = self.select_proportional(**self.method_kwargs)

        elif self.method == "unbiased":
            dim_embeddings = self.select_unbiased(**self.method_kwargs)

        hyperparameters = {"dim_embeddings": dim_embeddings}

        return hyperparameters
