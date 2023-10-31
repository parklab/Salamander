from abc import abstractmethod

import numpy as np
import pandas as pd

from ..utils import match_signatures_pair
from .initialization import initialize
from .signature_nmf import SignatureNMF

EPSILON = np.finfo(np.float32).eps


class NMF(SignatureNMF):
    """
    The abstract class NMF unifies the structure of NMF algorithms
    with a single signature matrix W and exposure matrix H.
    Examples of these algorithms include the standard NMF algorithm
    (Lee and Seung, 1999), minimum volume NMF (mvNMF) or NMF variants
    with regularizations on the entries of W and H.
    All of these NMF algorithms have the same parameters. Therefore,
    many properties of interest such as the signature correlation martrix
    or the sample embeddings are computed in the same manner.

    Overview:

    Every child class has to implement the following attributes:

        - reconstruction_error: float
            The reconstruction error between the count matrix and
            the reconstructed count matrix.

        - samplewise_reconstruction_error: np.ndarray
            The samplewise reconstruction error between the sample counts
            and the reconstructed sample counts.

        - objective: str
            "minimize" or "maximize". Whether the NMF algorithm maximizes or
            minimizes the objective function.
            Some algorithms maximize a likelihood, others minimize a distance.
            The distinction is useful for filtering NMF runs based on
            the fitted objective function value downstream.


    Every child class has to implement the following methods:

        - objective_function:
            The algorithm-specific objective function

        - loglikelihood:
            The loglikelihood of the underyling generative model

        - _update_W:
            update the signature matrix W

        - _update_H:
            update the exposure matrix H

        - fit:
            Apply the NMF algorithm for a given mutation count data or
            for given signatures and mutation count data


    The following attributes are implemented in the abstract class NMF:

        - signatures: pd.DataFrame
            The signature matrix including mutation type names and signature names

        - exposures: pd.DataFrame
            The exposure matrix including the signature names and sample names

        - _n_parameters:
            The number of parameters fitted

        - corr_signatures: pd.DataFrame
            The signature correlation matrix induced by their sample exposures

        - corr_samples: pd.DataFrame
            The sample correlation matrix induced by their signature exposures


    The following methods are implemented in the abstract class NMF:

        - initialize:
            Initialize all model parameters and latent variables depending on the
            initialization method chosen

        - _get_embedding_annotations:
            A helper function to get the sample names for the embedding plots

        - plot_embeddings:
            Plot signature or sample embeddings in 2D using PCA, tSNE or UMAP.
            The respective plotting functions are implemented in the plot.py module.

    More details are explained in the respective attributes and methods.
    """

    def __init__(
        self,
        n_signatures=1,
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
            have generated the mutation count data.

        init_method: str
            One of "custom", "flat", "hierarchical_cluster", "nndsvd",
            "nndsvda", "nndsvdar" "random" and "separableNMF".
            See the initialization module for further details on each method.

        min_iterations: int
            The minimum number of iterations to perform during inference

        max_iterations: int
            The maximum number of iterations to perform during inference

        tol: float
            The NMF algorithm is converged when the relative change
            of the objective function of one iteration is smaller
            than the tolerance 'tol'.
        """
        super().__init__(n_signatures, init_method, min_iterations, max_iterations, tol)

        # initialize data/fitting dependent attributes
        self.W, self.H = None, None

    @property
    def signatures(self) -> pd.DataFrame:
        signatures = pd.DataFrame(
            self.W, index=self.mutation_types, columns=self.signature_names
        )

        return signatures

    @property
    def exposures(self) -> pd.DataFrame:
        exposures = pd.DataFrame(
            self.H, index=self.signature_names, columns=self.sample_names
        )

        return exposures

    @property
    def _n_parameters(self) -> int:
        """
        There are n_features * n_signatures parameters corresponding to
        the signature matrix and n_signatures * n_samples parameters
        corresponding to the exposure matrix.
        """
        return self.n_signatures * (self.n_features + self.n_samples)

    @abstractmethod
    def _update_W(self):
        pass

    @abstractmethod
    def _update_H(self):
        pass

    def _initialize(self, given_signatures=None, init_kwargs=None):
        """
        Initialize the signature matrix W and exposure matrix H.
        When the signatures are given, the initialization
        of W is overwritten by the given signatures.

        Input:
        ------
        given_signatures : pd.Dataframe, default=None
            At most 'n_signatures' many signatures can be provided to
            overwrite some of the initialized signatures. This does not
            change the initialized exposurse.

        init_kwargs: dict
            Any further keywords arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.
        """
        if given_signatures is not None:
            self._check_given_signatures(given_signatures)
            self.n_given_signatures = len(given_signatures.columns)
        else:
            self.n_given_signatures = 0

        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.W, self.H, self.signature_names = initialize(
            self.X, self.n_signatures, self.init_method, given_signatures, **init_kwargs
        )

    @property
    def corr_signatures(self) -> pd.DataFrame:
        """
        The correlation of two signatures is given by the pearson correlation of
        the respective rows of the exposure matrix H.

        The pandas dataframe method 'corr' computes the pairwise correlation of columns.
        """
        return self.exposures.T.corr(method="pearson")

    @property
    def corr_samples(self) -> pd.DataFrame:
        """
        The correlation of two samples is given by the pearson correlation of
        the respective columns of the exposure matrix H.

        The pandas dataframe method 'corr' computes the pairwise correlation of columns.
        """
        return self.exposures.corr(method="pearson")

    def reorder(self, other_signatures, metric="cosine", keep_names=False):
        reordered_indices = match_signatures_pair(
            other_signatures, self.signatures, metric=metric
        )
        self.W = self.W[:, reordered_indices]
        self.H = self.H[reordered_indices, :]

        if keep_names:
            self.signature_names = self.signature_names[reordered_indices]

        return reordered_indices

    def _get_embedding_data(self):
        """
        In most NMF models like KL-NMF or mvNMF, the data for the embedding plot
        are just the (transposed) exposures.
        """
        return self.H.T.copy()

    def _get_default_embedding_annotations(self) -> np.ndarray:
        """
        The embedding plot defaults to no annotations.
        """
        return np.empty(self.n_samples, dtype=str)
