from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from .. import plot as pl
from .. import tools as tl
from ..utils import type_checker, value_checker
from .initialization import _INIT_METHODS, EPSILON, _Init_methods

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.axes import Axes
    from seaborn.matrix import ClusterGrid

_Dim_reduction_methods = Literal[
    "pca",
    "tsne",
    "umap",
]
_DIM_REDUCTION_METHODS = get_args(_Dim_reduction_methods)


class SignatureNMF(ABC):
    """
    The abstract class SignatureNMF unifies the structure of
    multiple NMF algorithms used for signature analysis.

    Common properties and methods of all algorithms are indicated,
    i.e. have to be implemented by child classes, or implemented. Overview:

    Every child class has to implement the following attributes:

        - objective: Literal["minimize", "maximize"]
            Whether the NMF algorithm minimize or maximize the objective function.
            Some algorithms maximize a likelihood, others minimize a distance.

    Every child class has to implement the following methods:

        - compute_reconstruction_errors:
            Add the samplewise reconstruction errors to adata.obs with the key
            'reconstruction_error'.

        - objective_fuction:
            The objective function to optimize during model training.

        - _initialize:
            Initialize all model parameters before model training.

        - _setup_fitting_parameters:
            Initialize any additional and parameters required to fit
            the NMF model.

        - _update_parameters:
            Update all model parameters.

        - reorder:
            Reorder the model parameters to match the order of another
            collection of signatures.

        - reduce_dimension_embeddings:
            Reduce the dimension of the canonical model embeddings.
            These are typically the sample exposures.

        - _get_embedding_plot_adata:
            A helper function for the embedding plot.

        - _get_default_embedding_plot_annotations:
            A helper function for the embedding plot.

    The following attributes and methods are implemented in SignatureNMF:

        - mutation_types: np.ndarray
            Wrapper around the var_names of the count data.

        - signature_names: np.ndarray
            Wrapper around the obs_names of the signatures AnnData object.

        - sample_names: np.ndarrray
            Wrapper around the obs_names of the count data.

        - signatures: pd.DataFrame
            Wrapper around the signatures AnnData object to return
            the signatures as a dataframe.

        - exposures: pd.DataFrame
            Wrapper around adata.obsm to return the signature exposures
            as a dataframe.

        - compute_reconstruction: None
            Add the reconstrcuted counts to adata.obsm with key 'X_reconstructed'.

        - data_reconstructed: pd.DataFrame
            The recovered mutation count data as a dataframe.

        - reconstruction_error: float
            The sum of the samplewise reconstruction errors.

        - _setup_adata:
            Perform parameter checks on the input AnnData count object and clip zeros.

        - _check_given_asignatures:
            Perform parameter checks on the optial input AnnData signature object.

        - fit:
            Fit all model parameters.

        - compute_correlation:
            Add sample or signature correlations to the AnnData objects.

        - correlation:
            The sample or signature correlation as a dataframe.

        - plot_history:
            Plot the history of the objective function values after fitting the model

        - plot_signatures:
            Plot the signatures as a barplot.

        - plot_exposures:
            Plot the exposures as a stacked barplot.

        - plot_correlation:
            Plot the correlation of either the signatures or exposures.

        - plot_embeddings:
            Plot the sample (and potentially the signature) embeddings in 2D
            using PCA, tSNE or UMAP.
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
        """
        Inputs
        ------
        n_signatures: int
            The number of signatures that are assumed to
            have generated the mutation count data.

        init_method: str, default='nndsvd'
            The model parameter initialization method.

        min_iterations: int, default=500
            The minimum number of iterations to perform by the NMF algorithm

        max_iterations: int, default=10000
            The maximum number of iterations to perform by the NMF algorithm

        conv_test_freq: int, default=10
            The frequency at which the algorithm is tested for convergence.
            The objective function value is only computed every 'conv_test_freq'
            many iterations.

        tol: float
            The convergence tolerance. The NMF algorithm is converged
            when the relative change of the objective function is smaller
            than 'tol'.
        """
        value_checker("init_method", init_method, _INIT_METHODS)

        self.n_signatures = n_signatures
        self.init_method = init_method
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.conv_test_freq = conv_test_freq
        self.tol = tol

        # initialize data/fitting dependent attributes
        self.adata = AnnData()
        self.asignatures = AnnData()
        self.history: dict[str, Any] = {}

    @property
    def mutation_types(self) -> np.ndarray:
        return self.adata.var_names.to_numpy(dtype=str)

    @property
    def signature_names(self) -> np.ndarray:
        return self.asignatures.obs_names.to_numpy(dtype=str)

    @property
    def sample_names(self) -> np.ndarray:
        return self.adata.obs_names.to_numpy(dtype=str)

    @property
    def signatures(self) -> pd.DataFrame:
        """
        Extract the mutational signatures as a pandas dataframe.
        """
        return self.asignatures.to_df()

    @property
    def exposures(self) -> pd.DataFrame:
        """
        Extract the signature exposures as a pandas dataframe.
        """
        assert (
            "exposures" in self.adata.obsm
        ), "Learning the sample exposures requires fitting the NMF model."
        exposures_df = pd.DataFrame(
            self.adata.obsm["exposures"],
            index=self.sample_names,
            columns=self.signature_names,
        )
        return exposures_df

    def compute_reconstruction(self) -> None:
        self.adata.obsm["X_reconstructed"] = (
            self.adata.obsm["exposures"] @ self.asignatures.X
        )

    @property
    def data_reconstructed(self) -> pd.DataFrame:
        if not "X_reconstructed" in self.adata.obsm:
            self.compute_reconstruction()

        return pd.DataFrame(
            self.adata.obsm["X_reconstructed"],
            index=self.sample_names,
            columns=self.mutation_types,
        )

    @abstractmethod
    def compute_reconstruction_errors(self) -> None:
        """
        The samplewise reconstruction errors between the data
        and the reconstructed data.
        """

    @property
    def reconstruction_error(self) -> float:
        """
        The total reconstruction error between the data and
        the reconstructed data.
        """
        if not "reconstruction_error" in self.adata.obs:
            self.compute_reconstruction_errors()

        return np.sum(self.adata.obs["reconstruction_error"])

    @property
    @abstractmethod
    def objective(self) -> Literal["minimize", "maximize"]:
        """
        Whether the NMF algorithm minimizes or maximizes its objective
        function.
        """

    @abstractmethod
    def objective_function(self) -> float:
        """
        The objective function to be optimized during fitting.
        """

    def _setup_adata(self, adata: AnnData) -> None:
        """
        Check the type of the input counts and clip them to
        avoid floating point errors.

        Inputs
        ------
        data: AnnData
            The AnnData object with the mutation count matrix.
        """
        type_checker("adata", adata, AnnData)
        self.adata = adata
        self.adata.X = self.adata.X.clip(EPSILON)

    def _check_given_asignatures(self, given_asignatures: AnnData) -> None:
        """
        Check if the given signatures are compatible with the
        number of signatures of the algorithm and the
        mutation types of the input data.
        Should be called by implementations of _initialize.

        Inputs
        ------
        given_asignatures: AnnData
            Known signatures that should be fixed by the algorithm.
            The number of known signatures can be less or equal to the
            number of signatures specified by the algorithm.
        """
        type_checker("given_asignatures", given_asignatures, AnnData)
        given_mutation_types = given_asignatures.var_names.to_numpy(dtype=str)
        compatible = (
            np.array_equal(given_mutation_types, self.mutation_types)
            and given_asignatures.n_obs <= self.n_signatures
        )
        if not compatible:
            raise ValueError(
                f"You have to provide at most {self.n_signatures} signatures with "
                "mutation types matching to your data."
            )

    @abstractmethod
    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Initialize the NMF model parameters and return the potentially
        updated 'given_parameters'.

        Example:
            Before running the Lee & Seung NMF multiplicative update rules to
            decompose the mutation count matrix into a signature matrix and
            an exposure matrix, both matrices have to be initialized.
        """

    @abstractmethod
    def _setup_fitting_parameters(
        self, fitting_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize any additional and parameters required to fit
        the NMF model.
        """

    @abstractmethod
    def _update_parameters(self, given_parameters: dict[str, Any]) -> None:
        """
        Update all model parameters.
        """

    def fit(
        self,
        adata: AnnData,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
        fitting_kwargs: dict[str, Any] | None = None,
        history: bool = True,
        verbose: Literal[0, 1] = 0,
        verbosity_freq: int = 1000,
    ) -> SignatureNMF:
        """
        Fit the model parameters. NMF models are expected to handle
        'given_parameters' appropriately.

        Inputs
        ------
        adata: AnnData
            The mutation count matrix as an AnnData object.

        given_parameters: dict, optional
            A priori known parameters. The key is expected to be the parameter
            name.

        init_kwargs: dict, optional
            Keyword arguments to pass to the model parameter initialization, e.g.,
            a seed when a stochastic initialization method is used.

        fitting_kwargs: dict, optional
            Keyword arguments to pass to the initialization of additional fitting
            parameters, e.g., sample-specific loss function weights.

        history: bool, default=True
            If True, the objective function values computed during model training
            will be stored.

        verbose: Literal[0, 1], default=0
            If True, intermediate objective function values obtained during model
            training are printed.

        verbosity_freq: int, default=1000
            The objective function values after every 'verbosity_freq' many
            iterations are printed. Only applies if 'verbose' is set to 1.
        """
        self._setup_adata(adata)
        given_parameters = self._initialize(given_parameters, init_kwargs)
        self._setup_fitting_parameters(fitting_kwargs)
        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % verbosity_freq == 0:
                print(f"iteration: {n_iteration}; objective: {of_values[-1]:.2f}")

            self._update_parameters(given_parameters)

            if n_iteration % self.conv_test_freq == 0:
                prev_of_value = of_values[-1]
                of_values.append(self.objective_function())
                rel_change_nominator = np.abs(prev_of_value - of_values[-1])
                rel_change = rel_change_nominator / np.abs(prev_of_value)
                converged = rel_change < self.tol and n_iteration >= self.min_iterations

            converged |= n_iteration >= self.max_iterations

        if history:
            self.history["objective_function"] = of_values[1:]

        return self

    @abstractmethod
    def reorder(self, asignatures_other: AnnData) -> None:
        """
        Reorder the model parameters to match the order of another
        collection of signatures.
        """

    def plot_history(self, outfile: str | None = None, **kwargs) -> Axes:
        """
        Plot the history of the objective function values. See
        the implemenation of 'history' in the plotting module.
        """
        assert "objective_function" in self.history, (
            "No history available, the model has to be fitted first. "
            "Remember to set 'history' to 'True' when calling 'fit()'."
        )
        ax = pl.history(
            values=self.history["objective_function"],
            conv_test_freq=self.conv_test_freq,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax

    def plot_signatures(
        self,
        annotate_mutation_types: bool = False,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes | Iterable[Axes]:
        """
        Plot the signatures, see the implementation of 'barplot' in
        the plotting module.
        """
        axes = pl.barplot(
            self.asignatures,
            annotate_vars=annotate_mutation_types,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes

    def plot_exposures(
        self,
        sample_order: np.ndarray | None = None,
        reorder_signatures: bool = True,
        annotate_samples: bool = True,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes:
        """
        Visualize the exposures as a stacked bar chart, see
        the implementation of 'stacked_barplot' in the plotting
        module.

        Inputs
        ------
        sample_order: np.ndarray, optional
            A pre-defined order of the samples along the x-axis.

        reorder_signatures: bool, default=True
            If True, the signatures are ordered by their total
            relative contributions.

        annotate_samples: bool, default=True
            If True, the x-axis is annotated with the sample names.

        outfile : str, default=None
            If not None, the figure will be saved in the specified file path.

        **kwargs:
            Any further keyword arguments to pass to 'stacked_barplot'.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """
        ax = pl.stacked_barplot(
            data=self.exposures,
            obs_order=sample_order,
            reorder_dimensions=reorder_signatures,
            annotate_obs=annotate_samples,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax

    def compute_correlation(
        self, data: Literal["samples", "signatures"] = "signatures", **kwargs
    ) -> None:
        """
        Compute the signature or sample correlation and store it in the
        respective anndata object.
        """
        value_checker("data", data, ["samples", "signatures"])

        assert (
            "exposures" in self.adata.obsm
        ), "Computing the sample or signature correlation requires fitting the NMF model."

        values = self.adata.obsm["exposures"]

        if data == "signatures":
            values = values.T

        correlation = tl._correlation(values, **kwargs)

        if data == "samples":
            self.adata.obsp["X_correlation"] = correlation
        else:
            self.asignatures.obsp["correlation"] = correlation

    def correlation(
        self, data: Literal["samples", "signatures"] = "signatures"
    ) -> pd.DataFrame:
        """
        Dataframe of the signature or sample correlation.
        """
        value_checker("data", data, ["samples", "signatures"])

        if data == "samples":
            if "X_correlation" not in self.adata.obsp:
                self.compute_correlation("samples")
            values = self.adata.obsp["X_correlation"]
            names = self.sample_names

        else:
            if "correlation" not in self.asignatures.obsp:
                self.compute_correlation("signatures")
            values = self.asignatures.obsp["correlation"]
            names = self.signature_names

        correlation_df = pd.DataFrame(values, index=names, columns=names)
        return correlation_df

    def plot_correlation(
        self,
        data: Literal["samples", "signatures"] = "signatures",
        annot: bool | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> ClusterGrid:
        """
        Plot the signature or sample correlation.
        """
        value_checker("data", data, ["samples", "signatures"])
        corr = self.correlation(data=data)

        if annot is None:
            annot = False if data == "samples" else True

        clustergrid = pl.correlation_pandas(corr, annot=annot, **kwargs)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return clustergrid

    @abstractmethod
    def reduce_dimension_embeddings(
        self, method: _Dim_reduction_methods = "umap", **kwargs
    ) -> None:
        """
        Reduce the dimension of the embeddings.
        Usually, the embeddings are the signature exposures of the samples.
        But in correlated NMF, they are the signature and sample embeddings
        in a shared embedding space.
        """

    @abstractmethod
    def _get_embedding_plot_adata(
        self, method: _Dim_reduction_methods = "umap"
    ) -> tuple[AnnData, str]:
        """
        Get the anndata object containing all embeddings
        and the name of the embedding basis.
        """

    @abstractmethod
    def _get_default_embedding_plot_annotations(self) -> Iterable[str] | None:
        """
        Get the default annotations of the data points in the embedding plot.
        """

    def plot_embeddings(
        self,
        method: _Dim_reduction_methods = "umap",
        n_components: int = 2,
        dimensions: tuple[int, int] = (0, 1),
        annotations: Iterable[str] | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plot a dimensionality reduction of the exposure representation.
        In most NMF algorithms, this is just the exposures of the samples.
        In CorrNMF, the exposures matrix is refactored, and there are both
        sample and signature embeddings in a shared embedding space.

        If the embedding dimension is one or two, the embeddings are be plotted
        directly, ignoring the chosen dimensionality reduction method.

        Inputs
        ------
        method: str, default='umap'
            The dimensionality reduction method. One of ['pca', 'tsne', 'umap'].

        n_components: int, default=2
            The target dimension of the dimensionality reduction.

        dimensions: tuple[int, int], default=(0,1)
            The indices of the dimensions to plot.

        annotations : Iterable[str], optional, default=None
            Annotations per data point, e.g. the sample names. If None,
            the algorithm-specific default annotations are used.
            For example, CorrNMF annotates the signature embeddings by default.
            Note that there are 'n_signatures' + 'n_samples' data points in CorrNMF,
            i.e. the first 'n_signatures' elements in 'annotations'
            are the signature annotations, not any sample annotations.

        outfile : str, default=None
            If not None, the figure will be saved in the specified file path.

        **kwargs :
            keyword arguments to pass to the scatterplot implementation.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """
        self.reduce_dimension_embeddings(method=method, n_components=n_components)
        adata, basis = self._get_embedding_plot_adata(method=method)

        if annotations is None:
            annotations = self._get_default_embedding_plot_annotations()

        ax = pl.embedding(
            adata=adata,
            basis=basis,
            dimensions=dimensions,
            annotations=annotations,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax
