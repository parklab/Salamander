"""
Multimodal correlated NMF fits multiple correlated NMF models jointly.

Assuming that the input data of each modality originates from the identical samples,
multimodal correlated NMF fixes the sample embeddings accross modalities and learns
signature embeddings of all modalities in a shared embedding space.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Literal

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd

from .. import plot as pl
from .. import tools as tl
from ..initialization.initialize import EPSILON, _Init_methods, initialize_mmcorrnmf
from ..utils import dict_checker, type_checker, value_checker
from . import _utils_corrnmf
from ._utils_klnmf import samplewise_kl_divergence, update_W

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType
    from seaborn.matrix import ClusterGrid

    from .signature_nmf import _Dim_reduction_methods


class MultimodalCorrNMF:
    def __init__(
        self,
        ns_signatures: list[int],
        dim_embeddings: int | None = None,
        init_method: _Init_methods = "nndsvd",
        min_iterations: int = 500,
        max_iterations: int = 10000,
        conv_test_freq: int = 10,
        tol: float = 1e-7,
    ):
        self.ns_signatures = ns_signatures

        if dim_embeddings is None:
            dim_embeddings = np.max(ns_signatures)

        self.dim_embeddings = dim_embeddings
        self.init_method = init_method
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.conv_test_freq = conv_test_freq
        self.tol = tol
        self.variance = 1.0

        # initialize data/fitting dependent attributes
        mod_names_default = [f"mod{n}" for n in range(1, len(ns_signatures) + 1)]
        self.mdata = md.MuData(
            {mod_name: ad.AnnData() for mod_name in mod_names_default}
        )
        self.asignatures = {mod_name: ad.AnnData() for mod_name in mod_names_default}
        self.history: dict[str, Any] = {}
        self.signature_correlation = np.empty((sum(ns_signatures), sum(ns_signatures)))
        self.signature_correlation[:] = np.nan

    @property
    def mod_names(self) -> list[str]:
        return list(self.mdata.mod.keys())

    @property
    def mutation_types(self) -> dict[str, list[str]]:
        return {
            mod_name: list(adata.var_names)
            for mod_name, adata in self.mdata.mod.items()
        }

    @property
    def signature_names(self) -> dict[str, list[str]]:
        return {
            mod_name: list(asigs.obs_names)
            for mod_name, asigs in self.asignatures.items()
        }

    @property
    def sample_names(self) -> list[str]:
        return list(self.mdata.obs_names)

    @property
    def signatures(self) -> dict[str, pd.DataFrame]:
        return {mod_name: asigs.to_df() for mod_name, asigs in self.asignatures.items()}

    @property
    def exposures(self) -> dict[str, pd.DataFrame]:
        exposures_all = {}
        for mod_name in self.mod_names:
            exposures_all[mod_name] = pd.DataFrame(
                self.mdata[mod_name].obsm["exposures"],
                index=self.sample_names,
                columns=self.asignatures[mod_name].obs_names,
            )
        return exposures_all

    def compute_exposures(self) -> None:
        for mod_name in self.mod_names:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            adata.obsm["exposures"] = _utils_corrnmf.compute_exposures(
                asigs.obs["scalings"].values,
                adata.obs["scalings"].values,
                asigs.obsm["embeddings"],
                self.mdata.obsm["embeddings"],
            )

    def compute_reconstruction(self) -> None:
        for mod_name in self.mod_names:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            adata.obsm["X_reconstructed"] = adata.obsm["exposures"] @ asigs.X

    @property
    def data_reconstructed(self) -> dict[str, pd.DataFrame]:
        for adata in self.mdata.mod.values():
            if "X_reconstructed" not in adata.obsm:
                self.compute_reconstruction()

        data_reconstructed_all = {}
        for mod_name, adata in self.mdata.mod.items():
            data_reconstructed_all[mod_name] = pd.DataFrame(
                adata.obsm["X_reconstructed"],
                index=adata.obs_names,
                columns=adata.var_names,
            )
        return data_reconstructed_all

    def compute_reconstruction_errors(self) -> None:
        self.compute_exposures()

        for mod_name in self.mod_names:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            errors = samplewise_kl_divergence(
                adata.X.T, asigs.X.T, adata.obsm["exposures"].T
            )
            adata.obs["reconstruction_error"] = errors

        self.mdata.update()

    @property
    def reconstruction_errors(self) -> dict[str, float]:
        if any(
            "reconstruction_error" not in self.mdata[mod_name].obs
            for mod_name in self.mod_names
        ):
            self.compute_reconstruction_errors()

        return {
            mod_name: np.sum(adata.obs["reconstruction_error"])
            for mod_name, adata in self.mdata.mod.items()
        }

    @property
    def reconstruction_error(self) -> float:
        return np.sum(list(self.reconstruction_errors.values()))

    def objective_function(self) -> float:
        """
        The ELBO of multimodal correlated NMF.
        """
        elbo = 0.0

        for mod_name in self.mod_names:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            elbo += _utils_corrnmf.elbo_corrnmf(
                adata.X,
                asigs.X,
                adata.obsm["exposures"],
                asigs.obsm["embeddings"],
                self.mdata.obsm["embeddings"],
                self.variance,
                penalize_sample_embeddings=False,
            )

        elbo -= (
            0.5
            * self.dim_embeddings
            * self.mdata.n_obs
            * np.log(2 * np.pi * self.variance)
        )
        elbo -= np.sum(self.mdata.obsm["embeddings"] ** 2) / (2 * self.variance)
        return elbo

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def _setup_mdata(self, mdata: md.MuData):
        type_checker("mdata", mdata, md.MuData)
        n_mod_expected = len(self.ns_signatures)

        if mdata.n_mod != n_mod_expected:
            raise ValueError(f"The data has to have {n_mod_expected} many modalities.")

        sample_names_expected = list(mdata.mod.values())[0].obs_names

        for adata in mdata.mod.values():
            if not all(adata.obs_names == sample_names_expected):
                raise ValueError(
                    "The sample names of the different modalities are not identical."
                )

        self.mdata = mdata

    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures, self.variance = initialize_mmcorrnmf(
            self.mdata,
            self.ns_signatures,
            self.dim_embeddings,
            self.init_method,
            given_parameters,
            **init_kwargs,
        )
        self.compute_exposures()

    def _compute_auxs(self) -> dict[str, np.ndarray]:
        r"""
        auxs: dict[str, np.ndarray]
            For every modality
                aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        auxs = {}
        for mod_name in self.mod_names:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            auxs[mod_name] = _utils_corrnmf.compute_aux(
                adata.X, asigs.X, adata.obsm["exposures"]
            )
        return auxs

    def update_sample_scalings_mod(
        self, mod_name: str, given_parameters_mod: dict[str, Any]
    ) -> None:
        if "sample_scalings" not in given_parameters_mod:
            adata = self.mdata[mod_name]
            asigs = self.asignatures[mod_name]
            adata.obs["scalings"] = _utils_corrnmf.update_sample_scalings(
                adata.X,
                asigs.obs["scalings"].values,
                asigs.obsm["embeddings"],
                self.mdata.obsm["embeddings"],
            )

    def update_sample_scalings(
        self,
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        for mod_name in self.mod_names:
            if mod_name in given_parameters:
                given_parameters_mod = given_parameters[mod_name]
            else:
                given_parameters_mod = {}
            self.update_sample_scalings_mod(mod_name, given_parameters_mod)

    def update_signature_scalings_mod(
        self, mod_name: str, aux: np.ndarray, given_parameters_mod: dict[str, Any]
    ) -> None:
        if "signature_scalings" not in given_parameters_mod:
            asigs = self.asignatures[mod_name]
            asigs.obs["scalings"] = _utils_corrnmf.update_signature_scalings(
                aux,
                self.mdata[mod_name].obs["scalings"].values,
                asigs.obsm["embeddings"],
                self.mdata.obsm["embeddings"],
            )

    def update_signature_scalings(
        self,
        auxs: dict[str, np.ndarray],
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        for mod_name in self.mod_names:
            if mod_name in given_parameters:
                given_parameters_mod = given_parameters[mod_name]
            else:
                given_parameters_mod = {}
            self.update_signature_scalings_mod(
                mod_name, auxs[mod_name], given_parameters_mod
            )

    def update_variance(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "variance" not in given_parameters:
            signature_embeddings = np.concatenate(
                [asigs.obsm["embeddings"] for asigs in self.asignatures.values()]
            )
            embeddings = np.concatenate(
                [signature_embeddings, self.mdata.obsm["embeddings"]]
            )
            variance = np.mean(embeddings**2)
            self.variance = np.clip(variance, EPSILON, None)

    def update_signatures_mod(
        self, mod_name: str, given_parameters_mod: dict[str, Any]
    ) -> None:
        if "asignatures" in given_parameters_mod:
            n_given_signatures = given_parameters_mod["asignatures"].n_obs
        else:
            n_given_signatures = 0

        asigs = self.asignatures[mod_name]
        W = update_W(
            self.mdata[mod_name].X.T,
            asigs.X.T,
            self.mdata[mod_name].obsm["exposures"].T,
            n_given_signatures=n_given_signatures,
        )
        asigs.X = W.T

    def update_signatures(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        for mod_name in self.mod_names:
            if mod_name in given_parameters:
                given_parameters_mod = given_parameters[mod_name]
            else:
                given_parameters_mod = {}
            self.update_signatures_mod(mod_name, given_parameters_mod)

    def update_signature_embeddings_mod(
        self,
        mod_name: str,
        aux: np.ndarray,
        outer_prods_sample_embeddings: np.ndarray,
        given_parameters_mod: dict[str, Any],
    ) -> None:
        if "signature_embeddings" not in given_parameters_mod:
            asigs = self.asignatures[mod_name]
            for k, aux_row in enumerate(aux):
                embedding_init = asigs.obsm["embeddings"][k, :]
                asigs.obsm["embeddings"][k, :] = _utils_corrnmf.update_embedding(
                    embedding_init,
                    self.mdata.obsm["embeddings"],
                    asigs.obs["scalings"][k],
                    self.mdata[mod_name].obs["scalings"].values,
                    self.variance,
                    aux_row,
                    outer_prods_sample_embeddings,
                )

    def update_signature_embeddings(
        self,
        auxs: dict[str, np.ndarray],
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Update all signature embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method.
        """
        if given_parameters is None:
            given_parameters = {}

        outer_prods_sample_embeddings = np.einsum(
            "Dm,Dn->Dmn",
            self.mdata.obsm["embeddings"],
            self.mdata.obsm["embeddings"],
        )
        for mod_name in self.mod_names:
            if mod_name in given_parameters:
                given_parameters_mod = given_parameters[mod_name]
            else:
                given_parameters_mod = {}
            self.update_signature_embeddings_mod(
                mod_name,
                auxs[mod_name],
                outer_prods_sample_embeddings,
                given_parameters_mod,
            )

    def update_sample_embeddings(self, auxs: dict[str, np.ndarray]) -> None:
        sig_embeddings = np.concatenate(
            [asigs.obsm["embeddings"] for asigs in self.asignatures.values()]
        )
        outer_prods_sig_embeddings = np.einsum(
            "Km,Kn->Kmn", sig_embeddings, sig_embeddings
        )
        sig_scalings = np.concatenate(
            [asigs.obs["scalings"] for asigs in self.asignatures.values()]
        )
        aux = np.concatenate([aux for aux in auxs.values()])

        for d, aux_col in enumerate(aux.T):
            embedding_init = self.mdata.obsm["embeddings"][d, :]
            scalings = [
                np.repeat(adata.obs["scalings"][d], n_signatures)
                for adata, n_signatures in zip(
                    self.mdata.mod.values(), self.ns_signatures
                )
            ]
            scalings = np.concatenate(scalings)
            self.mdata.obsm["embeddings"][d, :] = _utils_corrnmf.update_embedding(
                embedding_init,
                sig_embeddings,
                scalings,
                sig_scalings,
                self.variance,
                aux_col,
                outer_prods_sig_embeddings,
                options={"maxiter": 3},
            )

    def update_embeddings(
        self,
        auxs: dict[str, np.ndarray],
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        self.update_signature_embeddings(auxs, given_parameters)

        if "sample_embeddings" not in given_parameters:
            self.update_sample_embeddings(auxs)

    def _update_parameters(self, given_parameters: dict[str, Any] | None = None):
        if given_parameters is None:
            given_parameters = {}

        self.update_sample_scalings(given_parameters)
        self.compute_exposures()
        auxs = self._compute_auxs()
        self.update_signature_scalings(auxs, given_parameters)
        self.update_embeddings(auxs, given_parameters)
        self.update_variance(given_parameters)
        self.update_signatures(given_parameters)

    def fit(
        self,
        mdata: md.MuData,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
        history: bool = True,
        verbose: Literal[0, 1] = 0,
        verbosity_freq: int = 100,
    ) -> MultimodalCorrNMF:
        self._setup_mdata(mdata)
        self._initialize(given_parameters, init_kwargs)
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

        self.mdata.update()
        return self

    def plot_history(self, outfile: str | None = None, **kwargs) -> Axes:
        if not self.history:
            raise ValueError(
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
        colors: dict[str, ColorType | list[ColorType]] | None = None,
        annotate_mutation_types: bool = False,
        figsize: tuple[float, float] | None = None,
        outfile: str | None = None,
        **kwargs,
    ):
        colors = {} if colors is None else colors.copy()
        dict_checker("colors", colors, self.mod_names)
        max_n_signatures = np.max(self.ns_signatures)

        if figsize is None:
            figsize = (4 * self.mdata.n_mod, max_n_signatures)

        fig, axes = plt.subplots(max_n_signatures, self.mdata.n_mod, figsize=figsize)

        for mod_name, axs in zip(self.mod_names, axes.T):
            sigs = self.asignatures[mod_name]
            cols = colors[mod_name] if mod_name in colors else None
            n_sigs = sigs.n_obs
            pl.barplot(
                sigs,
                colors=cols,
                annotate_vars=annotate_mutation_types,
                axes=axs[:n_sigs],
                **kwargs,
            )
            for ax in axs[n_sigs:]:
                fig.delaxes(ax)

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes

    def plot_exposures(
        self,
        sample_order: np.ndarray | None = None,
        reorder_signatures: bool = True,
        annotate_samples: bool = True,
        colors: dict[str, Iterable[ColorType]] | None = None,
        axes: Iterable[Axes] | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> Iterable[Axes]:
        """
        Visualize the exposures as a stacked bar chart,
        see plot.py for the implementation.

        Input:
        ------
        **kwargs:
            arguments to be passed to exposure_plot
        """
        if axes is None:
            _, axes = plt.subplots(self.mdata.n_mod, figsize=(20, 3 * self.mdata.n_mod))

        colors = {} if colors is None else colors.copy()
        dict_checker("colors", colors, self.mod_names)
        exposures = self.exposures

        if sample_order is None:
            exposures_all_normalized = pd.concat(
                [df.div(df.sum(axis=1), axis=0) for df in exposures.values()], axis=1
            )
            sample_order = pl.get_obs_order(exposures_all_normalized)

        for n, (mod_name, ax) in enumerate(zip(self.mod_names, axes)):
            expos = exposures[mod_name]
            cols = colors[mod_name] if mod_name in colors else None

            if n < self.mdata.n_mod - 1:
                annotate = False
            else:
                annotate = annotate_samples

            ax = pl.stacked_barplot(
                data=expos,
                obs_order=sample_order,
                reorder_dimensions=reorder_signatures,
                annotate_obs=annotate,
                colors=cols,
                ax=ax,
                **kwargs,
            )
            ax.set_title(f"{self.mod_names[n]} signature exposures")

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes

    def compute_correlation(
        self, data: Literal["samples", "signatures"] = "signatures", **kwargs
    ) -> None:
        """
        Compute the signature or sample correlation. The signature
        correlation is stored as a new model attribute, the sample correlation
        is stored in mdata.
        """
        value_checker("data", data, ["samples", "signatures"])

        for adata in self.mdata.mod.values():
            assert "exposures" in adata.obsm, (
                "Computing the sample or signature correlation "
                "requires fitting the NMF model."
            )

        values = np.concatenate(
            [adata.obsm["exposures"] for adata in self.mdata.mod.values()], axis=1
        )

        if data == "signatures":
            values = values.T

        correlation = tl.correlation_numpy(values, **kwargs)

        if data == "samples":
            self.mdata.obsp["X_correlation"] = correlation
        else:
            self.signature_correlation = correlation

    def correlation(
        self, data: Literal["samples", "signatures"] = "signatures"
    ) -> pd.DataFrame:
        """
        Dataframe of the signature or sample correlation.
        """
        value_checker("data", data, ["samples", "signatures"])

        if data == "samples":
            if "X_correlation" not in self.mdata.obsp:
                self.compute_correlation("samples")
            values = self.mdata.obsp["X_correlation"]
            names = self.sample_names

        else:
            if np.isnan(self.signature_correlation).all():
                self.compute_correlation("signatures")
            values = self.signature_correlation
            names = sum(self.signature_names.values(), [])

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

    def plot_embeddings(
        self,
        method: _Dim_reduction_methods = "umap",
        n_components: int = 2,
        dimensions: tuple[int, int] = (0, 1),
        color: str | None = None,
        zorder: str | None = None,
        annotations: Iterable[str] | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes:
        adatas = list(self.asignatures.values()) + [self.mdata]
        tl.reduce_dimension_multiple(
            adatas=adatas,
            basis="embeddings",
            method=method,
            n_components=n_components,
        )
        if self.dim_embeddings <= 2:
            warnings.warn(
                f"The embedding dimension is {self.dim_embeddings}. "
                "The embeddings are plotted without an additional "
                "dimensionality reduction.",
                UserWarning,
            )
            basis = "embeddings"
        else:
            basis = method

        if color is None:
            color = "color_embeddings"
            for asigs in self.asignatures.values():
                asigs.obs[color] = asigs.n_obs * ["black"]
            self.mdata.obs[color] = self.mdata.n_obs * ["#1f77b4"]  # default blue

        if zorder is None:
            zorder = "zorder_embeddings"
            for asigs in self.asignatures.values():
                asigs.obs[zorder] = asigs.n_obs * [2]
            self.mdata.obs[zorder] = self.mdata.n_obs * [1]

        if annotations is None:
            annotations = sum(self.signature_names.values(), [])

        ax = pl.embedding_multiple(
            adatas=adatas,
            basis=basis,
            dimensions=dimensions,
            color=color,
            zorder=zorder,
            annotations=annotations,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax
