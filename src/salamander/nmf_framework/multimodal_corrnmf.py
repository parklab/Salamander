"""
Multimodal correlated NMF (MultiCorrNMF) fits multiple correlated NMF (CorrNMF)
models jointly in the following manner:
Assuming that the input data for each modality originates from the identical samples,
MultiCorrNMF fixes the sample embeddings accross modalities and learns signature
embeddings for all modalities in the same embedding space.
"""
# This implementation heavily relies on the implementaion of CorrNMF in
# corrnmf_det.py. In particular, CorrNMFDet methods with a leading '_'
# are accessed.
# pylint: disable=protected-access

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial.distance import squareform

from ..plot import (
    corr_plot,
    paper_style,
    pca_2d,
    scatter_1d,
    scatter_2d,
    signatures_plot,
    tsne_2d,
    umap_2d,
)
from ..utils import type_checker, value_checker
from .corrnmf_det import CorrNMFDet

EPSILON = np.finfo(np.float32).eps


class MultimodalCorrNMF:
    def __init__(
        self,
        n_modalities,
        ns_signatures=None,
        dim_embeddings=None,
        init_method="nndsvd",
        update_W="1999-Lee",
        min_iterations=500,
        max_iterations=10000,
        tol=1e-7,
    ):
        self.n_modalities = n_modalities

        if ns_signatures is None:
            ns_signatures = np.ones(n_modalities, dtype=int)

        self.ns_signatures = ns_signatures

        if dim_embeddings is None:
            dim_embeddings = np.max(ns_signatures)

        self.dim_embeddings = dim_embeddings
        self.init_method = init_method
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.tol = tol
        self.models = [
            CorrNMFDet(n_signatures, dim_embeddings, init_method, update_W)
            for n_signatures in ns_signatures
        ]

        # initialize data/fitting dependent attributes
        self.modality_names = np.empty(n_modalities, dtype=str)
        self.n_samples = 0
        self.history = {}

    @property
    def signatures(self) -> dict:
        return {
            name: model.signatures
            for name, model in zip(self.modality_names, self.models)
        }

    @property
    def exposures(self) -> dict:
        return {
            name: model.exposures
            for name, model in zip(self.modality_names, self.models)
        }

    @property
    def data_reconstructed(self) -> dict:
        return {
            name: model.data_reconstructred
            for name, model in zip(self.modality_names, self.models)
        }

    @property
    def Xs_reconstructed(self) -> np.ndarray:
        return {
            name: model.X_reconstructed
            for name, model in zip(self.modality_names, self.models)
        }

    @property
    def reconstruction_errors(self) -> float:
        return {
            name: model.reconstruction_error
            for name, model in zip(self.modality_names, self.models)
        }

    @property
    def samplewise_reconstruction_errors(self) -> np.ndarray:
        return {
            name: model.samplewise_reconstruction_error
            for name, model in zip(self.modality_names, self.models)
        }

    def objective_function(self) -> float:
        """
        The objective function to be optimized during fitting.
        """
        elbo = np.sum(
            [
                model.objective_function(penalize_sample_embeddings=False)
                for model in self.models
            ]
        )
        elbo -= (
            0.5
            * self.dim_embeddings
            * self.n_samples
            * np.log(2 * np.pi * self.models[0].sigma_sq)
        )
        elbo -= np.sum(self.models[0].U ** 2) / (2 * self.models[0].sigma_sq)

        return elbo

    @property
    def objective(self) -> str:
        return "maximize"

    def _surrogate_objective_function(self, ps) -> float:
        """
        The surrogate lower bound of the ELBO.
        """
        sof_value = np.sum(
            [
                model._surrogate_objective_function(p, penalize_sample_embeddings=False)
                for model, p in zip(self.models, ps)
            ]
        )
        sof_value -= (
            0.5
            * self.dim_embeddings
            * self.n_samples
            * np.log(2 * np.pi * self.models[0].sigma_sq)
        )
        sof_value -= np.sum(self.models[0].U ** 2) / (2 * self.models[0].sigma_sq)

        return sof_value

    def loglikelihood(self) -> float:
        """
        The log-likelihood of the underlying generative model.
        """
        return self.objective_function()

    @property
    def _n_parameters(self) -> int:
        n_parameters_signatures = np.sum(
            [model.n_features * model.n_signatures for model in self.models]
        )
        n_parameters_embeddings = self.dim_embeddings * (
            np.sum(self.ns_signatures) + self.n_samples
        )
        n_parameters_biases = self.n_modalities * self.n_samples
        n_parameters_exposures = n_parameters_embeddings + n_parameters_biases
        n_parameters = n_parameters_signatures + n_parameters_exposures + 1

        return n_parameters

    @property
    def bic(self) -> float:
        return self._n_parameters * np.log(self.n_samples) - 2 * self.loglikelihood()

    def _update_alphas(self):
        for model in self.models:
            model._update_alpha()

    def _update_sigma_sq(self):
        sum_norm_sigs = np.sum([np.sum(model.L**2) for model in self.models])
        sum_norm_samples = np.sum(self.models[0].U ** 2)

        sigma_sq = (sum_norm_sigs + sum_norm_samples) / (
            self.dim_embeddings * (np.sum(self.ns_signatures) + self.n_samples)
        )
        sigma_sq = np.clip(sigma_sq, EPSILON, None)

        for model in self.models:
            model.sigma_sq = sigma_sq

    def _update_Ws(self, ps, given_signatures):
        for model, p, given_sigs in zip(self.models, ps, given_signatures):
            if given_sigs is None:
                model._update_W(p)

    def _update_ps(self):
        return [model._update_p() for model in self.models]

    def _objective_fun_u(self, u, index, aux_cols):
        s = -np.sum(
            [
                model._objective_fun_u(u, index, aux_col, add_penalty_u=False)
                for model, aux_col in zip(self.models, aux_cols)
            ]
        )
        s -= np.dot(u, u) / (2 * self.models[0].sigma_sq)

        return -s

    def _gradient_u(self, u, index, s_grads):
        s = -np.sum(
            [
                model._gradient_u(u, index, s_grad, add_penalty_u=False)
                for model, s_grad in zip(self.models, s_grads)
            ],
            axis=0,
        )
        s -= u / self.models[0].sigma_sq

        return -s

    def _hessian_u(self, u, index, outer_prods_Ls):
        s = -np.sum(
            [
                model._hessian_u(u, index, outer_prods_L, add_penalty_u=False)
                for model, outer_prods_L in zip(self.models, outer_prods_Ls)
            ],
            axis=0,
        )
        s -= np.diag(np.full(self.dim_embeddings, 1 / self.models[0].sigma_sq))

        return -s

    def _update_u(self, index, aux_cols, outer_prods_Ls):
        def objective_fun(u):
            return self._objective_fun_u(u, index, aux_cols)

        s_grads = np.array(
            [
                np.sum(aux_col * model.L, axis=1)
                for model, aux_col in zip(self.models, aux_cols)
            ]
        )

        def gradient(u):
            return self._gradient_u(u, index, s_grads)

        def hessian(u):
            return self._hessian_u(u, index, outer_prods_Ls)

        u = optimize.minimize(
            fun=objective_fun,
            x0=self.models[0].U[:, index],
            method="Newton-CG",
            jac=gradient,
            hess=hessian,
            options={"maxiter": 3},
        ).x
        u[(0 < u) & (u < EPSILON)] = EPSILON
        u[(-EPSILON < u) & (u < 0)] = -EPSILON

        for model in self.models:
            model.U[:, index] = u

    def _update_U(self, auxs):
        outer_prods_Ls = [
            np.einsum("mK,nK->Kmn", model.L, model.L) for model in self.models
        ]

        for d in range(self.n_samples):
            aux_cols = [aux[:, d] for aux in auxs]
            self._update_u(d, aux_cols, outer_prods_Ls)

    def _update_Ls(self, auxs, outer_prods_U, given_signature_embeddings):
        for model, aux, given_sig_embs in zip(
            self.models, auxs, given_signature_embeddings
        ):
            if given_sig_embs is None:
                model._update_L(aux, outer_prods_U)

    def _update_LsU(self, ps, given_signature_embeddings, given_sample_embeddings):
        auxs = [
            np.einsum("vd,vkd->kd", model.X, p) for model, p in zip(self.models, ps)
        ]
        outer_prods_U = np.einsum("mD,nD->Dmn", self.models[0].U, self.models[0].U)
        self._update_Ls(auxs, outer_prods_U, given_signature_embeddings)

        if given_sample_embeddings is None:
            self._update_U(auxs)

    def _setup_data_parameters(self, data: list):
        type_checker("data", data, list)

        if len(data) != self.n_modalities:
            raise ValueError(
                f"The input data has to be {self.n_modalities} "
                "many named pandas dataframes."
            )

        for df in data:
            type_checker("input dataframe", df, pd.DataFrame)

            if df.index.name is None:
                raise ValueError(
                    "You have to set 'df.index.name' to a "
                    "meaningful name for every input dataframe."
                )

        self.modality_names = np.array([df.index.name for df in data])
        self.n_samples = data[0].shape[1]

        for model, df in zip(self.models, data):
            model._setup_data_parameters(df)

    def _initialize(
        self,
        given_signatures=None,
        given_signature_embeddings=None,
        given_sample_embeddings=None,
        init_kwargs=None,
    ):
        if given_sample_embeddings is None:
            U = np.random.multivariate_normal(
                np.zeros(self.dim_embeddings),
                np.identity(self.dim_embeddings),
                size=self.n_samples,
            ).T
        else:
            U = given_sample_embeddings

        for model, modality_name, given_sigs, given_sig_embs in zip(
            self.models,
            self.modality_names,
            given_signatures,
            given_signature_embeddings,
        ):
            if given_sigs is None:
                model.signature_names = np.char.add(
                    modality_name + " ", model.signature_names
                )

            model._initialize(
                given_signatures=given_sigs,
                given_signature_embeddings=given_sig_embs,
                given_sample_embeddings=U,
                init_kwargs=init_kwargs,
            )

    def fit(
        self,
        data: list,
        given_signatures=None,
        given_signature_embeddings=None,
        given_sample_embeddings=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        if given_signatures is None:
            given_signatures = [None for _ in range(self.n_modalities)]

        if given_signature_embeddings is None:
            given_signature_embeddings = [None for _ in range(self.n_modalities)]

        self._setup_data_parameters(data)
        self._initialize(
            given_signatures=given_signatures,
            given_signature_embeddings=given_signature_embeddings,
            given_sample_embeddings=given_sample_embeddings,
            init_kwargs=init_kwargs,
        )
        of_values = [self.objective_function()]
        sof_values = [self.objective_function()]

        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 100 == 0:
                print("iteration ", n_iteration)

            self._update_alphas()
            ps = self._update_ps()
            self._update_LsU(ps, given_signature_embeddings, given_sample_embeddings)
            self._update_sigma_sq()
            self._update_Ws(ps, given_signatures)

            of_values.append(self.objective_function())
            prev_sof_value = sof_values[-1]
            sof_values.append(self._surrogate_objective_function(ps))
            rel_change = (sof_values[-1] - prev_sof_value) / np.abs(prev_sof_value)
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]
            self.history["surrogate_objective_function"] = sof_values[1:]

        return self

    @paper_style
    def plot_signatures(
        self,
        colors=None,
        annotate_mutation_types=False,
        figsize=None,
        outfile=None,
        **kwargs,
    ):
        if colors is None:
            colors = [None for _ in range(self.n_modalities)]

        max_n_signatures = np.max(self.ns_signatures)

        if figsize is None:
            figsize = (8 * self.n_modalities, 2 * max_n_signatures)

        fig, axes = plt.subplots(max_n_signatures, self.n_modalities, figsize=figsize)

        for axs, model, cols in zip(axes.T, self.models, colors):
            model.plot_signatures(
                colors=cols,
                annotate_mutation_types=annotate_mutation_types,
                axes=axs[: model.n_signatures],
                **kwargs,
            )

            for ax in axs[model.n_signatures :]:
                fig.delaxes(ax)

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes

    @paper_style
    def plot_exposures(
        self,
        reorder_signatures=True,
        annotate_samples=True,
        colors=None,
        ncol_legend=1,
        axes=None,
        outfile=None,
        **kwargs,
    ):
        """
        Visualize the exposures as a stacked bar chart,
        see plot.py for the implementation.

        Input:
        ------
        **kwargs:
            arguments to be passed to exposure_plot
        """
        if axes is None:
            _, axes = plt.subplots(
                self.n_modalities, figsize=(20, 3 * self.n_modalities)
            )

        if colors is None:
            colors = [None for _ in range(self.n_modalities)]

        for n, (ax, model, cols) in enumerate(zip(axes, self.models, colors)):
            ax = model.plot_exposures(
                reorder_signatures=reorder_signatures,
                annotate_samples=annotate_samples,
                colors=cols,
                ncol_legend=ncol_legend,
                ax=ax,
                **kwargs,
            )
            ax.set_title(f"{self.modality_names[n]} signature exposures")

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes

    @property
    def corr_signatures(self) -> pd.DataFrame:
        Ls = np.concatenate([model.L for model in self.models], axis=1)
        signature_names = np.concatenate(
            [model.signature_names for model in self.models]
        )
        norms = np.sqrt(np.sum(Ls**2, axis=0))

        corr_vector = np.array(
            [
                np.dot(l1, l2) / (norms[k1] * norms[k1 + k2 + 1])
                for k1, l1 in enumerate(Ls.T)
                for k2, l2 in enumerate(Ls[:, k1 + 1 :].T)
            ]
        )
        corr_matrix = squareform(corr_vector) + np.identity(np.sum(self.ns_signatures))
        corr = pd.DataFrame(corr_matrix, index=signature_names, columns=signature_names)

        return corr

    @property
    def corr_samples(self) -> pd.DataFrame:
        return self.models[0].corr_samples

    @paper_style
    def plot_correlation(self, data="signatures", annot=False, outfile=None, **kwargs):
        """
        Plot the correlation matrix of the signatures or samples.
        See plot.py for the implementation of corr_plot.

        Input:
        ------
        *args, **kwargs:
            arguments to be passed to corr_plot
        """
        value_checker("data", data, ["signatures", "samples"])

        if data == "signatures":
            corr = self.corr_signatures

        else:
            corr = self.corr_samples

        clustergrid = corr_plot(corr, annot=annot, **kwargs)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return clustergrid

    def _get_embedding_annotations(self, annotate_signatures, annotate_samples):
        # Only annotate with the first 20 characters of names
        annotations = np.empty(np.sum(self.ns_signatures) + self.n_samples, dtype="U20")

        if annotate_signatures:
            signature_names = np.concatenate(
                [model.signature_names for model in self.models]
            )
            annotations[: len(signature_names)] = signature_names

        if annotate_samples:
            annotations[-self.n_samples :] = self.models[0].sample_names

        return annotations

    @paper_style
    def plot_embeddings(
        self,
        method="umap",
        annotate_signatures=True,
        annotate_samples=False,
        normalize=False,
        ax=None,
        outfile=None,
        **kwargs,
    ):
        """
        Plot the signature and sample embeddings. If the embedding dimension
        is two, the embeddings will be plotted directly, ignoring the chosen method.
        See plot.py for the implementation of scatter_2d, tsne_2d, pca_2d, umap_2d.

        Input:
        ------
        methdod: str
            Either 'tsne', 'pca' or 'umap'. The respective dimensionality reduction
            will be applied to plot the signature and sample embeddings in 2D space.

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

        Ls = np.concatenate([model.L for model in self.models], axis=1)
        data = np.concatenate([Ls, self.models[0].U], axis=1).T

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
            ax = scatter_1d(data[:, 0], annotations=annotations, ax=ax, **kwargs)

        elif self.dim_embeddings == 2:
            ax = scatter_2d(data, annotations=annotations, ax=ax, **kwargs)

        elif method == "tsne":
            ax = tsne_2d(data, annotations=annotations, ax=ax, **kwargs)

        elif method == "pca":
            ax = pca_2d(data, annotations=annotations, ax=ax, **kwargs)

        else:
            ax = umap_2d(data, annotations=annotations, ax=ax, **kwargs)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax

    def feature_change(self, in_modality=None, out_modalities="all", normalize=True):
        if in_modality is None:
            in_modality = self.modality_names[0]

        in_model = self.models[list(self.modality_names).index(in_modality)]

        if out_modalities == "all":
            out_modalities = self.modality_names

        if type(out_modalities) is str:
            out_modalities = [out_modalities]

        out_modalities = [name for name in out_modalities if name != in_modality]
        out_modalities_indices = [
            n for n, name in enumerate(self.modality_names) if name in out_modalities
        ]
        results = [in_model.signatures]

        for n in out_modalities_indices:
            result = self.models[n].signatures @ np.exp(self.models[n].L.T @ in_model.L)
            result.columns = in_model.signature_names

            if normalize:
                result = result / result.sum(axis=0)

            results.append(result)

        return results

    @paper_style
    def plot_feature_change(
        self,
        in_modality=None,
        out_modalities="all",
        normalize=True,
        colors=None,
        annotate_mutation_types=False,
        figsize=None,
        outfile=None,
        **kwargs,
    ):
        # result[0] are the 'in_modality' signatures
        results = self.feature_change(in_modality, out_modalities, normalize)
        n_signatures = results[0].shape[1]
        n_feature_spaces = len(results)

        if colors is None:
            colors = [None for _ in range(n_feature_spaces)]

        if figsize is None:
            figsize = (8 * n_feature_spaces, 2 * n_signatures)

        fig, axes = plt.subplots(n_signatures, n_feature_spaces, figsize=figsize)
        fig.suptitle("Signature feature change")

        for axs, result, cols in zip(axes.T, results, colors):
            signatures_plot(
                result,
                colors=cols,
                annotate_mutation_types=annotate_mutation_types,
                axes=axs,
                **kwargs,
            )

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return axes
