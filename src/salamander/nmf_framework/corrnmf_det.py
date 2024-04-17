from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from . import _utils_corrnmf
from ._utils_klnmf import update_W
from .corrnmf import CorrNMF
from .initialization import EPSILON

if TYPE_CHECKING:
    from typing import Any


class CorrNMFDet(CorrNMF):
    """
    An implementation of a variant of the deterministic batch version of
    correlated NMF.

    "Bayesian Nonnegative Matrix Factorization with Stochastic Variational
    Inference" by Paisley et al.

    Reference
    ---------
    JW Paisley, DM Blei, MI Jordan: Bayesian Nonnegative Matrix Factorization
    with Stochastic Variational Inference, 2014
    """

    def update_sample_scalings(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "sample_scalings" not in given_parameters:
            self.adata.obs["scalings"] = _utils_corrnmf.update_sample_scalings(
                self.adata.X,
                self.asignatures.obs["scalings"].values,
                self.asignatures.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

    def update_signature_scalings(
        self, p: np.ndarray, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_scalings" not in given_parameters:
            self.asignatures.obs["scalings"] = _utils_corrnmf.update_signature_scalings(
                self.adata.X,
                p,
                self.adata.obs["scalings"].values,
                self.asignatures.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

    def update_variance(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "variance" not in given_parameters:
            embeddings = np.concatenate(
                [self.asignatures.obsm["embeddings"], self.adata.obsm["embeddings"]]
            )
            variance = np.mean(embeddings**2)
            self.variance = np.clip(variance, EPSILON, None)

    def update_signatures(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "asignatures" in given_parameters:
            n_given_signatures = given_parameters["asignatures"].n_obs
        else:
            n_given_signatures = 0

        W = update_W(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            n_given_signatures=n_given_signatures,
        )
        self.asignatures.X = W.T

    def update_p(self) -> np.ndarray:
        p = _utils_corrnmf.update_p_unnormalized(
            self.asignatures.X, self.adata.obsm["exposures"]
        )
        p /= np.sum(p, axis=1, keepdims=True)
        p = p.clip(EPSILON)
        return p

    def update_signature_embedding(
        self, index: int, aux_row: np.ndarray, outer_prods_sample_embeddings: np.ndarray
    ) -> None:
        scaling = self.asignatures.obs["scalings"][index]

        def objective_fun(embedding):
            return _utils_corrnmf.objective_function_embedding(
                embedding,
                self.adata.obsm["embeddings"],
                scaling,
                self.adata.obs["scalings"].values,
                self.variance,
                aux_row,
            )

        summand_grad = np.sum(
            aux_row[:, np.newaxis] * self.adata.obsm["embeddings"], axis=0
        )

        def gradient(embedding):
            return _utils_corrnmf.gradient_embedding(
                embedding,
                self.adata.obsm["embeddings"],
                scaling,
                self.adata.obs["scalings"].values,
                self.variance,
                summand_grad,
            )

        def hessian(embedding):
            return _utils_corrnmf.hessian_embedding(
                embedding,
                self.adata.obsm["embeddings"],
                scaling,
                self.adata.obs["scalings"].values,
                self.variance,
                outer_prods_sample_embeddings,
            )

        embedding = optimize.minimize(
            fun=objective_fun,
            x0=self.asignatures.obsm["embeddings"][index, :],
            method="Newton-CG",
            jac=gradient,
            hess=hessian,
        ).x
        embedding[(0 < embedding) & (embedding < EPSILON)] = EPSILON
        embedding[(-EPSILON < embedding) & (embedding < 0)] = -EPSILON
        self.asignatures.obsm["embeddings"][index, :] = embedding

    def update_signature_embeddings(
        self, aux: np.ndarray, outer_prods_sample_embeddings: np.ndarray | None = None
    ) -> None:
        r"""
        Update all signature embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method.

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        if outer_prods_sample_embeddings is None:
            outer_prods_sample_embeddings = np.einsum(
                "Dm,Dn->Dmn",
                self.adata.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

        for k, aux_row in enumerate(aux):
            self.update_signature_embedding(k, aux_row, outer_prods_sample_embeddings)

    def update_sample_embedding(
        self,
        index: int,
        aux_col: np.ndarray,
        outer_prods_signature_embeddings: np.ndarray,
    ) -> None:
        scaling = self.adata.obs["scalings"][index]

        def objective_fun(embedding):
            return _utils_corrnmf.objective_function_embedding(
                embedding,
                self.asignatures.obsm["embeddings"],
                scaling,
                self.asignatures.obs["scalings"].values,
                self.variance,
                aux_col,
            )

        summand_grad = np.sum(
            aux_col[:, np.newaxis] * self.asignatures.obsm["embeddings"], axis=0
        )

        def gradient(embedding):
            return _utils_corrnmf.gradient_embedding(
                embedding,
                self.asignatures.obsm["embeddings"],
                scaling,
                self.asignatures.obs["scalings"].values,
                self.variance,
                summand_grad,
            )

        def hessian(embedding):
            return _utils_corrnmf.hessian_embedding(
                embedding,
                self.asignatures.obsm["embeddings"],
                scaling,
                self.asignatures.obs["scalings"].values,
                self.variance,
                outer_prods_signature_embeddings,
            )

        embedding = optimize.minimize(
            fun=objective_fun,
            x0=self.adata.obsm["embeddings"][index, :],
            method="Newton-CG",
            jac=gradient,
            hess=hessian,
            options={"maxiter": 3},
        ).x
        embedding[(0 < embedding) & (embedding < EPSILON)] = EPSILON
        embedding[(-EPSILON < embedding) & (embedding < 0)] = -EPSILON
        self.adata.obsm["embeddings"][index, :] = embedding

    def update_sample_embeddings(self, aux: np.ndarray) -> None:
        r"""
        Update all sample embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method (strictly convex for each embedding).

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        outer_prods_signature_embeddings = np.einsum(
            "Km,Kn->Kmn",
            self.asignatures.obsm["embeddings"],
            self.asignatures.obsm["embeddings"],
        )

        for d, aux_col in enumerate(aux.T):
            self.update_sample_embedding(d, aux_col, outer_prods_signature_embeddings)

    def update_embeddings(
        self,
        p: np.ndarray,
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        aux = np.einsum("dv,vkd->kd", self.adata.X, p)

        if "signature_embeddings" not in given_parameters:
            self.update_signature_embeddings(aux)

        if "sample_embeddings" not in given_parameters:
            self.update_sample_embeddings(aux)

    def _update_parameters(self, given_parameters: dict[str, Any]) -> None:
        self.update_sample_scalings(given_parameters)
        p = self.update_p()
        self.update_signature_scalings(p, given_parameters)
        self.update_embeddings(p, given_parameters)
        self.update_variance(given_parameters)

        if "asignatures" in given_parameters:
            n_given_signatures = given_parameters["asignatures"].n_obs
        else:
            n_given_signatures = 0

        if n_given_signatures < self.n_signatures:
            self.update_signatures(given_parameters)

        self.compute_exposures()
