from __future__ import annotations

from typing import Any

import numpy as np

from ..initialization.initialize import EPSILON
from . import _utils_corrnmf
from ._utils_klnmf import update_W
from .corrnmf import CorrNMF


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

    def _compute_aux(self) -> np.ndarray:
        return _utils_corrnmf.compute_aux(
            self.adata.X, self.asignatures.X, self.adata.obsm["exposures"]
        )

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
        self, aux: np.ndarray, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_scalings" not in given_parameters:
            self.asignatures.obs["scalings"] = _utils_corrnmf.update_signature_scalings(
                aux,
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

    def update_signature_embeddings(self, aux: np.ndarray) -> None:
        r"""
        Update all signature embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method.

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        outer_prods_sample_embeddings = np.einsum(
            "Dm,Dn->Dmn",
            self.adata.obsm["embeddings"],
            self.adata.obsm["embeddings"],
        )
        for k, aux_row in enumerate(aux):
            embedding_init = self.asignatures.obsm["embeddings"][k, :]
            self.asignatures.obsm["embeddings"][k, :] = _utils_corrnmf.update_embedding(
                embedding_init,
                self.adata.obsm["embeddings"],
                self.asignatures.obs["scalings"][k],
                self.adata.obs["scalings"].values,
                self.variance,
                aux_row,
                outer_prods_sample_embeddings,
            )

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
            embedding_init = self.adata.obsm["embeddings"][d, :]
            self.adata.obsm["embeddings"][d, :] = _utils_corrnmf.update_embedding(
                embedding_init,
                self.asignatures.obsm["embeddings"],
                self.adata.obs["scalings"][d],
                self.asignatures.obs["scalings"].values,
                self.variance,
                aux_col,
                outer_prods_signature_embeddings,
                options={"maxiter": 3},
            )

    def update_embeddings(
        self,
        aux: np.ndarray,
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_embeddings" not in given_parameters:
            self.update_signature_embeddings(aux)

        if "sample_embeddings" not in given_parameters:
            self.update_sample_embeddings(aux)

    def _update_parameters(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        self.update_sample_scalings(given_parameters)
        self.compute_exposures()
        aux = self._compute_aux()
        self.update_signature_scalings(aux, given_parameters)
        self.update_embeddings(aux, given_parameters)
        self.update_variance(given_parameters)
        self.update_signatures(given_parameters)
