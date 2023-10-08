import numpy as np
import pandas as pd
from scipy import optimize

from .corrnmf import CorrNMF

EPSILON = np.finfo(np.float32).eps


class CorrNMFDet(CorrNMF):
    r"""
    The CorrNMFDet class implements the deterministic batch version of
    a variant of the correlated NMF (CorrNMF) algorithm devolped in
    "Bayesian Nonnegative Matrix Factorization with Stochastic Variational
    Inference" by Paisley et al.

    The following methods are implemented to match the structure
    of the abstract class CorrNMF:

        - _update_alpha:
            update the sample exposure biases \alpha

        - _update_sigma_sq:
            update the variance \sigma^2 assumed in the generative model

        - _update_W:
            update the signature matrix W

        - _update_p:
            update the auxiliary parameters p

        - _update_l:
            update a single signature embedding l

        - _update_u:
            update a single sample embedding u

    The following method is implemented to match the structure of SignatureNMF:

        - fit:
            Perform CorrNMF for the given mutation count data or
            for given signatures and mutation count data
    """

    def _update_alpha(self):
        exp_LTU = np.exp(self.L.T @ self.U)
        self.alpha = np.log(np.sum(self.X, axis=0)) - np.log(np.sum(exp_LTU, axis=0))

    def _update_sigma_sq(self):
        sum_norm_sigs = np.sum(self.L**2)
        sum_norm_samples = np.sum(self.U**2)

        self.sigma_sq = (sum_norm_sigs + sum_norm_samples) / (
            self.dim_embeddings * (self.n_signatures + self.n_samples)
        )
        self.sigma_sq = np.clip(self.sigma_sq, EPSILON, None)

    def _update_W(self, p):
        if self.update_W == "1999-Lee":
            self.W = self.W * (
                (self.X / (self.W @ self.exposures.values)) @ self.exposures.values.T
            )

        else:
            self.W = np.einsum("vd,vkd->vk", self.X, p)

        self.W /= np.sum(self.W, axis=0)
        self.W = self.W.clip(EPSILON)

    def _update_p(self):
        p = np.einsum("vk,kd->vkd", self.W, self.exposures.values)
        p /= np.sum(p, axis=1, keepdims=True)
        p = p.clip(EPSILON)

        return p

    def _objective_fun_l(self, l, aux_row):
        UTl = self.U.T.dot(l)
        s = np.dot(aux_row, UTl)
        s -= np.sum(np.exp(self.alpha + UTl))
        s -= np.dot(l, l) / (2 * self.sigma_sq)

        return -s

    def _gradient_l(self, l, s_grad):
        s = -np.sum(np.exp(self.alpha + self.U.T.dot(l)) * self.U, axis=1)
        s -= l / self.sigma_sq

        return -(s_grad + s)

    def _hessian_l(self, l, outer_prods_U):
        scalings = np.exp(self.alpha + self.U.T.dot(l))
        s = -np.einsum("D,Dmn->mn", scalings, outer_prods_U)
        s -= np.diag(np.full(self.dim_embeddings, 1 / self.sigma_sq))

        return -s

    def _update_l(self, index, aux_row, outer_prods_U):
        def objective_fun(l):
            return self._objective_fun_l(l, aux_row)

        s_grad = np.sum(aux_row * self.U, axis=1)

        def gradient(l):
            return self._gradient_l(l, s_grad)

        def hessian(l):
            return self._hessian_l(l, outer_prods_U)

        self.L[:, index] = optimize.minimize(
            fun=objective_fun,
            x0=self.L[:, index],
            method="Newton-CG",
            jac=gradient,
            hess=hessian,
        ).x

    def _update_L(self, aux, outer_prods_U=None):
        r"""
        Update all signature embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method (strictly convex for each embedding).

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        if outer_prods_U is None:
            outer_prods_U = np.einsum("mD,nD->Dmn", self.U, self.U)

        for k, aux_row in enumerate(aux):
            self._update_l(k, aux_row, outer_prods_U)

        self.L[(0 < self.L) & (self.L < EPSILON)] = EPSILON
        self.L[(-EPSILON < self.L) & (self.L < 0)] = -EPSILON

    def _objective_fun_u(self, u, index, aux_col, add_penalty_u=True):
        LTu = self.L.T.dot(u)
        s = np.dot(aux_col, LTu)
        s -= np.sum(np.exp(self.alpha[index] + LTu))

        if add_penalty_u:
            s -= np.dot(u, u) / (2 * self.sigma_sq)

        return -s

    def _gradient_u(self, u, index, s_grad, add_penalty_u=True):
        s = -np.exp(self.alpha[index]) * np.sum(
            np.exp(self.L.T.dot(u)) * self.L, axis=1
        )

        if add_penalty_u:
            s -= u / self.sigma_sq

        return -(s_grad + s)

    def _hessian_u(self, u, index, outer_prods_L, add_penalty_u=True):
        scalings = np.exp(self.alpha[index] + self.L.T.dot(u))
        s = -np.einsum("K,Kmn->mn", scalings, outer_prods_L)

        if add_penalty_u:
            s -= np.diag(np.full(self.dim_embeddings, 1 / self.sigma_sq))

        return -s

    def _update_u(self, index, aux_col, outer_prods_L):
        def objective_fun(u):
            return self._objective_fun_u(u, index, aux_col)

        s_grad = np.sum(aux_col * self.L, axis=1)

        def gradient(u):
            return self._gradient_u(u, index, s_grad)

        def hessian(u):
            return self._hessian_u(u, index, outer_prods_L)

        u = optimize.minimize(
            fun=objective_fun,
            x0=self.U[:, index],
            method="Newton-CG",
            jac=gradient,
            hess=hessian,
            options={"maxiter": 3},
        ).x
        u[(0 < u) & (u < EPSILON)] = EPSILON
        u[(-EPSILON < u) & (u < 0)] = -EPSILON
        self.U[:, index] = u

    def _update_U(self, aux):
        r"""
        Update all sample embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method (strictly convex for each embedding).

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        outer_prods_L = np.einsum("mK,nK->Kmn", self.L, self.L)

        for d, aux_col in enumerate(aux.T):
            self._update_u(d, aux_col, outer_prods_L)

    def _update_LU(self, p, given_signature_embeddings, given_sample_embeddings):
        aux = np.einsum("vd,vkd->kd", self.X, p)

        if given_signature_embeddings is None:
            self._update_L(aux)

        if given_sample_embeddings is None:
            self._update_U(aux)

    def fit(
        self,
        data: pd.DataFrame,
        given_signatures=None,
        given_signature_embeddings=None,
        given_sample_embeddings=None,
        init_kwargs=None,
        history=False,
        verbose=0,
    ):
        """
        Maximize the surrogate objective function of correlated NMF (CNMF).

        Input:
        ------
        data: pd.DataFrame
            The mutation count data

        given_signatures: pd.DataFrame, default=None
            Known signatures which will be fixed during model fitting.

        given_signature_embeddings: np.ndarray, default=None
            Known signature embeddings which will be fixed during model fitting.

        given_sample_embeddings: np.ndarray, default=None
            Known sample embeddings which will be fixed during model fitting.

        init_kwargs: dict
            Any further keywords arguments to be passed to the initialization method.
            This includes, for example, a possible 'seed' keyword argument
            for all stochastic methods.

        history: bool
            When set to true, the history of the objective function and
            surrogate objective function will be stored in a dictionary.

        verbose: int
            Every 100th iteration number will be printed when set unequal to zero.
        """
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

            self._update_alpha()
            p = self._update_p()
            self._update_LU(p, given_signature_embeddings, given_sample_embeddings)
            self._update_sigma_sq()

            if given_signatures is None:
                self._update_W(p)

            of_values.append(self.objective_function())
            prev_sof_value = sof_values[-1]
            sof_values.append(self._surrogate_objective_function(p))
            rel_change = (sof_values[-1] - prev_sof_value) / np.abs(prev_sof_value)
            converged = (
                rel_change < self.tol and n_iteration >= self.min_iterations
            ) or (n_iteration >= self.max_iterations)

        if history:
            self.history["objective_function"] = of_values[1:]
            self.history["surrogate_objective_function"] = sof_values[1:]

        return self
