from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import gammaln

EPSILON = np.finfo(np.float32).eps
GIVEN_PARAMETERS_ALLOWED = ["asignatures"]


@njit(fastmath=True)
def kl_divergence(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, weights: np.ndarray | None = None
) -> float:
    r"""
    The generalized Kullback-Leibler divergence
    D_KL(X || WH) = \sum_vd X_vd * ln(X_vd / (WH)_vd) - \sum_vd X_vd + \sum_vd (WH)_vd.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    weights : np.ndarray of shape (n_samples,)
        per sample weights

    Returns
    -------
    result : float
    """
    V, D = X.shape
    WH = W @ H
    result = 0.0

    for d in range(D):
        summand_sample = 0.0

        for v in range(V):
            if X[v, d] != 0:
                summand_sample += X[v, d] * np.log(X[v, d] / WH[v, d])
                summand_sample -= X[v, d]
            summand_sample += WH[v, d]

        if weights is not None:
            summand_sample *= weights[d]

        result += summand_sample

    return result


def samplewise_kl_divergence(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Per sample (weighted) generalized Kullback-Leibler divergence D_KL(x || Wh).

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    weights : np.ndarray of shape (n_samples,)
        per sample weights

    Returns
    -------
    errors : np.ndarray of shape (n_samples,)
    """
    X_data = np.copy(X).astype(float)
    indices = X == 0
    X_data[indices] = EPSILON
    WH_data = W @ H
    WH_data[indices] = EPSILON

    s1 = np.einsum("vd,vd->d", X_data, np.log(X_data / WH_data))
    s2 = -np.sum(X, axis=0)
    s3 = np.dot(H.T, np.sum(W, axis=0))

    errors = s1 + s2 + s3

    if weights is not None:
        errors *= weights

    return errors


@njit(fastmath=True)
def _poisson_llh_wo_factorial(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """
    The Poisson log-likelihood generalized to X, W and H having
    non-negative real numbers without the summands involving the log-factorial
    of elements of X.
    Note:
        scipy-special, which is required to computed the log-factorial,
        is not supported by numba.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    Returns
    -------
    result : float
    """
    V, D = X.shape
    WH = W @ H
    result = 0.0

    for v in range(V):
        for d in range(D):
            if WH[v, d] != 0:
                result += X[v, d] * np.log(WH[v, d])
            result -= WH[v, d]

    return result


def poisson_llh(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """
    The Poisson log-likelihood generalized to X, W and H having
    non-negative real numbers.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    Returns
    -------
    result : float
    """
    result = _poisson_llh_wo_factorial(X, W, H)
    result -= np.sum(gammaln(1 + X))

    return result


@njit
def update_W(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    weights_kl: np.ndarray | None = None,
    n_given_signatures: int = 0,
) -> np.ndarray:
    """
    The multiplicative update rule of the signature matrix W
    under the constraint of normalized signatures. It can be shown
    that the generalized KL-divegence D_KL(X || WH) is decreasing
    under the implemented update rule.

    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    weights_kl : np.ndarray of shape (n_samples,)
        per sample weights in the KL-divergence loss

    n_given_signatures : int
        The number of known signatures which will not be updated.

    Returns
    -------
    W : np.ndarray of shape (n_features, n_signatures)
        updated signature matrix
    """
    n_signatures = W.shape[1]

    if n_given_signatures == n_signatures:
        return W

    aux = X / (W @ H)

    if weights_kl is not None:
        aux *= weights_kl

    W_updated = W * (aux @ H.T)
    W_updated /= W_updated.sum(axis=0)
    W_updated[:, :n_given_signatures] = W[:, :n_given_signatures].copy()
    W_updated[:, n_given_signatures:] = W_updated[:, n_given_signatures:].clip(EPSILON)

    return W_updated


@njit
def update_H(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    weights_kl: np.ndarray | None = None,
    weights_l_half: np.ndarray | None = None,
) -> np.ndarray:
    """
    The multiplicative update rule of the exposure matrix H
    under the constraint of normalized signatures.

    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    weights_kl : np.ndarray of shape (n_samples,)
        per sample weights in the KL-divergence loss

    weights_l_half : np.ndarray of shape (n_samples,)
        per sample l_half penalty weights. They can be used to induce
        sparse exposures.

    Returns
    -------
    H_updated : np.ndarray of shape (n_signatures, n_samples)
        The updated exposure matrix. If possible, the update is performed
        in-place.
    """
    aux = X / (W @ H)

    if weights_l_half is None:
        # in-place
        H *= W.T @ aux
        H = H.clip(EPSILON)
        return H

    intermediate = 4.0 * H * (W.T @ aux)

    if weights_kl is not None:
        intermediate *= weights_kl**2

    discriminant = 0.25 * weights_l_half**2 + intermediate
    H_updated = 0.25 * (weights_l_half / 2 - np.sqrt(discriminant)) ** 2

    if weights_kl is not None:
        H_updated /= weights_kl**2

    H_updated = H_updated.clip(EPSILON)
    return H_updated


@njit
def update_WH(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    weights_kl: np.ndarray | None = None,
    weights_l_half: np.ndarray | None = None,
    n_given_signatures: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A joint update rule for the signature matrix W and
    the exposure matrix H under the constraint of normalized
    signatures.

    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    weights_kl : np.ndarray of shape (n_samples,)
        per sample weights in the KL-divergence loss

    weights_l_half : np.ndarray of shape (n_samples,)
        per sample l_half penalty weights. They can be used to induce
        sparse exposures.

    n_given_signatures : int
        The number of known signatures which will not be updated.

    Returns
    -------
    W_updated : np.ndarray of shape (n_features, n_signatures)
        updated signature matrix

    H_updated : np.ndarray of shape (n_signatures, n_samples)
        The updated exposure matrix. If possible, the update is performed
        in-place.
    """
    n_signatures = W.shape[1]
    aux = X / (W @ H)

    if n_given_signatures == n_signatures:
        W_updated = W
    else:
        if weights_kl is None:
            scaled_aux = aux
        else:
            scaled_aux = weights_kl * aux
        # the old signatures are needed for updating H
        W_updated = W * (scaled_aux @ H.T)
        W_updated /= np.sum(W_updated, axis=0)
        W_updated[:, :n_given_signatures] = W[:, :n_given_signatures].copy()
        W_updated = W_updated.clip(EPSILON)

    if weights_l_half is None:
        # in-place
        H *= W.T @ aux
        H = H.clip(EPSILON)
        return W_updated, H

    intermediate = 4.0 * H * (W.T @ aux)

    if weights_kl is not None:
        intermediate *= weights_kl**2

    discriminant = 0.25 * weights_l_half**2 + intermediate
    H_updated = 0.25 * (weights_l_half / 2 - np.sqrt(discriminant)) ** 2

    if weights_kl is not None:
        H_updated /= weights_kl**2

    H_updated = H_updated.clip(EPSILON)
    return W_updated, H_updated
