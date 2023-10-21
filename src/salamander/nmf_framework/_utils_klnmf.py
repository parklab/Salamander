import numpy as np
from numba import njit
from scipy.special import gammaln

EPSILON = np.finfo(np.float32).eps


@njit(fastmath=True)
def kl_divergence(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
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

    Returns
    -------
    result : float
    """
    V, D = X.shape
    WH = W @ H
    result = 0.0

    for v in range(V):
        for d in range(D):
            if X[v, d] != 0:
                result += X[v, d] * np.log(X[v, d] / WH[v, d])
                result -= X[v, d]
            result += WH[v, d]

    return result


def samplewise_kl_divergence(X, W, H):
    """
    Per sample generalized Kullback-Leibler divergence D_KL(x || Wh).

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
    X: np.ndarray, W: np.ndarray, H: np.ndarray, n_given_signatures: int = 0
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

    n_given_signatures : int
        The number of known signatures, which will not be updated.

    Returns
    -------
    W : np.ndarray of shape (n_features, n_signatures)
        updated signature matrix
    """
    W_updated = W * ((X / (W @ H)) @ H.T)
    W_updated /= W_updated.sum(axis=0)
    W_updated[:, :n_given_signatures] = W[:, :n_given_signatures].copy()
    W_updated[:, n_given_signatures:] = W_updated[:, n_given_signatures:].clip(EPSILON)

    return W_updated


@njit
def update_H(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
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

    Returns
    -------
    H : np.ndarray of shape (n_signatures, n_samples)
        updated exposure matrix

    Reference
    ---------
    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """
    H *= W.T @ (X / (W @ H))
    H = H.clip(EPSILON)

    return H


@njit
def update_WH(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, n_given_signatures: int = 0
) -> np.ndarray:
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

    n_given_signatures : int
        The number of known signatures, which will not be updated.

    Returns
    -------
    W : np.ndarray of shape (n_features, n_signatures)
        updated signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        updated exposure matrix
    """
    n_signatures = W.shape[1]
    aux = X / (W @ H)

    if n_given_signatures < n_signatures:
        # the old signatures are needed for updating H
        W_updated = W * (aux @ H.T)
        W_updated /= np.sum(W_updated, axis=0)
        W_updated[:, :n_given_signatures] = W[:, :n_given_signatures].copy()
        W_updated = W_updated.clip(EPSILON)
    else:
        W_updated = W

    H *= W.T @ aux
    H = H.clip(EPSILON)

    return W_updated, H
