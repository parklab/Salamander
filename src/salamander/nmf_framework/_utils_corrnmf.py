import numpy as np
from numba import njit

EPSILON = np.finfo(np.float32).eps


@njit
def update_alpha(
    X: np.ndarray, beta: np.ndarray, L: np.ndarray, U: np.ndarray
) -> np.ndarray:
    """
    Compute the new sample biases alpha according to the update rule of CorrNMF.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    beta : np.ndarray of shape (n_signatures,)
        signature biases

    L : np.ndarray of shape (dim_embeddings, n_signatures)
        signature embeddings

    U : np.ndarray of shape (dim_embeddings, n_samples)
        sample embeddings

    Returns
    -------
    alpha : np.ndarray of shape (n_samples,)
        The new sample biases
    """
    second_term = np.exp(beta[:, np.newaxis] + L.T @ U)
    alpha = np.log(np.sum(X, axis=0)) - np.log(np.sum(second_term, axis=0))
    return alpha


@njit
def update_beta(
    X: np.ndarray, p: np.ndarray, alpha: np.ndarray, L: np.ndarray, U: np.ndarray
) -> np.ndarray:
    """
    Compute the new signature biases beta according to the update rule of CorrNMF.

    Parameters
    ----------
    X : np.ndarray of shape (n_features, n_samples)
        data matrix

    alpha : np.ndarray of shape (n_samples,)
        sample biases

    L : np.ndarray of shape (dim_embeddings, n_signatures)
        signature embeddings

    U : np.ndarray of shape (dim_embeddings, n_samples)
        sample embeddings

    Returns
    -------
    beta : np.ndarray of shape (n_signatures,)
        The new signature biases
    """
    n_features, n_signatures, n_samples = p.shape
    first_term = np.zeros(n_signatures)

    for k in range(n_signatures):
        for v in range(n_features):
            for d in range(n_samples):
                first_term[k] += X[v, d] * p[v, k, d]

    second_term = np.exp(alpha + L.T @ U)
    beta = np.log(first_term) - np.log(np.sum(second_term, axis=1))
    return beta


@njit
def update_p_unnormalized(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the new auxiliary parameters according to the update rule of CorrNMF.
    The normalization per mutation type and sample is not performed yet.

    Parameters
    ----------
    W : np.ndarray of shape (n_features, n_signatures)
        signature matrix

    H : np.ndarray of shape (n_signatures, n_samples)
        exposure matrix

    Returns
    -------
    p: np.ndarray of shape (n_features, n_signatures, n_samples)
    """
    n_features, n_signatures = W.shape
    n_samples = H.shape[1]
    p = np.zeros((n_features, n_signatures, n_samples))

    for v in range(n_features):
        for k in range(n_signatures):
            for d in range(n_samples):
                p[v, k, d] = W[v, k] * H[k, d]

    return p


@njit
def objective_function_embedding(
    embedding,
    embeddings_other,
    alpha,
    beta,
    sigma_sq,
    aux_vec,
    add_penalty=True,
):
    r"""
    The negative objective function of a signature or sample embedding in CorrNMF.

    Parameters
    ----------
    embedding : np.ndarray of shape (dim_embeddings,)
        The signature or sample embedding

    embeddings_other : np.ndarray of shape (dim_embeddings, n_samples | n_signatures)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    alpha : float | np.narray of shape (n_samples,)
        If 'embedding' is a signature embedding, 'alpha' are
        all sample biases. If 'embedding' is a sample embedding,
        'alpha' is the bias of the corresponding sample.

    beta : np.array of shape (n_signatures,) | float
        If 'embedding' is a signature embedding, 'beta' is
        the corresponding signature bias. If 'embedding' is a sample embedding,
        'beta' are all signature biases.

    sigma_sq : float
        model variance

    aux_vec : np.ndarray of shape (n_signatures | n_samples,)
        A row or column of
        aux[k, d] = \sum_v X_vd * p_vkd,
        where X is the data matrix and p are the auxiliary parameters of CorrNMF.
        If 'embedding' is a signature embedding, the corresponding row is provided.
        If 'embedding' is a sample embedding, the corresponding column is provided.

    add_penalty : bool, default=True
        Set to True, the norm of the embedding will be penalized.
        This argument is useful for the implementation of multimodal CorrNMF.
    """
    n_embeddings_other = embeddings_other.shape[1]
    of_value = 0.0
    scalar_products = embeddings_other.T.dot(embedding)

    # aux_vec not necessarily contiguous:
    # np.dot(scalar_products, aux_vec) doesn't work
    for i in range(n_embeddings_other):
        of_value += scalar_products[i] * aux_vec[i]

    # works for alpha and beta being a scalar or vector
    of_value -= np.sum(np.exp(alpha + beta + scalar_products))

    if add_penalty:
        of_value -= np.dot(embedding, embedding) / (2 * sigma_sq)

    return -of_value


@njit
def gradient_embedding(
    embedding,
    embeddings_other,
    alpha,
    beta,
    sigma_sq,
    summand_grad,
    add_penalty=True,
):
    r"""
    The negative gradient of the objective function w.r.t. a signature or
    sample embedding in CorrNMF.

    Parameters
    ----------
    embedding : np.ndarray of shape (dim_embeddings,)
        The signature or sample embedding

    embeddings_other : np.ndarray of shape (dim_embeddings, n_samples | n_signatures)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    alpha : float | np.narray of shape (n_samples,)
        If 'embedding' is a signature embedding, 'alpha' are
        all sample biases. If 'embedding' is a sample embedding,
        'alpha' is the bias of the corresponding sample.

    beta : np.array of shape (n_signatures,) | float
        If 'embedding' is a signature embedding, 'beta' is
        the corresponding signature bias. If 'embedding' is a sample embedding,
        'beta' are all signature biases.

    sigma_sq : float
        model variance

    summand_grad : np.ndarray of shape (dim_embeddings,)
        A signature/sample-independent summand of the gradient.

    add_penalty : bool, default=True
        Set to True, the norm of the embedding will be penalized.
        This argument is useful for the implementation of multimodal CorrNMF.
    """
    scalar_products = embeddings_other.T.dot(embedding)
    # works for alpha and beta being a scalar or vector
    gradient = -np.sum(
        np.exp(alpha + beta + scalar_products) * embeddings_other, axis=1
    )
    gradient += summand_grad

    if add_penalty:
        gradient -= embedding / sigma_sq

    return -gradient


@njit
def hessian_embedding(
    embedding,
    embeddings_other,
    alpha,
    beta,
    sigma_sq,
    outer_prods_embeddings_other,
    add_penalty=True,
):
    r"""
    The negative Hessian of the objective function w.r.t. a signature or
    sample embedding in CorrNMF.

    Parameters
    ----------
    embedding : np.ndarray of shape (dim_embeddings,)
        The signature or sample embedding

    embeddings_other : np.ndarray of shape (dim_embeddings, n_samples | n_signatures)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    alpha : float | np.narray of shape (n_samples,)
        If 'embedding' is a signature embedding, 'alpha' are
        all sample biases. If 'embedding' is a sample embedding,
        'alpha' is the bias of the corresponding sample.

    beta : np.array of shape (n_signatures,) | float
        If 'embedding' is a signature embedding, 'beta' is
        the corresponding signature bias. If 'embedding' is a sample embedding,
        'beta' are all signature biases.

    sigma_sq : float
        model variance

    aux_vec : np.ndarray of shape (n_signatures | n_samples,)
        A row or column of
        aux[k, d] = \sum_v X_vd * p_vkd,
        where X is the data matrix and p are the auxiliary parameters of CorrNMF.
        If 'embedding' is a signature embedding, the corresponding row is provided.
        If 'embedding' is a sample embedding, the corresponding column is provided.

    add_penalty : bool, default=True
        Set to True, the norm of the embedding will be penalized.
        This argument is useful for the implementation of multimodal CorrNMF.
    """
    dim_embeddings, n_embeddings_other = embeddings_other.shape
    scalar_products = embeddings_other.T.dot(embedding)
    scalings = np.exp(alpha + beta + scalar_products)
    hessian = np.zeros((dim_embeddings, dim_embeddings))

    for m1 in range(dim_embeddings):
        for m2 in range(dim_embeddings):
            for i in range(n_embeddings_other):
                hessian[m1, m2] -= scalings[i] * outer_prods_embeddings_other[i, m1, m2]
            if add_penalty and m1 == m2:
                hessian[m1, m2] -= 1 / sigma_sq

    return -hessian
