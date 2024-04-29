from __future__ import annotations

import numpy as np
from numba import njit
from scipy import optimize

from ..initialization.initialize import EPSILON
from ._utils_klnmf import poisson_llh


@njit
def compute_exposures(
    signature_scalings: np.ndarray,
    sample_scalings: np.ndarray,
    signature_embeddings: np.ndarray,
    sample_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Get the exposure matrix of shape (n_samples, n_signatures).
    """
    return np.exp(
        signature_scalings[:, np.newaxis]
        + sample_scalings
        + signature_embeddings @ sample_embeddings.T
    ).T


@njit
def compute_aux(
    data_mat: np.ndarray, signatures_mat: np.ndarray, exposures_mat: np.ndarray
) -> np.ndarray:
    r"""
    'aux' is the numpy array of shape (n_signatures, n_samples) given by
        aux_kd = \sum_v x_vd * p_vkd
    It can be computed without explicitly storing the parameters p.
    Notice that 'aux' is sufficient to update all model parameters.
    There is no need to compute and store p.

    Inputs
    ------
    data_mat : np.ndarray
        shape (n_samples, n_features)

    signatures_mat : np.ndarray
        shape (n_signatures, n_features)

    exposures_mat : np.ndarray
        shape (n_samples, n_signatures)
    """
    error_ratios = data_mat / (exposures_mat @ signatures_mat)
    aux = exposures_mat.T * (signatures_mat @ error_ratios.T)
    return aux


def elbo_corrnmf(
    data_mat: np.ndarray,
    signatures_mat: np.ndarray,
    exposures_mat: np.ndarray,
    signature_embeddings: np.ndarray,
    sample_embeddings: np.ndarray,
    variance: float,
    penalize_sample_embeddings: bool = True,
) -> float:
    """
    The evidence lower bound (ELBO) of correlated NMF.

    Inputs
    ------
    data_mat : np.ndarray
        shape (n_samples, n_features)

    signatures_mat : np.ndarray
        shape (n_signatures, n_features)

    exposures_mat : np.ndarray
        shape (n_samples, n_signatures)

    signature_embeddings : np.ndarray
        shape (n_signatures, dim_embeddings)

    sample_embeddings : np.ndarray
        shape (n_samples, dim_embeddings)

    variance : float

    penalize_sample_embeddings : bool, default=True
        If False, the sample embedding penalty is not added.
        This is useful for the implementation of multimodal correlated NMF.
    """
    n_signatures, dim_embeddings = signature_embeddings.shape
    n_samples = sample_embeddings.shape[0]
    elbo = poisson_llh(data_mat.T, signatures_mat.T, exposures_mat.T)
    elbo -= 0.5 * dim_embeddings * n_signatures * np.log(2 * np.pi * variance)
    elbo -= np.sum(signature_embeddings**2) / (2 * variance)

    if penalize_sample_embeddings:
        elbo -= 0.5 * dim_embeddings * n_samples * np.log(2 * np.pi * variance)
        elbo -= np.sum(sample_embeddings**2) / (2 * variance)

    return elbo


@njit
def update_signature_scalings(
    aux: np.ndarray,
    sample_scalings: np.ndarray,
    signature_embeddings: np.ndarray,
    sample_embeddings: np.ndarray,
) -> np.ndarray:
    r"""
    Compute the new signature scalings according to the update rule of CorrNMF.

    Inputs
    ------
    aux : np.ndarray
        auxiliary parameters of shape (n_signatures, n_samples) with
        aux[k,d] = \sum_v x_vd p_vkd

    sample_scalings : np.ndarray
        shape (n_samples,)

    signature_embeddings : np.ndarray
        shape (n_signatures, dim_embeddings)

    sample_embeddings : np.ndarray
        shape (n_samples, dim_embeddings)

    Returns
    -------
    signature_scalings : np.ndarray
        shape (n_signatures,)
    """
    first_sum = np.sum(aux, axis=1)
    second_sum = np.sum(
        np.exp(sample_scalings + signature_embeddings @ sample_embeddings.T), axis=1
    )
    signature_scalings = np.log(first_sum) - np.log(second_sum)
    return signature_scalings


@njit
def update_sample_scalings(
    data_mat: np.ndarray,
    signature_scalings: np.ndarray,
    signature_embeddings: np.ndarray,
    sample_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute the new sample scalings according to the update rule of CorrNMF.

    Parameters
    ----------
    data_mat : np.ndarray
        shape (n_features, n_samples)

    signature_scalings : np.ndarray
        shape (n_signatures,)

    signature_embeddings : np.ndarray
        shape (n_signatures, dim_embeddings)

    sample_embeddings : np.ndarray
        shape (n_samples, dim_embeddings)

    Returns
    -------
    sample_scalings : np.ndarray
        shape (n_samples,)
    """
    first_sum = np.sum(data_mat, axis=1)
    second_sum = np.sum(
        np.exp(
            signature_scalings[:, np.newaxis]
            + signature_embeddings @ sample_embeddings.T
        ),
        axis=0,
    )
    sample_scalings = np.log(first_sum) - np.log(second_sum)
    return sample_scalings


@njit
def objective_function_embedding(
    embedding: np.ndarray,
    embeddings_other: np.ndarray,
    scaling: float | np.ndarray,
    scalings_other: np.ndarray,
    variance: float,
    aux_vector: np.ndarray,
) -> float:
    r"""
    The negative objective function of a signature or sample embedding in CorrNMF.

    Parameters
    ----------
    embedding : np.ndarray
        shape (dim_embeddings,)

    embeddings_other : np.ndarray
        shape (n_samples | n_signatures, dim_embeddings)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    scaling : float | np.ndarray
        The scaling of the signature or sample corresponding to the
        embedding.
        In correlated NMF, 'scaling' is a float. In multimodal correlated NMF,
        'scaling' is an array if 'embedding' is a sample embedding because
        it depends on the modality.

    scalings_other : np.ndarray
        shape (n_samples | n_signatures,)
        The scalings of all samples or all signatures.
        If 'embedding' is a signature embeddings, 'scalings_other'
        are all sample scalingss. If 'embedding' is a sample embedding,
        'scalings_other' are all sample scalings.

    variance : float

    aux_vector : np.ndarray of
        shape (n_signatures | n_samples,)
        A row or column of aux[k, d] = \sum_v X_dv * p_vkd,
        where X is the data matrix and p are the auxiliary parameters of CorrNMF.
        If 'embedding' is a signature embedding, the corresponding row is provided.
        If 'embedding' is a sample embedding, the corresponding column is provided.
    """
    n_embeddings_other = embeddings_other.shape[0]
    of_value = 0.0
    scalar_products = embeddings_other.dot(embedding)

    # aux_vector not necessarily contiguous:
    # np.dot(scalar_products, aux_vec) doesn't work
    for i in range(n_embeddings_other):
        of_value += scalar_products[i] * aux_vector[i]

    of_value -= np.sum(np.exp(scaling + scalings_other + scalar_products))
    of_value -= np.dot(embedding, embedding) / (2 * variance)
    return -of_value


@njit
def gradient_embedding(
    embedding: np.ndarray,
    embeddings_other: np.ndarray,
    scaling: float | np.ndarray,
    scalings_other: np.ndarray,
    variance: float,
    summand_grad: np.ndarray,
) -> np.ndarray:
    r"""
    The negative gradient of the objective function w.r.t. a signature or
    sample embedding in CorrNMF.

    Inputs
    ------
    embedding : np.ndarray
        shape (dim_embeddings,)

    embeddings_other : np.ndarray
        shape (n_samples | n_signatures, dim_embeddings)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    scaling : float
        The scaling of the signature or sample corresponding to the
        embedding.
        In correlated NMF, 'scaling' is a float. In multimodal correlated NMF,
        'scaling' is an array if 'embedding' is a sample embedding because
        it depends on the modality.

    scalings_other : np.ndarray
        shape (n_samples | n_signatures,)
        The scalings of all samples or all signatures.
        If 'embedding' is a signature embeddings, 'scalings_other'
        are all sample scalingss. If 'embedding' is a sample embedding,
        'scalings_other' are all sample scalings.

    variance : float

    summand_grad : np.ndarray
        shape (dim_embeddings,). A signature/sample-independent summand.
    """
    scalar_products = embeddings_other.dot(embedding)
    gradient = -np.sum(
        np.exp(scaling + scalings_other + scalar_products)[:, np.newaxis]
        * embeddings_other,
        axis=0,
    )
    gradient += summand_grad
    gradient -= embedding / variance
    return -gradient


@njit
def hessian_embedding(
    embedding: np.ndarray,
    embeddings_other: np.ndarray,
    scaling: float | np.ndarray,
    scalings_other: np.ndarray,
    variance: float,
    outer_prods_embeddings_other: np.ndarray,
) -> np.ndarray:
    r"""
    The negative Hessian of the objective function w.r.t. a signature or
    sample embedding in CorrNMF.

    Inputs
    ------
    embedding : np.ndarray
        shape (dim_embeddings,)

    embeddings_other : np.ndarray
        shape (n_samples | n_signatures, dim_embeddings)
        If 'embedding' is a signature embedding, 'embeddings_other' are
        all sample embeddings. If 'embedding' is a sample embedding,
        'embeddings_other' are all signature embeddings.

    scaling : float
        The scaling of the signature or sample corresponding to the
        embedding.
        In correlated NMF, 'scaling' is a float. In multimodal correlated NMF,
        'scaling' is an array if 'embedding' is a sample embedding because
        it depends on the modality.

    scalings_other : np.ndarray
        shape (n_samples | n_signatures,)
        The scalings of all samples or all signatures.
        If 'embedding' is a signature embeddings, 'scalings_other'
        are all sample scalingss. If 'embedding' is a sample embedding,
        'scalings_other' are all sample scalings.

    variance : float

    outer_prods_embeddings_other : np.ndarray
        shape (n_samples | n_signatures, dim_embeddings, dim_embeddings)
    """
    n_embeddings_other, dim_embeddings = embeddings_other.shape
    scalar_products = embeddings_other.dot(embedding)
    scalings = np.exp(scaling + scalings_other + scalar_products)
    hessian = np.zeros((dim_embeddings, dim_embeddings))

    for m1 in range(dim_embeddings):
        for m2 in range(dim_embeddings):
            for i in range(n_embeddings_other):
                hessian[m1, m2] -= scalings[i] * outer_prods_embeddings_other[i, m1, m2]
            if m1 == m2:
                hessian[m1, m2] -= 1 / variance

    return -hessian


def update_embedding(
    embedding_init: np.ndarray,
    embeddings_other: np.ndarray,
    scaling: float | np.ndarray,
    scalings_other: np.ndarray,
    variance: float,
    aux_vec: np.ndarray,
    outer_prods_embeddings_other: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Optimize a signature or sample embedding using scipy.
    """

    def objective_fun(embedding):
        return objective_function_embedding(
            embedding,
            embeddings_other,
            scaling,
            scalings_other,
            variance,
            aux_vec,
        )

    summand_grad = np.sum(aux_vec[:, np.newaxis] * embeddings_other, axis=0)

    def gradient(embedding):
        return gradient_embedding(
            embedding,
            embeddings_other,
            scaling,
            scalings_other,
            variance,
            summand_grad,
        )

    def hessian(embedding):
        return hessian_embedding(
            embedding,
            embeddings_other,
            scaling,
            scalings_other,
            variance,
            outer_prods_embeddings_other,
        )

    embedding = optimize.minimize(
        fun=objective_fun,
        x0=embedding_init,
        method="Newton-CG",
        jac=gradient,
        hess=hessian,
        **kwargs,
    ).x
    embedding[(0 < embedding) & (embedding < EPSILON)] = EPSILON
    embedding[(-EPSILON < embedding) & (embedding < 0)] = -EPSILON
    return embedding
