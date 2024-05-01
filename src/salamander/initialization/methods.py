"""
Initialization methods for non-negative matrix factorization (NMF)
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from sklearn.decomposition import _nmf as sknmf

from ..utils import shape_checker, type_checker

EPSILON = np.finfo(np.float32).eps
_Init_methods = Literal[
    "custom",
    "flat",
    "nndsvd",
    "nndsvda",
    "nndsvdar",
    "random",
    "separableNMF",
]
_INIT_METHODS = get_args(_Init_methods)


def init_custom(
    data_mat: np.ndarray,
    n_signatures: int,
    signatures_mat: np.ndarray,
    exposures_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform type and shape checks on custom signature and
    exposure matrix initializations.

    Inputs:
    -------
    data_mat: np.ndarray
        shape (n_samples, n_features)

    n_signatures: int

    signatures_mat: np.ndarray
        shape (n_signatures, n_features)

    exposures_mat: np.ndarray
        shape (n_samples, n_signatures)
    """
    type_checker("signatures_mat", signatures_mat, np.ndarray)
    type_checker("exposures_mat", exposures_mat, np.ndarray)
    n_samples, n_features = data_mat.shape
    shape_checker("signatures_mat", signatures_mat, (n_signatures, n_features))
    shape_checker("exposures_mat", exposures_mat, (n_samples, n_signatures))
    return signatures_mat, exposures_mat


def init_flat(data_mat: np.ndarray, n_signatures: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the signature and exposure matrices with one float, respectively.
    """
    n_features = data_mat.shape[1]
    signatures_mat = np.full((n_signatures, n_features), 1 / n_features)
    exposures = np.sum(data_mat, axis=1) / n_signatures
    exposures_mat = np.tile(exposures, (n_signatures, 1)).T
    return signatures_mat, exposures_mat


def init_nndsvd(
    data_mat: np.ndarray,
    n_signatures: int,
    method: Literal["nndsvd", "nndsvda", "nndsvdar"] = "nndsvd",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A wrapper around the non-negative double singular value decomposition (NNDSVD)
    initialization methods "nndsvd", "nndsvda" and "nndsvdar" from scikit-learn.
    """
    if seed is not None:
        np.random.seed(seed)

    # pylint: disable-next=W0212
    exposures_mat, signatures_mat = sknmf._initialize_nmf(
        data_mat, n_signatures, init=method
    )
    return signatures_mat, exposures_mat


def init_random(
    data_mat: np.ndarray, n_signatures: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize each signature by drawing from the uniform
    distribution on the simplex.
    Initialize the exposures of each sample as a scaled sample
    from the uniform distribution on a simplex.
    The scaling is chosen such that the expected total exposure is equal to
    the column sum of that sample in the count matrix X.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = data_mat.shape
    signatures_mat = np.random.dirichlet(np.ones(n_features), size=n_signatures)
    scaling = np.sum(data_mat, axis=1)
    exposures_mat = scaling[:, np.newaxis] * np.random.dirichlet(
        np.ones(n_signatures), size=n_samples
    )
    return signatures_mat, exposures_mat


def init_separableNMF(
    data_mat: np.ndarray, n_signatures: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    This code is following Algorithm 1 from "Fast and Robust Recursive
    Algorithms for Separable Nonnegative Matrix Factorization"
    (Gillis and Vavasis, 2013), with the canonical choice of
    f(x) = \| x \|_2^2 as the strongly convex function f satisfying
    Assumption 2 from the paper.
    """
    signature_indices = np.empty(n_signatures, dtype=int)
    R = data_mat.T / np.sum(data_mat.T, axis=0)

    for k in range(n_signatures):
        column_norms = np.sum(R**2, axis=0)
        kstar = np.argmax(column_norms)
        u = R[:, kstar]
        R = (np.identity(R.shape[0]) - np.outer(u, u) / column_norms[kstar]) @ R
        signature_indices[k] = kstar

    signatures_mat = data_mat[signature_indices, :].astype(float)
    signatures_mat /= signatures_mat.sum(axis=1)[:, np.newaxis]
    _, exposures_mat = init_random(data_mat, n_signatures, seed=seed)
    return signatures_mat, exposures_mat
