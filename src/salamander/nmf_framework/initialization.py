"""
Initialization methods for non-negative matrix factorization (NMF)
"""

from __future__ import annotations

from typing import Literal, get_args

import anndata as ad
import numpy as np
from sklearn.decomposition import _nmf as sknmf

from ..utils import normalize_WH, shape_checker, type_checker, value_checker

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
    n_samples, n_features = data_mat.shape
    scaling = np.mean(np.sum(data_mat, axis=1))
    signatures_mat = np.full((n_signatures, n_features), 1 / n_features)
    exposures_mat = np.full((n_samples, n_signatures), scaling / n_signatures)
    return signatures_mat, exposures_mat


def init_nndsvd(
    data_mat: np.ndarray,
    n_signatures: int,
    init: Literal["nndsvd", "nndsvda", "nndsvdar"] = "nndsvd",
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
        data_mat, n_signatures, init=init
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


def initialize_mat(
    data_mat: np.ndarray,
    n_signatures: int,
    init_method: _Init_methods = "nndsvd",
    given_signatures_mat: np.ndarray | None = None,
    **kwargs,
) -> [np.ndarray, np.ndarray]:
    """
    Initialize the signature and exposure matrices.

    Inputs
    ------
    data_mat : np.ndarray
        shape (n_samples, n_features)

    n_signatures : int

    init_method : str
        initialization method. One of 'custom', 'flat',
        'nndsvd', 'nndsvda', 'nndsvdar', 'random', 'separableNMF'.

    given_signatures_mat : np.ndarray, optional
        At most 'n_signatures' many signatures can be provided to
        overwrite some of the initialized signatures. This does not
        change the initialized exposures.

    kwargs : dict
        Any keyword arguments to be passed to the initialization method.
        This includes, for example, a possible 'seed' keyword argument
        for all stochastic methods.

    Returns
    -------
    signatures_mat : np.ndarray
        shape (n_signatures, n_features)

    exposures_mat : np.ndarray
        shape (n_samples, n_signatures)
    """
    value_checker("init_method", init_method, _INIT_METHODS)

    if init_method == "custom":
        matrices = init_custom(data_mat, n_signatures, **kwargs)
    elif init_method == "flat":
        matrices = init_flat(data_mat, n_signatures)
    elif init_method in ["nndsvd", "nndsvda", "nndsvdar"]:
        # mypy does not recognize that init_method is compatible
        # with Literal["nndsvd", "nndsvda", "nndsvdar"]
        matrices = init_nndsvd(
            data_mat, n_signatures, init=init_method, **kwargs  # type: ignore[arg-type] # noqa: E501
        )
    elif init_method == "random":
        matrices = init_random(data_mat, n_signatures, **kwargs)
    else:
        matrices = init_separableNMF(data_mat, n_signatures, **kwargs)

    signatures_mat, exposures_mat = matrices

    if given_signatures_mat is not None:
        type_checker("given_signatures_mat", given_signatures_mat, np.ndarray)
        given_n_signatures, given_n_features = given_signatures_mat.shape

        if given_n_features != data_mat.shape[1]:
            raise ValueError(
                "The given signature matrix has a different number of features "
                "than the data."
            )
        if given_n_signatures > n_signatures:
            raise ValueError("The given signature matrix contains too many signatures.")

        signatures_mat[:given_n_signatures, :] = given_signatures_mat.copy()

    W, H = normalize_WH(signatures_mat.T, exposures_mat.T)
    W, H = W.clip(EPSILON), H.clip(EPSILON)
    signatures_mat, exposures_mat = W.T, H.T
    return signatures_mat, exposures_mat


def initialize(
    adata: ad.AnnData,
    n_signatures: int,
    init_method: _Init_methods = "nndsvd",
    given_asignatures: ad.AnnData | None = None,
    **kwargs,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Initialize the signature ann data object and the exposure matrix.

    Inputs
    ------
    adata : ad.AnnData

    n_signatures : int

    init_method : str
        initialization method. One of 'custom', 'flat',
        'nndsvd', 'nndsvda', 'nndsvdar', 'random', 'separableNMF'.

    given_asignatures : ad.AnnData, optional
        At most 'n_signatures' many signatures can be provided to
        overwrite some of the initialized signatures. This does not
        change the initialized exposures.

    kwargs : dict
        Any keyword arguments to be passed to the initialization method.
        This includes, for example, a possible 'seed' keyword argument
        for all stochastic methods.

    Returns
    -------
    asignatures : ad.AnnData
        Annotated signature matrix of shape (n_signatures, n_features)

    exposures_mat : np.ndarray
        shape (n_samples, n_signatures)
    """
    if given_asignatures is not None:
        if given_asignatures.n_vars != adata.n_vars:
            raise ValueError(
                "The given signatures have a different number of features "
                "than the data."
            )
        if not all(given_asignatures.var_names == adata.var_names):
            raise ValueError(
                "The features of the given signatures and the data are not identical."
            )
        given_signatures_mat = given_asignatures.X
    else:
        given_signatures_mat = None

    signatures_mat, exposures_mat = initialize_mat(
        adata.X, n_signatures, init_method, given_signatures_mat, **kwargs
    )
    asignatures = ad.AnnData(signatures_mat)
    asignatures.var_names = adata.var_names
    asignatures.obs_names = [f"Sig{k+1}" for k in range(n_signatures)]

    # keep signature annotations
    if given_asignatures is not None:
        n_given_signatures = given_asignatures.n_obs
        asignatures.obs_names = np.roll(asignatures.obs_names, n_given_signatures)
        asignatures = ad.concat(
            [given_asignatures, asignatures[n_given_signatures:, :]], join="outer"
        )

    return asignatures, exposures_mat
