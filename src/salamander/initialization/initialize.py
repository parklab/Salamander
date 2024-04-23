"""
Initialization methods for non-negative matrix factorization (NMF) models.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from ..utils import normalize_WH, type_checker, value_checker
from .methods import (
    _INIT_METHODS,
    _Init_methods,
    init_custom,
    init_flat,
    init_nndsvd,
    init_random,
    init_separableNMF,
)

EPSILON = np.finfo(np.float32).eps


def initialize_mat(
    data_mat: np.ndarray,
    n_signatures: int,
    method: _Init_methods = "nndsvd",
    given_signatures_mat: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the signature and exposure matrices.

    Inputs
    ------
    data_mat : np.ndarray
        shape (n_samples, n_features)

    n_signatures : int

    method : str
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
    value_checker("method", method, _INIT_METHODS)

    if method == "custom":
        matrices = init_custom(data_mat, n_signatures, **kwargs)
    elif method == "flat":
        matrices = init_flat(data_mat, n_signatures)
    elif method in ["nndsvd", "nndsvda", "nndsvdar"]:
        # mypy does not recognize that 'method' is compatible
        # with Literal["nndsvd", "nndsvda", "nndsvdar"]
        matrices = init_nndsvd(
            data_mat, n_signatures, method=method, **kwargs  # type: ignore[arg-type] # noqa: E501
        )
    elif method == "random":
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
    method: _Init_methods = "nndsvd",
    given_asignatures: ad.AnnData | None = None,
    **kwargs,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Initialize the signature ann data object and the exposure matrix.

    Inputs
    ------
    adata : ad.AnnData

    n_signatures : int

    method : str
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
        adata.X, n_signatures, method, given_signatures_mat, **kwargs
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
