"""
Initialization methods for non-negative matrix factorization (NMF) models.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import mudata as md
import numpy as np

from ..utils import (
    dict_checker,
    normalize_WH,
    shape_checker,
    type_checker,
    value_checker,
)
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

# allowed given parameters
GIVEN_PARAMETERS_STANDARD_NMF = ["asignatures"]
GIVEN_PARAMETERS_CORRNMF = [
    "asignatures",
    "signature_scalings",
    "sample_scalings",
    "signature_embeddings",
    "sample_embeddings",
    "variance",
]


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


def check_given_asignatures(
    given_asignatures: ad.AnnData, adata: ad.AnnData, n_signatures: int
) -> None:
    """
    Check if the given signatures are compatible with
    the input data and the number of signatures to be initialized.
    The number of given signatures can be less or equal to the number of
    signatures specified.

    Inputs
    ------
    given_asignatures: AnnData
        Known signatures that should be fixed by the algorithm.

    adata: ad.AnnData
        Input data.

    n_signatures: int
        The number of signatures to initialize.
    """
    type_checker("given_asignatures", given_asignatures, ad.AnnData)
    if given_asignatures.n_vars != adata.n_vars:
        raise ValueError(
            "The given signatures have a different number of features than the data."
        )
    if not all(given_asignatures.var_names == adata.var_names):
        raise ValueError(
            "The features of the given signatures and the data are not identical."
        )
    if given_asignatures.n_obs > n_signatures:
        raise ValueError(
            "The number of given signatures exceeds "
            "the number of signatures to initialize."
        )


def initialize_base(
    adata: ad.AnnData,
    n_signatures: int,
    method: _Init_methods = "nndsvd",
    given_asignatures: ad.AnnData | None = None,
    **kwargs,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Initialize the signature anndata object and the exposure matrix.
    The anndata object is unchanged and the exposure matrix is returned.

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
        check_given_asignatures(given_asignatures, adata, n_signatures)
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


def check_given_parameters_standard_nmf(
    adata: ad.AnnData,
    n_signatures: int,
    given_parameters: dict[str, Any],
) -> None:
    dict_checker("given_parameters", given_parameters, GIVEN_PARAMETERS_STANDARD_NMF)

    if "asignatures" in given_parameters:
        check_given_asignatures(given_parameters["asignatures"], adata, n_signatures)


def initialize_standard_nmf(
    adata: ad.AnnData,
    n_signatures: int,
    method: _Init_methods = "nndsvd",
    given_parameters: dict[str, Any] | None = None,
    **kwargs,
) -> ad.AnnData:
    given_parameters = {} if given_parameters is None else given_parameters.copy()
    check_given_parameters_standard_nmf(adata, n_signatures, given_parameters)

    if "asignatures" in given_parameters:
        given_asignatures = given_parameters["asignatures"]
    else:
        given_asignatures = None

    asignatures, exposures_mat = initialize_base(
        adata,
        n_signatures,
        method,
        given_asignatures,
        **kwargs,
    )
    adata.obsm["exposures"] = exposures_mat
    return asignatures


def check_given_scalings_corrnmf(
    given_scalings: np.ndarray, n_scalings_expected: int, name: str
) -> None:
    type_checker(name, given_scalings, np.ndarray)
    shape_checker(name, given_scalings, (n_scalings_expected,))


def check_given_embeddings_corrnmf(
    given_embeddings: np.ndarray,
    n_embeddings_expected: int,
    dim_embeddings_expected: int,
    name: str,
) -> None:
    type_checker(name, given_embeddings, np.ndarray)
    shape_checker(
        name, given_embeddings, (n_embeddings_expected, dim_embeddings_expected)
    )


def check_given_parameters_corrnmf(
    adata: ad.AnnData,
    n_signatures: int,
    dim_embeddings: int,
    given_parameters: dict[str, Any],
) -> None:
    dict_checker("given_parameters", given_parameters, GIVEN_PARAMETERS_CORRNMF)

    if "asignatures" in given_parameters:
        check_given_asignatures(given_parameters["asignatures"], adata, n_signatures)

    if "signature_scalings" in given_parameters:
        check_given_scalings_corrnmf(
            given_parameters["signature_scalings"],
            n_signatures,
            "given_signature_scalings",
        )
    if "sample_scalings" in given_parameters:
        check_given_scalings_corrnmf(
            given_parameters["sample_scalings"], adata.n_obs, "given_sample_scalings"
        )
    if "signature_embeddings" in given_parameters:
        check_given_embeddings_corrnmf(
            given_parameters["signature_embeddings"],
            n_signatures,
            dim_embeddings,
            "given_signature_embeddings",
        )
    if "sample_embeddings" in given_parameters:
        check_given_embeddings_corrnmf(
            given_parameters["sample_embeddings"],
            adata.n_obs,
            dim_embeddings,
            "given_sample_embeddings",
        )
    if "variance" in given_parameters:
        given_variance = given_parameters["variance"]
        type_checker("given_variance", given_variance, [float, int])
        if given_variance <= 0.0:
            raise ValueError("The variance has to be a positive real number.")


def initialize_corrnmf(
    adata: ad.AnnData,
    n_signatures: int,
    dim_embeddings: int,
    method: _Init_methods = "nndsvd",
    given_parameters: dict[str, Any] | None = None,
    initialize_sample_embeddings: bool = True,
    **kwargs,
) -> tuple[ad.AnnData, float]:
    if method == "custom":
        raise ValueError(
            "Custom parameter initializations are currently not supported "
            "for (multimodal) correlated NMF."
        )

    given_parameters = {} if given_parameters is None else given_parameters.copy()
    check_given_parameters_corrnmf(
        adata, n_signatures, dim_embeddings, given_parameters
    )

    if "asignatures" in given_parameters:
        given_asignatures = given_parameters["asignatures"]
    else:
        given_asignatures = None

    asignatures, _ = initialize_base(
        adata,
        n_signatures,
        method,
        given_asignatures,
        **kwargs,
    )

    if "signature_scalings" in given_parameters:
        asignatures.obs["scalings"] = given_parameters["signature_scalings"]
    else:
        asignatures.obs["scalings"] = np.zeros(n_signatures)

    if "sample_scalings" in given_parameters:
        adata.obs["scalings"] = given_parameters["sample_scalings"]
    else:
        adata.obs["scalings"] = np.zeros(adata.n_obs)

    if "signature_embeddings" in given_parameters:
        asignatures.obsm["embeddings"] = given_parameters["signature_embeddings"]
    else:
        asignatures.obsm["embeddings"] = np.random.multivariate_normal(
            np.zeros(dim_embeddings), np.identity(dim_embeddings), size=n_signatures
        )

    if initialize_sample_embeddings:
        if "sample_embeddings" in given_parameters:
            adata.obsm["embeddings"] = given_parameters["sample_embeddings"]
        else:
            adata.obsm["embeddings"] = np.random.multivariate_normal(
                np.zeros(dim_embeddings),
                np.identity(dim_embeddings),
                size=adata.n_obs,
            )

    if "variance" in given_parameters:
        variance = float(given_parameters["variance"])
    else:
        variance = 1.0

    return asignatures, variance


def check_given_parameters_mmcorrnmf(
    mdata: md.MuData,
    ns_signatures: list[int],
    dim_embeddings: int,
    given_parameters: dict[str, Any],
) -> None:
    valid_keys = list(mdata.mod.keys()) + ["sample_embeddings", "variance"]
    dict_checker("given_parameters", given_parameters, valid_keys)

    for (mod_name, adata), n_signatures in zip(mdata.mod.items(), ns_signatures):
        if mod_name in given_parameters:
            given_parameters_mod = given_parameters[mod_name]
        else:
            given_parameters_mod = {}

        check_given_parameters_corrnmf(
            adata, n_signatures, dim_embeddings, given_parameters_mod
        )
        if "sample_embeddings" in given_parameters_mod:
            raise KeyError(
                "The sample embeddings are shared across modalities in multimodal "
                "correlated NMF. They cannot be provided as given parameters on the "
                "modality level."
            )
        if "variance" in given_parameters_mod:
            raise KeyError(
                "The variance parameters of multimodal correlated NMF is shared "
                "across modalies. It cannot be provided as a given parameter on the "
                "modality level."
            )


def initialize_mmcorrnmf(
    mdata: md.MuData,
    ns_signatures: list[int],
    dim_embeddings: int,
    method: _Init_methods = "nndsvd",
    given_parameters: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[dict[str, ad.AnnData], float]:
    """
    Initialize the MuData object and the annotated signatures of all modalities.

    Multimodal correlated NMF shares the sample embeddings across modalities.
    There are sample biases, signatures, and signature scalings & embeddings for
    each individual modality.
    """
    given_parameters = {} if given_parameters is None else given_parameters.copy()
    check_given_parameters_mmcorrnmf(
        mdata, ns_signatures, dim_embeddings, given_parameters
    )
    asignatures = {}

    for (mod_name, adata), n_signatures in zip(mdata.mod.items(), ns_signatures):
        if mod_name in given_parameters:
            given_parameters_mod = given_parameters[mod_name]
        else:
            given_parameters_mod = {}

        asigs, _ = initialize_corrnmf(
            adata,
            n_signatures,
            dim_embeddings,
            method,
            given_parameters_mod,
            initialize_sample_embeddings=False,
            **kwargs,
        )
        if "asignatures" in given_parameters_mod:
            n_given_sigs = given_parameters_mod["asignatures"].n_obs
        else:
            n_given_sigs = 0

        sig_names_new = [
            f"{mod_name} " + sig_name for sig_name in asigs.obs_names[n_given_sigs:]
        ]
        asigs.obs_names = list(asigs.obs_names[:n_given_sigs]) + sig_names_new
        asignatures[mod_name] = asigs

    if "sample_embeddings" in given_parameters:
        mdata.obsm["embeddings"] = given_parameters["sample_embeddings"]
    else:
        mdata.obsm["embeddings"] = np.random.multivariate_normal(
            np.zeros(dim_embeddings),
            np.identity(dim_embeddings),
            size=mdata.n_obs,
        )

    if "variance" in given_parameters:
        variance = float(given_parameters["variance"])
    else:
        variance = 1.0

    return asignatures, variance
