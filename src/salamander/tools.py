from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .utils import _get_basis_obsm, value_checker

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData


def _pca(data: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    from sklearn.decomposition import PCA

    data_reduced_dim = PCA(n_components=n_components, **kwargs).fit_transform(data)
    return data_reduced_dim


def pca(adata: AnnData, basis: str, **kwargs) -> None:
    """
    Compute and store the PCA of the multi-dimensional
    observation annotations named 'basis'.
    """
    data = _get_basis_obsm(adata, basis)
    adata.obsm["X_pca"] = _pca(data, **kwargs)


def _tsne(
    data: np.ndarray, n_components: int = 2, perplexity: float = 30.0, **kwargs
) -> np.ndarray:
    from sklearn.manifold import TSNE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perplexity = min(perplexity, len(data) - 1)
        data_reduced_dim = TSNE(
            n_components=n_components, perplexity=perplexity, **kwargs
        ).fit_transform(data)

    return data_reduced_dim


def tsne(adata: AnnData, basis: str, **kwargs) -> None:
    """
    Compute and store the t-SNE of the multi-dimensional
    observation annotations named 'basis'.
    """
    data = _get_basis_obsm(adata, basis)
    adata.obsm["X_tsne"] = _tsne(data, **kwargs)


def _umap(
    data: np.ndarray,
    n_components: int = 2,
    n_neighbors: float = 15,
    min_dist: float = 0.1,
    **kwargs,
) -> np.ndarray:
    import umap  # pylint: disable=redefined-outer-name

    n_neighbors = min(n_neighbors, len(data) - 1.0)
    data_reduced_dim = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, **kwargs
    ).fit_transform(data)

    return data_reduced_dim


def umap(adata: AnnData, basis: str, **kwargs) -> None:
    """
    Compute and store the UMAP of the multi-dimensional
    observation annotations named 'basis'.
    """
    data = _get_basis_obsm(adata, basis)
    adata.obsm["X_umap"] = _umap(data, **kwargs)


def reduce_dimension_numpy(
    data: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    normalize: bool = False,
    **kwargs,
) -> np.ndarray:
    value_checker("method", method, ["pca", "tsne", "umap"])

    if normalize:
        data /= np.sqrt(np.sum(data**2, axis=1))[:, np.newaxis]

    n_dimensions = data.shape[1]

    if n_dimensions in [1, 2]:
        warnings.warn(
            f"The dimension of the data points is {n_dimensions}. "
            "The dimensionality of the data will not be reduced.",
            UserWarning,
        )
        return data

    if method == "pca":
        data_reduced_dim = _pca(data, n_components=n_components, **kwargs)
    elif method == "tsne":
        data_reduced_dim = _tsne(data, n_components=n_components, **kwargs)
    else:
        data_reduced_dim = _umap(data, n_components=n_components, **kwargs)

    return data_reduced_dim


def reduce_dimension(
    adata: AnnData, basis: str, method="umap", n_components: int = 2, **kwargs
) -> None:
    """
    Compute and store a dimensionality reduction of the multi-dimensional
    observation annotations named 'basis'.
    """
    data = _get_basis_obsm(adata, basis)
    n_dimensions = data.shape[1]

    if n_dimensions in [1, 2]:
        warnings.warn(
            f"The dimension of the observation annotations is {n_dimensions}. "
            "No dimensionality reduction will be applied.",
            UserWarning,
        )
        return

    adata.obsm[f"X_{method}"] = reduce_dimension_numpy(
        data, method=method, n_components=n_components, **kwargs
    )


def reduce_dimension_multiple(
    adatas: Iterable[AnnData | MuData], basis: str, method="umap", **kwargs
) -> None:
    """
    Compute a joint dimensionality reduction of the same multi-dimensional observation
    annotations of multiple AnnData instances.
    """
    data = np.concatenate([_get_basis_obsm(adata, basis) for adata in adatas])
    n_dimensions = data.shape[1]

    if n_dimensions in [1, 2]:
        warnings.warn(
            f"The dimension of the observation annotations is {n_dimensions}. "
            "No dimensionality reduction will be applied.",
            UserWarning,
        )
        return

    data_reduced_dim = reduce_dimension_numpy(data, method=method, **kwargs)
    sum_n_obs = 0

    for adata in adatas:
        n_obs = adata.n_obs
        adata.obsm[f"X_{method}"] = data_reduced_dim[sum_n_obs : sum_n_obs + n_obs, :]
        sum_n_obs += n_obs


def correlation_numpy(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the correlation of the rows of the data.
    """
    return pd.DataFrame(data.T).corr(**kwargs).values


def correlation(adata: AnnData, basis: str, **kwargs) -> None:
    """
    Compute and store the correlation of the multi-dimensional
    observation annotations named 'basis'.
    """
    data = _get_basis_obsm(adata, basis)
    adata.obsp["X_correlation"] = correlation_numpy(data, **kwargs)
