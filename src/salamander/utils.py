from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

EPSILON = np.finfo(np.float32).eps


def dict_checker(
    dict_name: str, dictionary: dict[Any, Any], valid_keys: list[Any]
) -> None:
    """
    A helper function to test the keys of a dictionary.

    Input:
    ------
    dict_name: str
        The name of the dictionary

    dictionary: dict[Any, Any]

    valid_keys: list[Any]
        The allowed keys of 'dictionary'
    """
    type_checker(dict_name, dictionary, dict)

    for key in dictionary.keys():
        if key not in valid_keys:
            raise ValueError(f"'{dict_name}' includes keys outside of {valid_keys}.")


def shape_checker(
    arg_name: str, arg: np.ndarray | pd.DataFrame, allowed_shape: tuple[int, ...]
) -> None:
    """
    A helper function to test the shape of a numpy ndarray or pandas dataframe.

    Input:
    ------
    arg_name: str
        The name of the argument
    arg:
        The actual value of the argument
    allowed_shape:
        The expected shape of 'arg'
    """
    type_checker(arg_name, arg, [np.ndarray, pd.DataFrame])

    if arg.shape != allowed_shape:
        raise ValueError(f"The shape of '{arg_name}' has to be {allowed_shape}.")


def type_checker(arg_name: str, arg: Any, allowed_types: type | Iterable[type]) -> None:
    """
    A helper function to test the type of an argument.

    Input:
    ------
    arg_name: str
        The name of the argument
    arg:
        The actual value of the argument
    allowed_types: a type or list of types
        The type or list of types allowed for 'arg'
    """
    if isinstance(allowed_types, type):
        allowed_types = [allowed_types]

    if type(arg) not in allowed_types:
        raise TypeError(f"The type of '{arg_name}' has to be one of {allowed_types}.")


def value_checker(arg_name: str, arg: Any, allowed_values: Iterable[Any]) -> None:
    """
    A helper function to test the value of an argument.

    Input:
    ------
    arg_name: str
        The name of the argument
    arg:
        The actual value of the argument
    allowed_values:
        A value or list of values allowed for 'arg'
    """
    if isinstance(allowed_values, type):
        allowed_values = [allowed_values]

    if arg not in allowed_values:
        raise ValueError(
            f"The value of '{arg_name}' has to be one of {allowed_values}."
        )


def _get_basis_obsm(adata: AnnData | MuData, basis: str) -> np.ndarray:
    """
    Get the multidimensional observation annotations named 'basis'.
    Tries to recover 'X_basis' if 'basis' is not a key of adata.obsm.
    """
    if basis in adata.obsm:
        return adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsm")


def _get_basis_obsp(adata: AnnData | MuData, basis: str) -> np.ndarray:
    """
    Get the pairwise observation annotations named 'basis'.
    Tries to recover 'X_basis' if 'basis' is not a key of adata.obsp.
    """
    if basis in adata.obsp:
        return adata.obsp[basis]
    elif f"X_{basis}" in adata.obsp:
        return adata.obsp[f"X_{basis}"]
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsp")


@njit
def normalize_WH(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalization_factor = np.sum(W, axis=0)
    return W / normalization_factor, H * normalization_factor[:, None]


def match_to_catalog(
    signatures: pd.DataFrame, catalog: pd.DataFrame, metric="cosine"
) -> pd.DataFrame:
    """
    Find the best matching signatures in catalog for all signatures.
    """
    cosine_sim = 1 - pairwise_distances(signatures, catalog, metric=metric)
    matches_indices = [int(np.argmax(row)) for row in cosine_sim]
    matches = catalog.iloc[matches_indices]
    return matches


def match_signatures_pair(
    signatures1: pd.DataFrame, signatures2: pd.DataFrame, metric: str = "cosine"
) -> np.ndarray:
    """
    Match a pair of signature catalogs using their pairwise distances,
    see https://en.wikipedia.org/wiki/Assignment_problem.

    Output:
    ------
    reordered_indices: np.ndarray
        The list of indices such that reordering signatures2 using this list
        minimizes the sum of the pairwise distances between
        signatures1 and signatures2.
    """
    if signatures1.shape != signatures2.shape:
        raise ValueError("The signatures must be of the same shape.")

    pdist = pairwise_distances(signatures1, signatures2, metric=metric)
    reordered_indices = linear_sum_assignment(pdist)[1]
    return reordered_indices
