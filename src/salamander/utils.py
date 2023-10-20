import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

EPSILON = np.finfo(np.float32).eps


def shape_checker(arg_name: str, arg, allowed_shape):
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


def type_checker(arg_name: str, arg, allowed_types):
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


def value_checker(arg_name: str, arg, allowed_values):
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
    if not isinstance(allowed_values, list):
        allowed_values = [allowed_values]

    if arg not in allowed_values:
        raise ValueError(
            f"The value of '{arg_name}' has to be one of {allowed_values}."
        )


@njit
def normalize_WH(W, H):
    normalization_factor = np.sum(W, axis=0)
    return W / normalization_factor, H * normalization_factor[:, None]


def match_to_catalog(signatures: pd.DataFrame, catalog: pd.DataFrame, metric="cosine"):
    """
    Find the best matching signatures in catalog for all signatures.
    """
    cosine_sim = 1 - pairwise_distances(signatures.T, catalog.T, metric=metric)
    matches_indices = [np.argmax(row) for row in cosine_sim]
    matches = catalog.iloc[:, matches_indices]

    return matches


def match_signatures_pair(
    signatures1: pd.DataFrame, signatures2: pd.DataFrame, metric="cosine"
):
    """
    Match a pair of signature catalogs using their pairwise column distances,
    see https://en.wikipedia.org/wiki/Assignment_problem.

    Output:
    ------
    reordered_indices: np.ndarray
        The list of column indices such that reordering signatures2 using this list
        minimizes the sum of the pairwise column distances between
        signatures1 and signatures2.
    """
    if signatures1.shape != signatures2.shape:
        raise ValueError("The signatures must be of the same shape.")

    pdist = pairwise_distances(signatures1.T, signatures2.T, metric=metric)
    reordered_indices = linear_sum_assignment(pdist)[1]

    return reordered_indices
