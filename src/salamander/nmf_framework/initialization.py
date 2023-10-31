"""
Initialization methods for non-negative matrix factorization (NMF)
"""
import numpy as np
from sklearn.decomposition import _nmf as sknmf

from ..utils import normalize_WH, shape_checker, type_checker, value_checker

EPSILON = np.finfo(np.float32).eps
INIT_METHODS = [
    "custom",
    "flat",
    "hierarchical_cluster",
    "nndsvd",
    "nndsvda",
    "nndsvdar",
    "random",
    "separableNMF",
]


def init_custom(
    X: np.ndarray, n_signatures: int, W_custom: np.ndarray, H_custom: np.ndarray
):
    """
    Perform type and shape checks on custom signature and
    exposure matrix initializations.
    """
    type_checker("W_custom", W_custom, np.ndarray)
    type_checker("H_custom", H_custom, np.ndarray)
    n_features, n_samples = X.shape
    shape_checker("W_custom", W_custom, (n_features, n_signatures))
    shape_checker("H_custom", H_custom, (n_signatures, n_samples))
    return W_custom, H_custom


def init_flat(X: np.ndarray, n_signatures: int):
    """
    Initialize the signature and exposure matrices with one float, respectively.
    """
    n_features, n_samples = X.shape
    scaling = np.mean(np.sum(X, axis=0))
    W = np.full((n_features, n_signatures), 1 / n_features)
    H = np.full((n_signatures, n_samples), scaling / n_signatures)
    return W, H


def init_nndsvd(X: np.ndarray, n_signatures: int, init: str, seed=None):
    """
    A wrapper around the non-negative double singular value decomposition (NNDSVD)
    initialization methods "nndsvd", "nndsvda" and "nndsvdar" from scikit-learn.

    Inputs:
    ------
    init: str
        One of "nndsvd", "nndsvda" and "nndsvdar"
    """
    if seed is not None:
        np.random.seed(seed)

    # pylint: disable-next=W0212
    W, H = sknmf._initialize_nmf(X, n_signatures, init=init)

    return W, H


def init_random(X: np.ndarray, n_signatures: int, seed=None):
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

    n_features, n_samples = X.shape
    W = np.random.dirichlet(np.ones(n_features), size=n_signatures).T
    scaling = np.sum(X, axis=0)
    H = scaling * np.random.dirichlet(np.ones(n_signatures), size=n_samples).T
    return W, H


def init_separableNMF(X: np.ndarray, n_signatures: int, seed=None):
    r"""
    This code is following Algorithm 1 from "Fast and Robust Recursive
    Algorithms for Separable Nonnegative Matrix Factorization"
    (Gillis and Vavasis, 2013), with the canonical choice of
    f(x) = \| x \|_2^2 as the strongly convex function f satisfying
    Assumption 2 from the paper.
    """
    signature_indices = np.empty(n_signatures, dtype=int)
    R = X / np.sum(X, axis=0)

    for k in range(n_signatures):
        column_norms = np.sum(R**2, axis=0)
        kstar = np.argmax(column_norms)
        u = R[:, kstar]
        R = (np.identity(X.shape[0]) - np.outer(u, u) / column_norms[kstar]) @ R
        signature_indices[k] = kstar

    W = X[:, signature_indices].astype(float)
    _, H = init_random(X, n_signatures, seed=seed)
    return W, H


def initialize(
    X: np.ndarray,
    n_signatures: int,
    init_method="nndsvd",
    given_signatures=None,
    **kwargs,
):
    """
    Initialize the signature and exposure matrices.

    Parameters
    ----------
    X : np.ndarray
        count matrix

    n_signatures : int
        number of signatures

    init_method : str
        initialization method. One of 'custom', 'flat', 'hierarchical_cluster',
        'nndsvd', 'nndsvda', 'nndsvdar', 'random', 'separableNMF'

    given_signatures : pd.Dataframe, default=None
        At most 'n_signatures' many signatures can be provided to
        overwrite some of the initialized signatures. This does not
        change the initialized exposurse.

    kwargs : dict
        Any keyword arguments to be passed to the initialization method.
        This includes, for example, a possible 'seed' keyword argument
        for all stochastic methods.

    Returns
    -------
    W : np.ndarray
        signature matrix

    H : np.ndarray
        exposure matrix

    signature_names : list
        The signature names. By default, the signatures are named
        'Sigk', where 'k' is one plus the index of the signature.
        If 'given_signatures' are provided, the names are adjusted
        accordingly.
    """
    value_checker("init_method", init_method, INIT_METHODS)

    if init_method == "custom":
        W, H = init_custom(X, n_signatures, **kwargs)

    elif init_method == "flat":
        W, H = init_flat(X, n_signatures)

    elif init_method in ["nndsvd", "nndsvda", "nndsvdar"]:
        W, H = init_nndsvd(X, n_signatures, init=init_method, **kwargs)

    elif init_method == "random":
        W, H = init_random(X, n_signatures, **kwargs)

    else:
        W, H = init_separableNMF(X, n_signatures, **kwargs)

    if given_signatures is not None:
        n_given_signatures = len(given_signatures.columns)
        W[:, :n_given_signatures] = given_signatures.copy().values
        given_signatures_names = given_signatures.columns.to_numpy(dtype=str)
        n_new_signatures = n_signatures - n_given_signatures
        new_signatures_names = np.array([f"Sig{k+1}" for k in range(n_new_signatures)])
        signature_names = np.concatenate([given_signatures_names, new_signatures_names])
    else:
        signature_names = np.array([f"Sig{k+1}" for k in range(n_signatures)])

    W, H = normalize_WH(W, H)
    W, H = W.clip(EPSILON), H.clip(EPSILON)
    return W, H, signature_names
