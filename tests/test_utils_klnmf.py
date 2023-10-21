import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import _utils_klnmf

PATH_TEST_DATA = "tests/test_data"
PATH_TEST_DATA_UTILS_KLNMF = f"{PATH_TEST_DATA}/nmf_framework/utils_klnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA_UTILS_KLNMF}/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def matrices_input(counts, n_signatures):
    W = np.load(f"{PATH_TEST_DATA_UTILS_KLNMF}/W_nsigs{n_signatures}.npy")
    H = np.load(f"{PATH_TEST_DATA_UTILS_KLNMF}/H_nsigs{n_signatures}.npy")

    return (counts.values, W, H)


@pytest.fixture
def kl_divergence_output(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS_KLNMF}/kl_divergence_nsigs{n_signatures}.npy"
    return np.load(path)


def test_kl_divergence(matrices_input, kl_divergence_output):
    assert np.allclose(
        _utils_klnmf.kl_divergence(*matrices_input), kl_divergence_output
    )


@pytest.fixture
def samplewise_kl_divergence_output(n_signatures):
    path = (
        f"{PATH_TEST_DATA_UTILS_KLNMF}/"
        f"samplewise_kl_divergence_nsigs{n_signatures}.npy"
    )
    return np.load(path)


def test_samplewise_kl_divergence(matrices_input, samplewise_kl_divergence_output):
    assert np.allclose(
        _utils_klnmf.samplewise_kl_divergence(*matrices_input),
        samplewise_kl_divergence_output,
    )


@pytest.fixture
def poisson_llh_output(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS_KLNMF}/poisson_llh_nsigs{n_signatures}.npy"
    return np.load(path)


def test_poisson_llh(matrices_input, poisson_llh_output):
    assert np.allclose(_utils_klnmf.poisson_llh(*matrices_input), poisson_llh_output)


@pytest.fixture
def W_updated(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS_KLNMF}/W_updated_mu-standard_nsigs{n_signatures}.npy"
    return np.load(path)


def test_update_W(matrices_input, W_updated):
    W_updated_utils = _utils_klnmf.update_W(*matrices_input)
    assert np.allclose(W_updated_utils, W_updated)


def test_given_signatures_update_W(matrices_input):
    X, W, H = matrices_input
    n_signatures = W.shape[1]

    for n_given_signatures in range(1, n_signatures + 1):
        W_updated = _utils_klnmf.update_W(
            X, W.copy(), H, n_given_signatures=n_given_signatures
        )
        assert np.array_equal(
            W_updated[:, :n_given_signatures], W[:, :n_given_signatures]
        )


@pytest.fixture
def H_updated(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS_KLNMF}/H_updated_mu-standard_nsigs{n_signatures}.npy"
    return np.load(path)


def test_update_H(matrices_input, H_updated):
    H_updated_utils = _utils_klnmf.update_H(*matrices_input)
    assert np.allclose(H_updated_utils, H_updated)


@pytest.fixture
def WH_updated(n_signatures):
    suffix = f"_updated_mu-standard_nsigs{n_signatures}.npy"
    path_W = f"{PATH_TEST_DATA_UTILS_KLNMF}/W{suffix}"
    path_H = f"{PATH_TEST_DATA_UTILS_KLNMF}/H{suffix}"
    return np.load(path_W), np.load(path_H)


def test_update_WH(matrices_input, WH_updated):
    W_updated, H_updated = WH_updated
    W_updated_utils, H_updated_utils = _utils_klnmf.update_WH(*matrices_input)
    assert np.allclose(W_updated_utils, W_updated)
    assert np.allclose(H_updated_utils, H_updated)


def test_given_signatures_update_WH(matrices_input):
    X, W, H = matrices_input
    n_signatures = W.shape[1]

    for n_given_signatures in range(1, n_signatures + 1):
        W_updated, _ = _utils_klnmf.update_WH(
            X, W.copy(), H, n_given_signatures=n_given_signatures
        )
        assert np.array_equal(
            W_updated[:, :n_given_signatures], W[:, :n_given_signatures]
        )
