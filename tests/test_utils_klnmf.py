import numpy as np
import pandas as pd
import pytest

from salamander.models import _utils_klnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/models/utils_klnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA}/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def matrices_input(counts, n_signatures):
    W = np.load(f"{PATH_TEST_DATA}/W_nsigs{n_signatures}.npy")
    H = np.load(f"{PATH_TEST_DATA}/H_nsigs{n_signatures}.npy")
    return (counts.values, W, H)


@pytest.fixture
def weights_kl(counts):
    n_samples = counts.shape[1]
    weights = 2 * np.ones(n_samples)
    return weights


@pytest.fixture
def weights_l_half(counts):
    n_samples = counts.shape[1]
    weights = 2 * np.zeros(n_samples)
    return weights


@pytest.fixture
def kl_divergence_output(n_signatures):
    path = f"{PATH_TEST_DATA}/kl_divergence_nsigs{n_signatures}.npy"
    return np.load(path)


def test_kl_divergence(matrices_input, kl_divergence_output):
    assert np.allclose(
        _utils_klnmf.kl_divergence(*matrices_input), kl_divergence_output
    )


def test_kl_divergence_weights(matrices_input, weights_kl, kl_divergence_output):
    assert np.allclose(
        _utils_klnmf.kl_divergence(*matrices_input, weights_kl),
        2 * kl_divergence_output,
    )


@pytest.fixture
def samplewise_kl_divergence_output(n_signatures):
    path = f"{PATH_TEST_DATA}/" f"samplewise_kl_divergence_nsigs{n_signatures}.npy"
    return np.load(path)


def test_samplewise_kl_divergence(matrices_input, samplewise_kl_divergence_output):
    assert np.allclose(
        _utils_klnmf.samplewise_kl_divergence(*matrices_input),
        samplewise_kl_divergence_output,
    )


def test_samplewise_kl_divergence_weights(
    matrices_input, weights_kl, samplewise_kl_divergence_output
):
    weights_kl[0] = 3
    samplewise_kl_divergence = _utils_klnmf.samplewise_kl_divergence(
        *matrices_input, weights_kl
    )
    assert np.allclose(
        samplewise_kl_divergence[0], 3 * samplewise_kl_divergence_output[0]
    )
    assert np.allclose(
        samplewise_kl_divergence[1:],
        2 * samplewise_kl_divergence_output[1:],
    )


@pytest.fixture
def poisson_llh_output(n_signatures):
    path = f"{PATH_TEST_DATA}/poisson_llh_nsigs{n_signatures}.npy"
    return np.load(path)


def test_poisson_llh(matrices_input, poisson_llh_output):
    assert np.allclose(_utils_klnmf.poisson_llh(*matrices_input), poisson_llh_output)


@pytest.fixture
def W_updated(n_signatures):
    path = f"{PATH_TEST_DATA}/W_updated_standard_nsigs{n_signatures}.npy"
    return np.load(path)


def test_update_W(matrices_input, W_updated):
    W_updated_utils = _utils_klnmf.update_W(*matrices_input)
    assert np.allclose(W_updated_utils, W_updated)


def test_update_W_weights_kl(matrices_input, weights_kl, W_updated):
    # constant loss function weights do not change the updated signatures
    W_updated_utils = _utils_klnmf.update_W(*matrices_input, weights_kl)
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
    path = f"{PATH_TEST_DATA}/H_updated_standard_nsigs{n_signatures}.npy"
    return np.load(path)


def test_update_H(matrices_input, H_updated):
    H_updated_utils = _utils_klnmf.update_H(*matrices_input)
    assert np.allclose(H_updated_utils, H_updated)


def test_update_H_weights_l_half(matrices_input, weights_kl, weights_l_half, H_updated):
    # no l_half penalty -> identical exposure updates,
    # independent of the loss function weights
    H_updated_utils = _utils_klnmf.update_H(*matrices_input, weights_kl, weights_l_half)
    assert np.allclose(H_updated_utils, H_updated)


@pytest.fixture
def WH_updated(n_signatures):
    suffix = f"_updated_joint_nsigs{n_signatures}.npy"
    path_W = f"{PATH_TEST_DATA}/W{suffix}"
    path_H = f"{PATH_TEST_DATA}/H{suffix}"
    return np.load(path_W), np.load(path_H)


def test_update_WH(matrices_input, WH_updated):
    W_updated, H_updated = WH_updated
    W_updated_utils, H_updated_utils = _utils_klnmf.update_WH(*matrices_input)
    assert np.allclose(W_updated_utils, W_updated)
    assert np.allclose(H_updated_utils, H_updated)


def test_update_WH_weights_kl(matrices_input, WH_updated, weights_kl):
    # constant loss function weights do not change the updated matrices
    W_updated, H_updated = WH_updated
    W_updated_utils, H_updated_utils = _utils_klnmf.update_WH(
        *matrices_input, weights_kl
    )
    assert np.allclose(W_updated_utils, W_updated)
    assert np.allclose(H_updated_utils, H_updated)


def test_update_WH_weights_l_half(
    matrices_input, WH_updated, weights_kl, weights_l_half
):
    # no l_half penalty -> identical exposure updates,
    # independent of the loss function weights
    W_updated, H_updated = WH_updated
    W_updated_utils, H_updated_utils = _utils_klnmf.update_WH(
        *matrices_input, weights_kl, weights_l_half
    )
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
