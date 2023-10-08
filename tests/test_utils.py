import numpy as np
import pandas as pd
import pytest

from salamander.utils import kl_divergence, poisson_llh, samplewise_kl_divergence

PATH_TEST_DATA = "tests/test_data"
PATH_TEST_DATA_UTILS = f"{PATH_TEST_DATA}/utils"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA_UTILS}/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def objective_inputs(counts, n_signatures):
    path = f"{PATH_TEST_DATA_UTILS}/objective_input_nsigs{n_signatures}"
    W = np.load(f"{path}_W.npy")
    H = np.load(f"{path}_H.npy")

    return (counts.values, W, H)


@pytest.fixture
def kl_divergence_output(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS}/kl_divergence_nsigs{n_signatures}_result.npy"
    return np.load(path)


def test_kl_divergence(objective_inputs, kl_divergence_output):
    assert np.allclose(kl_divergence(*objective_inputs), kl_divergence_output)


@pytest.fixture
def samplewise_kl_divergence_output(n_signatures):
    path = (
        f"{PATH_TEST_DATA_UTILS}/"
        f"samplewise_kl_divergence_nsigs{n_signatures}_result.npy"
    )
    return np.load(path)


def test_samplewise_kl_divergence(objective_inputs, samplewise_kl_divergence_output):
    assert np.allclose(
        samplewise_kl_divergence(*objective_inputs), samplewise_kl_divergence_output
    )


@pytest.fixture
def poisson_llh_output(n_signatures):
    path = f"{PATH_TEST_DATA_UTILS}/poisson_llh_nsigs{n_signatures}_result.npy"
    return np.load(path)


def test_poisson_llh(objective_inputs, poisson_llh_output):
    assert np.allclose(poisson_llh(*objective_inputs), poisson_llh_output)
