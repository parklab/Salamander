import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import mvnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/mvnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH}/nmf_framework/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def model(request):
    return mvnmf.MvNMF(n_signatures=request.param)


@pytest.fixture
def path(model):
    return f"{PATH_TEST_DATA}/mvnmf_nsigs{model.n_signatures}"


@pytest.fixture
def W_init(path):
    return np.load(f"{path}_W_init.npy")


@pytest.fixture
def H_init(path):
    return np.load(f"{path}_H_init.npy")


@pytest.fixture
def model_init(model, counts, W_init, H_init):
    model.X = counts.values
    model.W = W_init
    model.H = H_init
    model.lam = 1.0
    model.delta = 1.0
    model.gamma = 1.0
    return model


@pytest.fixture
def objective_init(path):
    return np.load(f"{path}_objective_init.npy")


@pytest.fixture
def W_updated(path):
    return np.load(f"{path}_W_updated.npy")


@pytest.fixture
def H_updated(path):
    return np.load(f"{path}_H_updated.npy")


class TestMVNMF:
    def test_objective_function(self, model_init, objective_init):
        assert np.allclose(model_init.objective_function(), objective_init)

    def test_update_W(self, model_init, objective_init, W_updated):
        model_init._update_W(objective_init)
        assert np.allclose(model_init.W, W_updated)

    def test_update_H(self, model_init, H_updated):
        model_init._update_H()
        assert np.allclose(model_init.H, H_updated)


@pytest.mark.parametrize("n_signatures", [1, 2])
def test_given_signatures(counts, n_signatures):
    given_signatures = counts.iloc[:, :n_signatures].astype(float).copy()
    given_signatures /= given_signatures.sum(axis=0)
    model = mvnmf.MvNMF(n_signatures=n_signatures, min_iterations=3, max_iterations=3)
    model.fit(counts, given_signatures=given_signatures)
    assert np.allclose(given_signatures, model.signatures)
