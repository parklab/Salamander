import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import mvnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/mvnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA}/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def W_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/W_init_nsigs{n_signatures}.npy")


@pytest.fixture
def H_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/H_init_nsigs{n_signatures}.npy")


@pytest.fixture
def model_init(counts, W_init, H_init):
    n_signatures = W_init.shape[1]
    model = mvnmf.MvNMF(n_signatures=n_signatures, lam=1.0, delta=1.0)
    model.X = counts.values
    model.W = W_init
    model.H = H_init
    model._gamma = 1.0
    return model


@pytest.fixture
def objective_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/objective_init_nsigs{n_signatures}.npy")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


class TestUpdatesMVNMF:
    @pytest.fixture
    def W_updated(self, n_signatures):
        return np.load(f"{PATH_TEST_DATA}/W_updated_nsigs{n_signatures}.npy")

    @pytest.fixture
    def H_updated(self, n_signatures):
        return np.load(f"{PATH_TEST_DATA}/H_updated_nsigs{n_signatures}.npy")

    def test_update_W(self, model_init, W_updated):
        model_init._update_W()
        assert np.allclose(model_init.W, W_updated)

    def test_update_H(self, model_init, H_updated):
        model_init._update_H()
        assert np.allclose(model_init.H, H_updated)

    def test_given_signatures(self, n_signatures, counts):
        for n_given_signatures in range(1, n_signatures + 1):
            given_signatures = counts.iloc[:, :n_given_signatures].astype(float).copy()
            given_signatures /= given_signatures.sum(axis=0)
            model = mvnmf.MvNMF(
                n_signatures=n_signatures, min_iterations=3, max_iterations=3
            )
            model.fit(counts, given_signatures=given_signatures)
            assert np.allclose(
                given_signatures, model.signatures.iloc[:, :n_given_signatures]
            )
