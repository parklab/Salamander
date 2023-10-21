import pickle

import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import klnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/klnmf"


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
    model = klnmf.KLNMF(n_signatures=n_signatures)
    model.X = counts.values
    model.W = W_init
    model.H = H_init
    return model


@pytest.fixture
def objective_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/objective_init_nsigs{n_signatures}.npy")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


@pytest.mark.parametrize("update_method", ["mu-standard", "mu-joint"])
class TestUpdatesKLNMF:
    @pytest.fixture
    def WH_updated(self, n_signatures, update_method):
        with open(
            f"{PATH_TEST_DATA}/WH_updated_{update_method}_nsigs{n_signatures}.pkl", "rb"
        ) as f:
            WH_updated = pickle.load(f)
        return WH_updated

    def test_update_WH(self, model_init, update_method, WH_updated):
        model_init.update_method = update_method
        model_init._update_WH()
        W_updated, H_updated = WH_updated
        assert np.allclose(model_init.W, W_updated)
        assert np.allclose(model_init.H, H_updated)

    def test_given_signatures(self, n_signatures, update_method, counts):
        for n_given_signatures in range(1, n_signatures + 1):
            given_signatures = counts.iloc[:, :n_given_signatures].astype(float).copy()
            given_signatures /= given_signatures.sum(axis=0)
            model = klnmf.KLNMF(
                n_signatures=n_signatures,
                update_method=update_method,
                min_iterations=3,
                max_iterations=3,
            )
            model.fit(counts, given_signatures=given_signatures)
            assert np.allclose(
                given_signatures, model.signatures.iloc[:, :n_given_signatures]
            )
