import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from salamander.nmf_framework import mvnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/mvnmf"


@pytest.fixture
def adata():
    counts = pd.read_csv(f"{PATH_TEST_DATA}/counts.csv", index_col=0)
    adata = AnnData(counts.T)
    return adata


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def W_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/W_init_nsigs{n_signatures}.npy")


@pytest.fixture
def asignatures_init(W_init, adata):
    asignatures = AnnData(W_init.T)
    asignatures.var_names = adata.var_names
    return asignatures


@pytest.fixture
def H_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/H_init_nsigs{n_signatures}.npy")


@pytest.fixture
def model_init(adata, asignatures_init, H_init):
    n_signatures = asignatures_init.n_obs
    model = mvnmf.MvNMF(n_signatures=n_signatures)
    model.adata = adata
    model.asignatures = asignatures_init
    model.adata.obsm["exposures"] = H_init.T
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
        model_init._update_W(n_given_signatures=0)
        assert np.allclose(model_init.asignatures.X, W_updated.T)

    def test_update_H(self, model_init, H_updated):
        model_init._update_H()
        assert np.allclose(model_init.adata.obsm["exposures"], H_updated.T)

    def test_given_signatures(self, n_signatures, adata):
        for n_given_signatures in range(1, n_signatures + 1):
            given_asignatures = adata[:n_given_signatures, :].copy()
            given_asignatures.X = given_asignatures.X / np.sum(
                given_asignatures.X, axis=1, keepdims=True
            )
            model = mvnmf.MvNMF(
                n_signatures=n_signatures, min_iterations=3, max_iterations=3
            )
            model.fit(adata, given_parameters={"asignatures": given_asignatures})
            assert np.allclose(
                given_asignatures.X, model.asignatures.X[:n_given_signatures, :]
            )
