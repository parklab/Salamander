import pickle

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from salamander.nmf_framework import klnmf
from salamander.nmf_framework.standard_nmf import _DEFAULT_GIVEN_PARAMETERS

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/klnmf"


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
    model = klnmf.KLNMF(n_signatures=n_signatures)
    model.adata = adata
    model.asignatures = asignatures_init
    model.adata.obsm["exposures"] = H_init.T
    return model


@pytest.fixture
def objective_init(n_signatures):
    return np.load(f"{PATH_TEST_DATA}/objective_init_nsigs{n_signatures}.npy")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


class TestUpdatesKLNMF:
    @pytest.fixture
    def WH_updated(self, n_signatures):
        with open(
            f"{PATH_TEST_DATA}/WH_updated_joint_nsigs{n_signatures}.pkl", "rb"
        ) as f:
            WH_updated = pickle.load(f)
        return WH_updated

    def test_update_parameters(self, model_init, WH_updated):
        model_init._update_parameters(given_parameters=_DEFAULT_GIVEN_PARAMETERS)
        W_updated, H_updated = WH_updated
        assert np.allclose(model_init.asignatures.X, W_updated.T)
        assert np.allclose(model_init.adata.obsm["exposures"], H_updated.T)

    def test_given_signatures(self, n_signatures, adata):
        for n_given_signatures in range(1, n_signatures + 1):
            given_asignatures = adata[:n_given_signatures, :].copy()
            given_asignatures.X = given_asignatures.X / np.sum(
                given_asignatures.X, axis=1, keepdims=True
            )
            model = klnmf.KLNMF(
                n_signatures=n_signatures,
                min_iterations=3,
                max_iterations=3,
            )
            model.fit(adata, given_parameters={"asignatures": given_asignatures})
            assert np.allclose(
                given_asignatures.X, model.asignatures.X[:n_given_signatures, :]
            )
