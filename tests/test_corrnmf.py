import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import corrnmf_det

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/corrnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH}/nmf_framework/counts.csv", index_col=0)


@pytest.fixture(params=[(1, 1), (2, 2)])
def model(request):
    param = request.param
    return corrnmf_det.CorrNMFDet(n_signatures=param[0], dim_embeddings=param[1])


@pytest.fixture
def path(model):
    return (
        f"{PATH_TEST_DATA}/"
        f"corrnmf_nsigs{model.n_signatures}_dim{model.dim_embeddings}"
    )


@pytest.fixture
def W_init(path):
    return np.load(f"{path}_W_init.npy")


@pytest.fixture
def alpha_init(path):
    return np.load(f"{path}_alpha_init.npy")


@pytest.fixture
def _p(path):
    return np.load(f"{path}_p.npy")


@pytest.fixture
def _aux(counts, _p):
    return np.einsum("vd,vkd->kd", counts.values, _p)


@pytest.fixture
def L_init(path):
    return np.load(f"{path}_L_init.npy")


@pytest.fixture
def U_init(path):
    return np.load(f"{path}_U_init.npy")


@pytest.fixture
def sigma_sq_init(path):
    return np.load(f"{path}_sigma_sq_init.npy")


@pytest.fixture
def model_init(model, counts, W_init, alpha_init, L_init, U_init, sigma_sq_init):
    model.X = counts.values
    model.W = W_init
    model.alpha = alpha_init
    model.L = L_init
    model.U = U_init
    model.sigma_sq = sigma_sq_init
    model.mutation_types = counts.index
    model.signature_names = ["_" for _ in range(model.n_signatures)]
    model.sample_names = counts.columns
    model.n_samples = len(counts.columns)
    model.given_signature_embeddings = None
    return model


@pytest.fixture
def objective_init(path):
    return np.load(f"{path}_objective_init.npy")


@pytest.fixture
def surrogate_objective_init(path):
    return np.load(f"{path}_surrogate_objective_init.npy")


@pytest.fixture
def W_updated(path):
    return np.load(f"{path}_W_updated.npy")


@pytest.fixture
def alpha_updated(path):
    return np.load(f"{path}_alpha_updated.npy")


@pytest.fixture
def L_updated(path):
    return np.load(f"{path}_L_updated.npy")


@pytest.fixture
def U_updated(path):
    return np.load(f"{path}_U_updated.npy")


@pytest.fixture
def sigma_sq_updated(path):
    return np.load(f"{path}_sigma_sq_updated.npy")


class TestCorrNMFDet:
    def test_objective_function(self, model_init, objective_init):
        assert np.allclose(model_init.objective_function(), objective_init)

    def test_surrogate_objective_function(
        self, model_init, _p, surrogate_objective_init
    ):
        assert np.allclose(
            model_init._surrogate_objective_function(_p), surrogate_objective_init
        )

    def test_update_W_Lee(self, model_init, W_updated):
        model_init._update_W()
        assert np.allclose(model_init.W, W_updated)

    def test_update_alpha(self, model_init, alpha_updated):
        model_init._update_alpha()
        assert np.allclose(model_init.alpha, alpha_updated)

    def test_p(self, model_init, _p):
        p_computed = model_init._update_p()
        assert np.allclose(p_computed, _p)

    def test_update_L(self, model_init, _aux, L_updated):
        model_init._update_L(_aux)
        assert np.allclose(model_init.L, L_updated)

    def test_update_U(self, model_init, _aux, U_updated):
        print("\n\n", "BEFORE UPDATE", model_init.U, "\n\n")
        model_init._update_U(_aux)
        print("UPDATED", model_init.U, "\n\n")
        print("SOLUTION", U_updated, "\n\n")
        print("DIFFERENCE", np.sum(np.abs(model_init.U - U_updated)))
        assert np.allclose(model_init.U, U_updated)

    def test_update_sigma_sq(self, model_init, sigma_sq_updated):
        model_init._update_sigma_sq()
        assert np.allclose(model_init.sigma_sq, sigma_sq_updated)


@pytest.mark.parametrize("n_signatures", [1, 2])
def test_given_signatures(counts, n_signatures):
    given_signatures = counts.iloc[:, :n_signatures].astype(float).copy()
    given_signatures /= given_signatures.sum(axis=0)
    model = corrnmf_det.CorrNMFDet(
        n_signatures=n_signatures,
        dim_embeddings=n_signatures,
        min_iterations=3,
        max_iterations=3,
    )
    model.fit(counts, given_signatures=given_signatures)
    assert np.allclose(given_signatures, model.signatures)


@pytest.mark.parametrize("n_signatures,dim_embeddings", [(1, 1), (2, 1), (2, 2)])
def test_given_signature_embeddings(counts, n_signatures, dim_embeddings):
    given_signature_embeddings = np.random.uniform(size=(dim_embeddings, n_signatures))
    model = corrnmf_det.CorrNMFDet(
        n_signatures=n_signatures,
        dim_embeddings=dim_embeddings,
        min_iterations=3,
        max_iterations=3,
    )
    model.fit(counts, given_signature_embeddings=given_signature_embeddings)
    assert np.allclose(given_signature_embeddings, model.L)


@pytest.mark.parametrize("n_signatures,dim_embeddings", [(1, 1), (2, 1), (2, 2)])
def test_given_sample_embeddings(counts, n_signatures, dim_embeddings):
    n_samples = len(counts.columns)
    given_sample_embeddings = np.random.uniform(size=(dim_embeddings, n_samples))
    model = corrnmf_det.CorrNMFDet(
        n_signatures=n_signatures,
        dim_embeddings=dim_embeddings,
        min_iterations=3,
        max_iterations=3,
    )
    model.fit(counts, given_sample_embeddings=given_sample_embeddings)
    assert np.allclose(given_sample_embeddings, model.U)
