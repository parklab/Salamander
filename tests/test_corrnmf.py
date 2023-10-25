import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import corrnmf_det

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/corrnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA}/counts.csv", index_col=0)


@pytest.fixture(params=[1, 2])
def n_signatures(request):
    return request.param


@pytest.fixture
def dim_embeddings(n_signatures):
    return n_signatures


@pytest.fixture
def path_suffix(n_signatures, dim_embeddings):
    return f"nsigs{n_signatures}_dim{dim_embeddings}.npy"


@pytest.fixture
def W_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/W_init_{path_suffix}")


@pytest.fixture
def alpha_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/alpha_init_{path_suffix}")


@pytest.fixture
def beta_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/beta_init_{path_suffix}")


@pytest.fixture
def _p(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/p_{path_suffix}")


@pytest.fixture
def _aux(counts, _p):
    return np.einsum("vd,vkd->kd", counts.values, _p)


@pytest.fixture
def L_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/L_init_{path_suffix}")


@pytest.fixture
def U_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/U_init_{path_suffix}")


@pytest.fixture
def sigma_sq_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sigma_sq_init_{path_suffix}")


@pytest.fixture
def model_init(counts, W_init, alpha_init, beta_init, L_init, U_init, sigma_sq_init):
    n_signatures, dim_embeddings = L_init.shape
    model = corrnmf_det.CorrNMFDet(
        n_signatures=n_signatures, dim_embeddings=dim_embeddings
    )
    model.X = counts.values
    model.W = W_init
    model.alpha = alpha_init
    model.beta = beta_init
    model.L = L_init
    model.U = U_init
    model.sigma_sq = sigma_sq_init
    model.mutation_types = counts.index
    model.signature_names = ["_" for _ in range(n_signatures)]
    model.sample_names = counts.columns
    model.n_samples = len(counts.columns)
    model.given_signature_embeddings = None
    return model


@pytest.fixture
def objective_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/objective_init_{path_suffix}")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


@pytest.fixture
def surrogate_objective_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/surrogate_objective_init_{path_suffix}")


def test_surrogate_objective_function(model_init, surrogate_objective_init):
    assert np.allclose(
        model_init._surrogate_objective_function(), surrogate_objective_init
    )


@pytest.fixture
def W_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/W_updated_{path_suffix}")


@pytest.fixture
def alpha_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/alpha_updated_{path_suffix}")


@pytest.fixture
def beta_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/beta_updated_{path_suffix}")


@pytest.fixture
def L_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/L_updated_{path_suffix}")


@pytest.fixture
def U_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/U_updated_{path_suffix}")


@pytest.fixture
def sigma_sq_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sigma_sq_updated_{path_suffix}")


class TestUpdatesCorrNMFDet:
    def test_update_W(self, model_init, W_updated):
        model_init._update_W()
        assert np.allclose(model_init.W, W_updated)

    def test_update_alpha(self, model_init, alpha_updated):
        model_init._update_alpha()
        assert np.allclose(model_init.alpha, alpha_updated)

    def test_update_beta(self, model_init, _p, beta_updated):
        model_init._update_beta(_p)
        assert np.allclose(model_init.beta, beta_updated)

    def test_p(self, model_init, _p):
        p_computed = model_init._update_p()
        assert np.allclose(p_computed, _p)

    def test_update_L(self, model_init, _aux, L_updated):
        model_init._update_L(_aux)
        assert np.allclose(model_init.L, L_updated)

    def test_update_U(self, model_init, _aux, U_updated):
        model_init._update_U(_aux)
        assert np.allclose(model_init.U, U_updated)

    def test_update_sigma_sq(self, model_init, sigma_sq_updated):
        model_init._update_sigma_sq()
        assert np.allclose(model_init.sigma_sq, sigma_sq_updated)

    def test_given_signatures(self, n_signatures, counts):
        for n_given_signatures in range(1, n_signatures + 1):
            given_signatures = counts.iloc[:, :n_given_signatures].astype(float).copy()
            given_signatures /= given_signatures.sum(axis=0)
            model = corrnmf_det.CorrNMFDet(
                n_signatures=n_signatures,
                dim_embeddings=n_signatures,
                min_iterations=3,
                max_iterations=3,
            )
            model.fit(counts, given_signatures=given_signatures)
            assert np.allclose(
                given_signatures, model.signatures.iloc[:, :n_given_signatures]
            )

    def test_given_signature_embeddings(self, n_signatures, counts):
        for dim_embeddings in range(1, n_signatures + 1):
            given_signature_embeddings = np.random.uniform(
                size=(dim_embeddings, n_signatures)
            )
            model = corrnmf_det.CorrNMFDet(
                n_signatures=n_signatures,
                dim_embeddings=dim_embeddings,
                min_iterations=3,
                max_iterations=3,
            )
            model.fit(counts, given_signature_embeddings=given_signature_embeddings)
            assert np.allclose(given_signature_embeddings, model.L)

    def test_given_sample_embeddings(self, n_signatures, counts):
        n_samples = len(counts.columns)

        for dim_embeddings in range(1, n_signatures + 1):
            given_sample_embeddings = np.random.uniform(
                size=(dim_embeddings, n_samples)
            )
            model = corrnmf_det.CorrNMFDet(
                n_signatures=n_signatures,
                dim_embeddings=dim_embeddings,
                min_iterations=3,
                max_iterations=3,
            )
            model.fit(counts, given_sample_embeddings=given_sample_embeddings)
            assert np.allclose(given_sample_embeddings, model.U)
