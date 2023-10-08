import numpy as np
import pandas as pd
import pytest

from salamander.nmf_framework import corrnmf_det, multimodal_corrnmf

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/multimodal_corrnmf"
N_MODALITIES = 2
NS_SIGNATURES = [2, 3]
DIM_EMBEDDINGS = 2


@pytest.fixture
def U_init():
    """
    Initial joint sample embeddings.
    """
    return np.load(f"{PATH_TEST_DATA}/U_init.npy")


@pytest.fixture
def sigma_sq_init():
    """
    Initial joint variance.
    """
    return np.load(f"{PATH_TEST_DATA}/sigma_sq_init.npy")


@pytest.fixture
def counts():
    """
    Input count data.
    """
    return [
        pd.read_csv(f"{PATH_TEST_DATA}/model{n}_counts.csv", index_col=0)
        for n in range(N_MODALITIES)
    ]


@pytest.fixture
def Ws_init():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_W_init.npy") for n in range(N_MODALITIES)
    ]


@pytest.fixture
def alphas_init():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_alpha_init.npy")
        for n in range(N_MODALITIES)
    ]


@pytest.fixture
def Ls_init():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_L_init.npy") for n in range(N_MODALITIES)
    ]


@pytest.fixture
def multi_model_init(counts, Ws_init, alphas_init, Ls_init, U_init, sigma_sq_init):
    models = []

    for n, n_signatures in enumerate(NS_SIGNATURES):
        model = corrnmf_det.CorrNMFDet(
            n_signatures=n_signatures, dim_embeddings=DIM_EMBEDDINGS
        )
        model.X = counts[n].values
        model.W = Ws_init[n]
        model.alpha = alphas_init[n]
        model.L = Ls_init[n]
        model.U = U_init
        model.sigma_sq = sigma_sq_init
        model.mutation_types = counts[n].index
        model.signature_names = [f"Sig {k}" for k in range(n_signatures)]
        model.sample_names = counts[n].columns
        models.append(model)

    multi_model = multimodal_corrnmf.MultimodalCorrNMF(
        n_modalities=N_MODALITIES,
        ns_signatures=NS_SIGNATURES,
        dim_embeddings=DIM_EMBEDDINGS,
    )
    multi_model.models = models
    multi_model.n_samples = len(counts[0].columns)
    return multi_model


@pytest.fixture
def _ps():
    return [np.load(f"{PATH_TEST_DATA}/model{n}_p.npy") for n in range(N_MODALITIES)]


@pytest.fixture
def _auxs(counts, _ps):
    return [np.einsum("vd,vkd->kd", data.values, p) for data, p in zip(counts, _ps)]


@pytest.fixture
def objective_init():
    return np.load(f"{PATH_TEST_DATA}/objective_init.npy")


@pytest.fixture
def surrogate_objective_init():
    return np.load(f"{PATH_TEST_DATA}/surrogate_objective_init.npy")


@pytest.fixture
def Ws_updated_Lee():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_W_Lee_updated.npy")
        for n in range(N_MODALITIES)
    ]


@pytest.fixture
def Ws_updated_surrogate():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_W_surrogate_updated.npy")
        for n in range(N_MODALITIES)
    ]


@pytest.fixture
def alphas_updated():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_alpha_updated.npy")
        for n in range(N_MODALITIES)
    ]


@pytest.fixture
def Ls_updated():
    return [
        np.load(f"{PATH_TEST_DATA}/model{n}_L_updated.npy") for n in range(N_MODALITIES)
    ]


@pytest.fixture
def U_updated():
    return np.load(f"{PATH_TEST_DATA}/U_updated.npy")


@pytest.fixture
def sigma_sq_updated():
    return np.load(f"{PATH_TEST_DATA}/sigma_sq_updated.npy")


class TestMultimodalCorrNMFDet:
    def test_objective_function(self, multi_model_init, objective_init):
        assert np.allclose(multi_model_init.objective_function(), objective_init)

    def test_surrogate_objective_function(
        self, multi_model_init, _ps, surrogate_objective_init
    ):
        assert np.allclose(
            multi_model_init._surrogate_objective_function(_ps),
            surrogate_objective_init,
        )

    def test_update_W_Lee(self, multi_model_init, _ps, Ws_updated_Lee):
        for model in multi_model_init.models:
            model.update_W = "1999-Lee"

        given_signatures = [None for _ in range(N_MODALITIES)]
        multi_model_init._update_Ws(_ps, given_signatures)

        for model, W_updated_Lee in zip(multi_model_init.models, Ws_updated_Lee):
            assert np.allclose(model.W, W_updated_Lee)

    def test_update_W_surrogate(self, multi_model_init, _ps, Ws_updated_surrogate):
        for model in multi_model_init.models:
            model.update_W = "surrogate"

        given_signatures = [None for _ in range(N_MODALITIES)]
        multi_model_init._update_Ws(_ps, given_signatures)

        for model, W_updated_surrogate in zip(
            multi_model_init.models, Ws_updated_surrogate
        ):
            assert np.allclose(model.W, W_updated_surrogate)

    def test_update_alpha(self, multi_model_init, alphas_updated):
        multi_model_init._update_alphas()

        for model, alpha_updated in zip(multi_model_init.models, alphas_updated):
            assert np.allclose(model.alpha, alpha_updated)

    def test_p(self, multi_model_init, _ps):
        ps_computed = multi_model_init._update_ps()

        for p1, p2 in zip(ps_computed, _ps):
            assert np.allclose(p1, p2)

    def test_update_L(self, multi_model_init, _auxs, U_init, Ls_updated):
        outer_prods_U = np.einsum("mD,nD->Dmn", U_init, U_init)
        given_signature_embeddings = [None for _ in range(N_MODALITIES)]
        multi_model_init._update_Ls(_auxs, outer_prods_U, given_signature_embeddings)

        for model, L_updated in zip(multi_model_init.models, Ls_updated):
            assert np.allclose(model.L, L_updated)

    def test_update_U(self, multi_model_init, _auxs, U_updated):
        multi_model_init._update_U(_auxs)

        for model in multi_model_init.models:
            assert np.allclose(model.U, U_updated)

    def test_update_sigma_sq(self, multi_model_init, sigma_sq_updated):
        multi_model_init._update_sigma_sq()

        for model in multi_model_init.models:
            assert np.allclose(model.sigma_sq, sigma_sq_updated)


@pytest.mark.parametrize("ns_signatures", [[1, 2], [2, 2]])
def test_given_signatures(counts, ns_signatures):
    given_signatures0 = counts[0].iloc[:, : ns_signatures[0]].astype(float).copy()
    given_signatures0 /= given_signatures0.sum(axis=0)
    given_signatures = [given_signatures0, None]
    multi_model = multimodal_corrnmf.MultimodalCorrNMF(
        n_modalities=2,
        ns_signatures=ns_signatures,
        dim_embeddings=2,
        min_iterations=3,
        max_iterations=3,
    )
    multi_model.fit(counts, given_signatures=given_signatures)
    assert np.allclose(given_signatures0, multi_model.models[0].W)
    assert not np.allclose(given_signatures0, multi_model.models[1].W)


@pytest.mark.parametrize(
    "ns_signatures,dim_embeddings", [([1, 2], 1), ([2, 2], 1), ([2, 2], 2)]
)
def test_given_signature_embeddings(counts, ns_signatures, dim_embeddings):
    given_signature_embeddings0 = np.random.uniform(
        size=(dim_embeddings, ns_signatures[0])
    )
    given_signature_embeddings = [given_signature_embeddings0, None]
    multi_model = multimodal_corrnmf.MultimodalCorrNMF(
        n_modalities=2,
        ns_signatures=ns_signatures,
        dim_embeddings=dim_embeddings,
        min_iterations=3,
        max_iterations=3,
    )
    multi_model.fit(counts, given_signature_embeddings=given_signature_embeddings)
    assert np.allclose(given_signature_embeddings0, multi_model.models[0].L)
    assert not np.allclose(given_signature_embeddings0, multi_model.models[1].L)


@pytest.mark.parametrize(
    "ns_signatures,dim_embeddings", [([1, 2], 1), ([2, 2], 1), ([2, 2], 2)]
)
def test_given_sample_embeddings(counts, ns_signatures, dim_embeddings):
    n_samples = len(counts[0].columns)
    given_sample_embeddings = np.random.uniform(size=(dim_embeddings, n_samples))
    multi_model = multimodal_corrnmf.MultimodalCorrNMF(
        n_modalities=2,
        ns_signatures=ns_signatures,
        dim_embeddings=dim_embeddings,
        min_iterations=3,
        max_iterations=3,
    )
    multi_model.fit(counts, given_sample_embeddings=given_sample_embeddings)

    for model in multi_model.models:
        assert np.allclose(given_sample_embeddings, model.U)
