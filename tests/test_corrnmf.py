import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from salamander.nmf_framework import corrnmf_det

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/corrnmf"


@pytest.fixture
def counts():
    return pd.read_csv(f"{PATH_TEST_DATA}/counts.csv", index_col=0).T


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
def sample_scalings_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sample_scalings_init_{path_suffix}")


@pytest.fixture
def sample_embeddings_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sample_embeddings_init_{path_suffix}").T


@pytest.fixture
def adata(counts, sample_scalings_init, sample_embeddings_init):
    adata = AnnData(counts)
    adata.obs["scalings"] = sample_scalings_init
    adata.obsm["embeddings"] = sample_embeddings_init
    return adata


@pytest.fixture
def signatures_mat_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signatures_mat_init_{path_suffix}").T


@pytest.fixture
def signature_scalings_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signature_scalings_init_{path_suffix}")


@pytest.fixture
def signature_embeddings_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signature_embeddings_init_{path_suffix}").T


@pytest.fixture
def asignatures_init(
    adata, signatures_mat_init, signature_scalings_init, signature_embeddings_init
):
    asignatures = AnnData(signatures_mat_init)
    asignatures.var_names = adata.var_names
    asignatures.obs["scalings"] = signature_scalings_init
    asignatures.obsm["embeddings"] = signature_embeddings_init
    return asignatures


@pytest.fixture
def _p(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/p_{path_suffix}")


@pytest.fixture
def _aux(adata, _p):
    return np.einsum("vd,vkd->kd", adata.X.T, _p)


@pytest.fixture
def variance_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/variance_init_{path_suffix}")


@pytest.fixture
def model_init(
    adata,
    asignatures_init,
    variance_init,
):
    n_signatures, dim_embeddings = asignatures_init.obsm["embeddings"].shape
    model = corrnmf_det.CorrNMFDet(
        n_signatures=n_signatures, dim_embeddings=dim_embeddings
    )
    model.adata = adata
    model.asignatures = asignatures_init
    model.compute_exposures()
    model.variance = variance_init
    return model


@pytest.fixture
def objective_init(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/objective_init_{path_suffix}")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


@pytest.fixture
def signatures_mat_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signatures_mat_updated_{path_suffix}").T


@pytest.fixture
def signature_scalings_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signature_scalings_updated_{path_suffix}")


@pytest.fixture
def sample_scalings_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sample_scalings_updated_{path_suffix}")


@pytest.fixture
def signature_embeddings_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/signature_embeddings_updated_{path_suffix}").T


@pytest.fixture
def sample_embeddings_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/sample_embeddings_updated_{path_suffix}").T


@pytest.fixture
def variance_updated(path_suffix):
    return np.load(f"{PATH_TEST_DATA}/variance_updated_{path_suffix}")


class TestUpdatesCorrNMFDet:
    def test_update_signatures(self, model_init, signatures_mat_updated):
        model_init.update_signatures()
        assert np.allclose(model_init.asignatures.X, signatures_mat_updated)

    def test_update_signature_scalings(
        self, model_init, _aux, signature_scalings_updated
    ):
        model_init.update_signature_scalings(_aux)
        assert np.allclose(
            model_init.asignatures.obs["scalings"].values, signature_scalings_updated
        )

    def test_update_sample_scalings(self, model_init, sample_scalings_updated):
        model_init.update_sample_scalings()
        assert np.allclose(
            model_init.adata.obs["scalings"].values, sample_scalings_updated
        )

    def test_update_signature_embeddings(
        self, model_init, _aux, signature_embeddings_updated
    ):
        model_init.update_signature_embeddings(_aux)
        assert np.allclose(
            model_init.asignatures.obsm["embeddings"], signature_embeddings_updated
        )

    def test_update_sample_embeddings(
        self, model_init, _aux, sample_embeddings_updated
    ):
        model_init.update_sample_embeddings(_aux)
        assert np.allclose(
            model_init.adata.obsm["embeddings"], sample_embeddings_updated
        )

    def test_update_variance(self, model_init, variance_updated):
        model_init.update_variance()
        assert np.allclose(model_init.variance, variance_updated)


@pytest.mark.parametrize("n_signatures,dim_embeddings", [(1, 1), (2, 1), (2, 2)])
class TestGivenParametersCorrNMFDet:
    @pytest.fixture
    def model(self, n_signatures, dim_embeddings):
        return corrnmf_det.CorrNMFDet(
            n_signatures=n_signatures,
            dim_embeddings=dim_embeddings,
            min_iterations=3,
            max_iterations=3,
        )

    @pytest.fixture
    def adata(self, counts):
        return AnnData(counts)

    def test_given_signatures(self, model, adata):
        for n_given_signatures in range(1, model.n_signatures + 1):
            given_asignatures = adata[:n_given_signatures, :].copy()
            given_asignatures.X = given_asignatures.X / np.sum(
                given_asignatures.X, axis=1, keepdims=True
            )
            model.fit(adata, given_parameters={"asignatures": given_asignatures})
            assert np.allclose(
                given_asignatures.X, model.asignatures.X[:n_given_signatures, :]
            )

    def test_given_signature_scalings(self, model, adata):
        given_signature_scalings = np.random.uniform(size=model.n_signatures)
        model.fit(
            adata, given_parameters={"signature_scalings": given_signature_scalings}
        )
        assert np.allclose(
            given_signature_scalings, model.asignatures.obs["scalings"].values
        )

    def test_given_sample_scalings(self, model, adata):
        given_sample_scalings = np.random.uniform(size=adata.n_obs)
        model.fit(adata, given_parameters={"sample_scalings": given_sample_scalings})
        assert np.allclose(given_sample_scalings, model.adata.obs["scalings"].values)

    def test_given_signature_embeddings(self, model, adata):
        given_signature_embeddings = np.random.uniform(
            size=(model.n_signatures, model.dim_embeddings)
        )
        model.fit(
            adata, given_parameters={"signature_embeddings": given_signature_embeddings}
        )
        assert np.allclose(
            given_signature_embeddings, model.asignatures.obsm["embeddings"]
        )

    def test_given_sample_embeddings(self, model, adata):
        given_sample_embeddings = np.random.uniform(
            size=(adata.n_obs, model.dim_embeddings)
        )
        model.fit(
            adata, given_parameters={"sample_embeddings": given_sample_embeddings}
        )
        assert np.allclose(given_sample_embeddings, model.adata.obsm["embeddings"])

    def test_given_variance(self, model, adata):
        given_variance = 3
        model.fit(adata, given_parameters={"variance": given_variance})
        assert np.allclose(given_variance, model.variance)
