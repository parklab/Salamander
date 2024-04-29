import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from salamander.models.mmcorrnmf import MultimodalCorrNMF

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/models/multimodal_corrnmf"
N_MOD = 2
NS_SIGNATURES = [2, 3]
DIM_EMBEDDINGS = 2


@pytest.fixture
def counts():
    """
    Input count data.
    """
    return {
        f"mod{n}": pd.read_csv(f"{PATH_TEST_DATA}/model{n}_counts.csv", index_col=0).T
        for n in range(N_MOD)
    }


@pytest.fixture
def sample_scalings_init():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_sample_scalings_init.npy")
        for n in range(N_MOD)
    }


@pytest.fixture
def sample_embeddings_init():
    """
    Initial joint sample embeddings.
    """
    return np.load(f"{PATH_TEST_DATA}/sample_embeddings_init.npy").T


@pytest.fixture
def mdata(counts, sample_scalings_init, sample_embeddings_init):
    adatas = {mod_name: AnnData(data) for mod_name, data in counts.items()}
    mdata = MuData(adatas)
    mdata.obsm["embeddings"] = sample_embeddings_init

    for mod_name in mdata.mod.keys():
        mdata[mod_name].obs["scalings"] = sample_scalings_init[mod_name]

    return mdata


@pytest.fixture
def signatures_mat_init():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_signatures_mat_init.npy").T
        for n in range(N_MOD)
    }


@pytest.fixture
def signature_scalings_init():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_signature_scalings_init.npy")
        for n in range(N_MOD)
    }


@pytest.fixture
def signature_embeddings_init():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_signature_embeddings_init.npy").T
        for n in range(N_MOD)
    }


@pytest.fixture
def asignatures_init(
    mdata, signatures_mat_init, signature_scalings_init, signature_embeddings_init
):
    asignatures = {}
    for mod_name in mdata.mod:
        asigs = AnnData(signatures_mat_init[mod_name])
        asigs.var_names = mdata[mod_name].var_names
        asigs.obs["scalings"] = signature_scalings_init[mod_name]
        asigs.obsm["embeddings"] = signature_embeddings_init[mod_name]
        asignatures[mod_name] = asigs
    return asignatures


@pytest.fixture
def _ps():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_p.npy") for n in range(N_MOD)
    }


@pytest.fixture
def _auxs(counts, _ps):
    return {
        mod_name: np.einsum("vd,vkd->kd", data.T.values, _ps[mod_name])
        for mod_name, data in counts.items()
    }


@pytest.fixture
def variance_init():
    """
    Initial joint variance.
    """
    return np.load(f"{PATH_TEST_DATA}/variance_init.npy")


@pytest.fixture
def model_init(
    mdata,
    asignatures_init,
    variance_init,
):
    model = MultimodalCorrNMF(
        ns_signatures=NS_SIGNATURES, dim_embeddings=DIM_EMBEDDINGS
    )
    model.mdata = mdata
    model.asignatures = asignatures_init
    model.compute_exposures()
    model.variance = variance_init
    return model


def test_init_signature_names(model_init):
    # one given signature per modality
    given_sig_name = "A"
    given_parameters = {}

    for mod_name, adata in model_init.mdata.mod.items():
        asigs = AnnData(np.zeros((1, adata.n_vars)))
        asigs.obs_names = [given_sig_name]
        asigs.var_names = adata.var_names
        given_parameters[mod_name] = {"asignatures": asigs}

    model_init._initialize(given_parameters)

    for mod_name, asigs in model_init.asignatures.items():
        for k, sig_name in enumerate(asigs.obs_names):
            if k == 0:
                assert sig_name == given_sig_name
            else:
                assert sig_name == f"{mod_name} Sig{k}"


@pytest.fixture
def objective_init():
    return np.load(f"{PATH_TEST_DATA}/objective_init.npy")


def test_objective_function(model_init, objective_init):
    assert np.allclose(model_init.objective_function(), objective_init)


@pytest.fixture
def signatures_mat_updated():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_signatures_mat_updated.npy").T
        for n in range(N_MOD)
    }


@pytest.fixture
def sample_scalings_updated():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_sample_scalings_updated.npy")
        for n in range(N_MOD)
    }


@pytest.fixture
def sample_embeddings_updated():
    return np.load(f"{PATH_TEST_DATA}/sample_embeddings_updated.npy").T


@pytest.fixture
def signature_scalings_updated():
    return {
        f"mod{n}": np.load(f"{PATH_TEST_DATA}/model{n}_signature_scalings_updated.npy")
        for n in range(N_MOD)
    }


@pytest.fixture
def signature_embeddings_updated():
    return {
        f"mod{n}": np.load(
            f"{PATH_TEST_DATA}/model{n}_signature_embeddings_updated.npy"
        ).T
        for n in range(N_MOD)
    }


@pytest.fixture
def variance_updated():
    return np.load(f"{PATH_TEST_DATA}/variance_updated.npy")


class TestUpdatesMultimodalCorrNMF:
    def test_update_signatures(self, model_init, signatures_mat_updated):
        model_init.update_signatures()

        for mod_name, sigs in model_init.asignatures.items():
            assert np.allclose(sigs.X, signatures_mat_updated[mod_name])

    def test_update_sample_scalings(self, model_init, sample_scalings_updated):
        model_init.update_sample_scalings()

        for mod_name, adata in model_init.mdata.mod.items():
            assert np.allclose(adata.obs["scalings"], sample_scalings_updated[mod_name])

    def test_update_signature_scalings(
        self, model_init, _auxs, signature_scalings_updated
    ):
        model_init.update_signature_scalings(_auxs)

        for mod_name, sigs in model_init.asignatures.items():
            assert np.allclose(
                sigs.obs["scalings"], signature_scalings_updated[mod_name]
            )

    def test_compute_aux(self, model_init, _auxs):
        auxs = model_init._compute_auxs()

        for mod_name, aux in auxs.items():
            assert np.allclose(aux, _auxs[mod_name])

    def test_update_signature_embeddings(
        self, model_init, _auxs, signature_embeddings_updated
    ):
        model_init.update_signature_embeddings(_auxs)

        for mod_name, asigs in model_init.asignatures.items():
            assert np.allclose(
                asigs.obsm["embeddings"], signature_embeddings_updated[mod_name]
            )

    def test_update_sample_embeddings(
        self, model_init, _auxs, sample_embeddings_updated
    ):
        model_init.update_sample_embeddings(_auxs)
        assert np.allclose(
            model_init.mdata.obsm["embeddings"], sample_embeddings_updated
        )

    def test_update_variance(self, model_init, variance_updated):
        model_init.update_variance()
        assert np.allclose(model_init.variance, variance_updated)


@pytest.mark.parametrize(
    "ns_signatures,dim_embeddings", [([1, 2], 1), ([2, 2], 1), ([2, 2], 2)]
)
class TestGivenParametersMultimodalCorrNMF:
    @pytest.fixture()
    def model(self, ns_signatures, dim_embeddings):
        model = MultimodalCorrNMF(
            ns_signatures=ns_signatures,
            dim_embeddings=dim_embeddings,
            max_iterations=3,
        )
        return model

    def test_given_asignatures(self, model, mdata):
        mod_name0, mod_name1 = mdata.mod.keys()
        n_sigs0 = model.ns_signatures[0]

        for n_given_sigs in range(1, n_sigs0 + 1):
            given_asigs0 = mdata.mod[mod_name0][:n_given_sigs, :].copy()
            given_asigs0.X = given_asigs0.X.astype(float)
            given_asigs0.X /= np.sum(given_asigs0.X, axis=1, keepdims=True)
            given_parameters = {mod_name0: {"asignatures": given_asigs0}}
            model.fit(mdata, given_parameters=given_parameters)
            assert np.allclose(
                given_asigs0.X, model.asignatures[mod_name0].X[:n_given_sigs, :]
            )
            assert not np.allclose(
                given_asigs0.X,
                model.asignatures[mod_name1].X[:n_given_sigs, :],
            )
            # check if other mod0 signatures are updated
            if n_given_sigs < n_sigs0:
                sigs0_other = model.asignatures[mod_name0].X[n_given_sigs:, :].copy()
                model._update_parameters(given_parameters)
                assert not np.allclose(
                    sigs0_other, model.asignatures[mod_name0].X[n_given_sigs:, :]
                )

    def test_given_signature_scalings(self, model, mdata):
        mod_name0, mod_name1 = mdata.mod.keys()
        n_sigs0 = model.ns_signatures[0]
        given_sig_scalings0 = np.random.uniform(size=n_sigs0)
        given_parameters = {mod_name0: {"signature_scalings": given_sig_scalings0}}
        model.fit(mdata, given_parameters=given_parameters)
        assert np.allclose(
            given_sig_scalings0, model.asignatures[mod_name0].obs["scalings"]
        )
        assert not np.allclose(
            given_sig_scalings0, model.asignatures[mod_name1].obs["scalings"][:n_sigs0]
        )

    def test_given_signature_embeddings(self, model, mdata):
        mod_name0, mod_name1 = mdata.mod.keys()
        n_sigs0 = model.ns_signatures[0]
        given_sig_embeddings0 = np.random.uniform(size=(n_sigs0, model.dim_embeddings))
        given_parameters = {mod_name0: {"signature_embeddings": given_sig_embeddings0}}
        model.fit(mdata, given_parameters=given_parameters)
        assert np.allclose(
            given_sig_embeddings0, model.asignatures[mod_name0].obsm["embeddings"]
        )
        assert not np.allclose(
            given_sig_embeddings0,
            model.asignatures[mod_name1].obsm["embeddings"][:n_sigs0, :],
        )

    def test_given_sample_scalings(self, model, mdata):
        mod_name0, mod_name1 = mdata.mod.keys()
        given_sample_scalings0 = np.random.uniform(size=mdata.n_obs)
        given_parameters = {mod_name0: {"sample_scalings": given_sample_scalings0}}
        model.fit(mdata, given_parameters=given_parameters)
        assert np.allclose(
            given_sample_scalings0, model.mdata.mod[mod_name0].obs["scalings"]
        )
        assert not np.allclose(
            given_sample_scalings0, model.mdata.mod[mod_name1].obs["scalings"]
        )

    def test_given_sample_embeddings(self, model, mdata):
        given_sample_embeddings = np.random.uniform(
            size=(mdata.n_obs, model.dim_embeddings)
        )
        given_parameters = {"sample_embeddings": given_sample_embeddings}
        model.fit(mdata, given_parameters=given_parameters)
        assert np.allclose(given_sample_embeddings, model.mdata.obsm["embeddings"])

    def test_given_variance(self, model, mdata):
        given_variance = 3.0
        given_parameters = {"variance": given_variance}
        model.fit(mdata, given_parameters=given_parameters)
        assert np.allclose(given_variance, model.variance)
