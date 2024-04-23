import numpy as np
import pytest

from salamander.initialization import initialize

PATH = "tests/test_data"
PATH_TEST_DATA = f"{PATH}/nmf_framework/initialization"

METHODS_DET = ["flat"]
METHODS_STOCH = [
    "nndsvd",
    "nndsvda",
    "nndsvdar",
    "random",
    "separableNMF",
]
METHODS_MAIN = METHODS_DET + METHODS_STOCH
METHODS_OTHER = ["custom"]
N_SIGNATURES = 2
SEED = 1


@pytest.fixture
def data_mat():
    return np.load(f"{PATH_TEST_DATA}/data_mat.npy")


@pytest.mark.parametrize("method", METHODS_MAIN)
class TestInitializeMain:
    @pytest.fixture
    def path_suffix(self, method):
        if method in METHODS_DET:
            return f"{method}.npy"
        else:
            return f"{method}_seed{SEED}.npy"

    @pytest.fixture
    def signatures_mat_expected(self, path_suffix):
        return np.load(f"{PATH_TEST_DATA}/signatures_mat_{path_suffix}")

    @pytest.fixture
    def exposures_mat_expected(self, path_suffix):
        return np.load(f"{PATH_TEST_DATA}/exposures_mat_{path_suffix}")

    def test_initialize_mat(
        self, data_mat, method, signatures_mat_expected, exposures_mat_expected
    ):
        kwargs = {"seed": SEED} if method in METHODS_STOCH else {}
        signatures_mat, exposures_mat = initialize.initialize_mat(
            data_mat, N_SIGNATURES, method, **kwargs
        )
        assert np.allclose(signatures_mat, signatures_mat_expected)
        assert np.allclose(exposures_mat, exposures_mat_expected)


def test_initialize_custom(data_mat):
    signatures_mat_custom = np.array([[0.1, 0.2, 0.7], [0.6, 0.1, 0.3]])
    exposures_mat_custom = np.arange(1, 9).reshape((4, 2))  # initialization clips zeros
    signatures_mat, exposures_mat = initialize.initialize_mat(
        data_mat,
        N_SIGNATURES,
        "custom",
        signatures_mat=signatures_mat_custom,
        exposures_mat=exposures_mat_custom,
    )
    assert np.array_equal(signatures_mat_custom, signatures_mat)
    assert np.array_equal(exposures_mat_custom, exposures_mat)
