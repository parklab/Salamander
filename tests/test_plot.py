import numpy as np
import pandas as pd
import pytest

from salamander import plot


@pytest.fixture
def exposures():
    mat = np.array([[1, 2, 3, 4], [1, 3, 2, 4]])
    exposures = pd.DataFrame(mat, columns=["a", "b", "c", "d"])
    return exposures


def test_get_sample_order_normalized(exposures):
    sample_order = plot._get_sample_order(exposures, normalize=True)

    # A next to D
    position_a = np.where(sample_order == "a")[0][0]
    position_d = np.where(sample_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 1

    # B as far away from C as possible
    position_b = np.where(sample_order == "b")[0][0]
    position_c = np.where(sample_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 3


def test_get_sample_order_unnormalized(exposures):
    sample_order = plot._get_sample_order(exposures, normalize=False)

    # A as fara away from D as possible
    position_a = np.where(sample_order == "a")[0][0]
    position_d = np.where(sample_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 3

    # B next to C
    position_b = np.where(sample_order == "b")[0][0]
    position_c = np.where(sample_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 1


def test_reorder_exposures(exposures):
    # reordering is based on the relative exposures
    exposures_reordered = plot._reorder_exposures(exposures)
    sample_order = exposures_reordered.columns.to_numpy()

    # A next to D
    position_a = np.where(sample_order == "a")[0][0]
    position_d = np.where(sample_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 1

    # B as far away from C as possible
    position_b = np.where(sample_order == "b")[0][0]
    position_c = np.where(sample_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 3


def test_reorder_custom(exposures):
    custom_sample_order = ["b", "a", "c", "d"]
    exposures_reordered = plot._reorder_exposures(
        exposures, sample_order=custom_sample_order
    )
    assert np.array_equal(exposures_reordered.columns, custom_sample_order)
