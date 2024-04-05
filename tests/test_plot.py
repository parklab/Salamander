import numpy as np
import pandas as pd
import pytest

from salamander import plot


@pytest.fixture
def data():
    counts = np.array([[1, 1], [2, 3], [3, 2], [4, 4]])
    sample_names = ["a", "b", "c", "d"]
    data = pd.DataFrame(counts, index=sample_names)
    return data


def test_get_obs_order_normalized(data):
    obs_order = plot._get_obs_order(data, normalize=True)

    # A next to D
    position_a = np.where(obs_order == "a")[0][0]
    position_d = np.where(obs_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 1

    # B as far away from C as possible
    position_b = np.where(obs_order == "b")[0][0]
    position_c = np.where(obs_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 3


def test_get_obs_order_unnormalized(data):
    obs_order = plot._get_obs_order(data, normalize=False)

    # A as far away from D as possible
    position_a = np.where(obs_order == "a")[0][0]
    position_d = np.where(obs_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 3

    # B next to C
    position_b = np.where(obs_order == "b")[0][0]
    position_c = np.where(obs_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 1


def test_reorder_data(data):
    # reordering is based on the relative values
    data_reordered = plot._reorder_data(data)
    obs_order = data_reordered.index.to_numpy()

    # A next to D
    position_a = np.where(obs_order == "a")[0][0]
    position_d = np.where(obs_order == "d")[0][0]
    assert np.abs(position_a - position_d) == 1

    # B as far away from C as possible
    position_b = np.where(obs_order == "b")[0][0]
    position_c = np.where(obs_order == "c")[0][0]
    assert np.abs(position_b - position_c) == 3


def test_reorder_data_custom(data):
    custom_obs_order = ["b", "a", "c", "d"]
    data_reordered = plot._reorder_data(data, obs_order=custom_obs_order)
    assert np.array_equal(data_reordered.index, custom_obs_order)
