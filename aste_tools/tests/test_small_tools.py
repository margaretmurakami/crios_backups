from __future__ import annotations

from datetime import datetime

import numpy as np

from aste_tools.binning import accumulate_ts_mesh, bin_array
from aste_tools.masks import expand_to_3d, masked_area_mean, wet_indices_in_mask
from aste_tools.timing import is_leap, monthly_file_steps, select_month_steps, timestep_to_datetime


def test_bin_array_marks_out_of_range_values_invalid():
    edges = np.array([0, 1, 2])
    values = np.array([-1, 0.2, 1.5, 3.0, np.nan])

    out = bin_array(values, edges)

    np.testing.assert_array_equal(out, np.array([-1, 0, 1, -1, -1]))


def test_accumulate_ts_mesh_sums_valid_cells():
    values = np.array([[1.0, 2.0], [3.0, np.nan]])
    salt_bins = np.array([[0, 0], [1, 1]])
    theta_bins = np.array([[0, 1], [1, -1]])

    mesh = accumulate_ts_mesh(values, salt_bins, theta_bins, n_salinity=2, n_theta=2)

    np.testing.assert_allclose(mesh, np.array([[1.0, 2.0], [0.0, 3.0]]))


def test_mask_helpers():
    field = np.array([[1.0, 2.0], [3.0, np.nan]])
    mask = np.array([[1, 0], [1, 1]])
    area = np.ones((2, 2))

    mean, total_area = masked_area_mean(field, mask, area)

    assert mean == 2.0
    assert total_area == 2.0
    np.testing.assert_array_equal(wet_indices_in_mask(mask, np.array([0, 1, 2, 3])), np.array([0, 2, 3]))
    np.testing.assert_array_equal(expand_to_3d(np.array([1, 2]), np.zeros((2, 2, 2)))[:, 0, 0], np.array([1, 2]))


def test_timing_helpers():
    assert is_leap(2000)
    assert not is_leap(1900)
    assert not is_leap(2001)
    assert timestep_to_datetime(24, delta_t=3600, start=datetime(2000, 1, 1)) == datetime(2000, 1, 2)

    steps = monthly_file_steps(3600, 1979, 1980)
    selected, datetimes = select_month_steps({1979: [2, 3]}, steps, delta_t=3600, start=datetime(1979, 1, 1))
    assert selected.tolist() == ["0000000744", "0000001416"]
    assert [d.month for d in datetimes] == [2, 3]
