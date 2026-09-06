import numpy as np
import pytest

from experiments.grid_runs_utils import GridAxes, GridAxis, GridConstants
from experiments.scaling_laws_utils import flatten_D_N_grid_data, flatten_N_Lambda_grid_data
from utils.plot import _load_and_aggregate_accuracy_grid


# ---------- seed aggregation ----------

def _save_grid(tmp_path):
    # (seeds=2, nx=2, ny=2) with one all-NaN cell and one partially-NaN cell
    array = np.array([
        [[0.2, np.nan], [0.5, np.nan]],
        [[0.8, np.nan], [np.nan, np.nan]],
    ])
    path = tmp_path / 'accuracy_grid__val.npy'
    np.save(path, array)
    return path, array


def test_aggregation_max(tmp_path):
    path, _ = _save_grid(tmp_path)
    aggregated = _load_and_aggregate_accuracy_grid(path, seed_aggregation='max')
    assert aggregated.shape == (2, 2)
    assert aggregated[0, 0] == pytest.approx(0.8)
    assert aggregated[1, 0] == pytest.approx(0.5)   # single-seed cell: NaN ignored
    assert np.isnan(aggregated[0, 1]) and np.isnan(aggregated[1, 1])  # all-NaN stays NaN


def test_aggregation_mean(tmp_path):
    path, _ = _save_grid(tmp_path)
    aggregated = _load_and_aggregate_accuracy_grid(path, seed_aggregation='mean')
    assert aggregated[0, 0] == pytest.approx(0.5)
    assert aggregated[1, 0] == pytest.approx(0.5)
    assert np.isnan(aggregated[0, 1])


def test_aggregation_none_returns_raw_stack(tmp_path):
    path, array = _save_grid(tmp_path)
    raw = _load_and_aggregate_accuracy_grid(path, seed_aggregation=None)
    np.testing.assert_array_equal(raw, array)


def test_aggregation_unknown_raises(tmp_path):
    path, _ = _save_grid(tmp_path)
    with pytest.raises(RuntimeError):
        _load_and_aggregate_accuracy_grid(path, seed_aggregation='median')


# ---------- grid flattening ----------

def test_flatten_D_N_drops_over_capacity_cells():
    axes = GridAxes(x=GridAxis(name='D', axis=np.array([8, 16])),
                    y=GridAxis(name='N', axis=np.array([8, 16])))
    grid = np.array([[0.1, 0.2],
                     [0.3, 0.4]])
    D, N, accuracy = flatten_D_N_grid_data(grid, axes, GridConstants({'V': 64}))
    # (D=8, N=16) violates N <= D and is dropped
    assert list(zip(D, N)) == [(8, 8), (16, 8), (16, 16)]
    np.testing.assert_allclose(accuracy, [0.1, 0.3, 0.4])


def test_flatten_N_Lambda_drops_cells_above_capacity():
    axes = GridAxes(x=GridAxis(name='N', axis=np.array([8, 16])),
                    y=GridAxis(name='Lambda', axis=np.array([1, 2])))
    grid = np.array([[0.1, 0.2],
                     [0.3, 0.4]])
    # callers pass grid constants as a plain dict here (see make_scaling_curve_figures)
    N, Lambda, accuracy = flatten_N_Lambda_grid_data(grid, axes, {'D': 16})
    # Lambda*N > D=16 drops (N=16, Lambda=2)
    assert list(zip(N, Lambda)) == [(8, 1), (8, 2), (16, 1)]
    np.testing.assert_allclose(accuracy, [0.1, 0.2, 0.3])
