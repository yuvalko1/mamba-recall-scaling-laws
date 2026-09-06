import numpy as np

from experiments.grid_runs import (
    GridPoint, Task, _get_seeds_to_run, _initialize_grid, _should_skip_grid_point)
from experiments.grid_runs_utils import GridConstants
from mqar import MqarDimensions


def test_grid_point_str_zero_pads_and_safe_name():
    point = GridPoint(name='D', value=16)
    assert str(point) == 'D=0016'
    assert point.safe_name == 'D_0016'


def test_task_dims_merge_point_and_constants():
    constants = GridConstants({'V': 64, 'L': 32, 'N_facts': 4, 'Lambda': 1, 'M': 1})
    task = Task(x=GridPoint('D', 16), y=GridPoint('N', 8), constants=constants, seed=2)
    assert task.dims == MqarDimensions(V=64, L=32, N_facts=4, D=16, N=8, Lambda=1, M=1)
    assert task.seed == 2
    assert 'seed=2' in task.name


def test_seeds_range_from_config():
    assert list(_get_seeds_to_run({'n_seeds': 3, 'start_seed': 5})) == [5, 6, 7]
    assert list(_get_seeds_to_run({})) == [0]


def test_initialize_grid_is_nan_with_axis_shapes():
    grid_config = {
        'V': 64, 'L': 32, 'N_facts': 4, 'Lambda': 1, 'M': 1,
        'D': 'np.array([16, 32, 48])',
        'N': 'np.array([8, 16])',
        'x_axis': 'D', 'y_axis': 'N',
    }
    empty_grid, x_axis, y_axis = _initialize_grid(grid_config)
    assert empty_grid.shape == (3, 2)
    assert np.isnan(empty_grid).all()
    np.testing.assert_array_equal(x_axis, [16, 32, 48])
    np.testing.assert_array_equal(y_axis, [8, 16])


def test_skip_rules():
    def dims(**overrides):
        base = dict(V=64, L=32, N_facts=4, D=16, N=8, Lambda=1, M=1)
        base.update(overrides)
        return MqarDimensions(**base)

    assert not _should_skip_grid_point(dims())
    # capacity rule: state dim above embedding dim
    assert _should_skip_grid_point(dims(N=17, D=16))
    assert not _should_skip_grid_point(dims(N=16, D=16))
    # zoology feasibility rules
    assert _should_skip_grid_point(dims(L=64, V=64))     # L >= V
    assert _should_skip_grid_point(dims(L=15, N_facts=4))  # L < 4 * N_facts
    assert not _should_skip_grid_point(dims(L=16, N_facts=4))
