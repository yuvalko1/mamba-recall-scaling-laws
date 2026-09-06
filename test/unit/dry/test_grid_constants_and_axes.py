import numpy as np
import pytest

from mqar import MqarDimensions
from experiments.grid_runs_utils import (
    GridConstants, get_axes_from_grid_config, prepare_grid_pairs_to_iterate, build_dataloaders)


# ---------- GridConstants naming ----------

def test_default_skip_hides_unit_repeats_from_names():
    constants = GridConstants({'V': 512, 'N_facts': 10, 'N_k': 1, 'N_q': 1})
    assert constants.name == 'V=512, Nf=10'
    assert constants.safe_name == 'V512_Nf10'


def test_non_default_repeats_are_shown():
    constants = GridConstants({'V': 512, 'N_k': 3, 'N_q': 1})
    assert constants.name == 'V=512, N_k=3'
    assert constants.safe_name == 'V512_N_k3'


def test_none_valued_constant_renders_bare_key():
    # a None constant (e.g. N when scale_N_with_Lambda fills it per-task) renders
    # as the bare key with no '=value'
    constants = GridConstants({'D': 100, 'N': None})
    assert constants.name == 'D=100, N'
    assert constants.safe_name == 'D100_N'


def test_alias_and_as_dict_roundtrip():
    data = {'N_facts': 16, 'V': 64}
    constants = GridConstants(data)
    assert constants.as_dict() == data
    assert constants._key('N_facts') == 'Nf'
    assert constants._key('V') == 'V'


def test_non_int_constant_raises():
    with pytest.raises(TypeError):
        GridConstants({'V': 1.5}).name


# ---------- grid-config parsing ----------

def _grid_config():
    return {
        'V': 64,
        'L': 32,
        'N_facts': 4,
        'D': 'np.array([16, 32])',
        'N': 'np.arange(8, 17, 8)',  # [8, 16]
        'Lambda': 1,
        'M': 1,
        'x_axis': 'D',
        'y_axis': 'N',
    }


def test_axes_eval_to_int_arrays_and_constants_exclude_axes():
    grid_axes, grid_constants = get_axes_from_grid_config(_grid_config())
    assert grid_axes.x.name == 'D' and grid_axes.y.name == 'N'
    np.testing.assert_array_equal(grid_axes.x.axis, [16, 32])
    np.testing.assert_array_equal(grid_axes.y.axis, [8, 16])
    assert grid_axes.x.axis.dtype.kind == 'i'
    constants = grid_constants.as_dict()
    assert 'D' not in constants and 'N' not in constants
    assert constants == {'V': 64, 'L': 32, 'N_facts': 4, 'Lambda': 1, 'M': 1}


def test_prepare_grid_pairs_covers_full_cartesian_product():
    grid_axes, grid_constants = get_axes_from_grid_config(_grid_config())
    xy_pairs, dims = prepare_grid_pairs_to_iterate(grid_axes, grid_constants, grid_options={})
    assert sorted(xy_pairs) == [(16, 8), (16, 16), (32, 8), (32, 16)]
    assert isinstance(dims, MqarDimensions)
    assert dims.V == 64 and dims.D is None and dims.N is None


# ---------- build_dataloaders config filling ----------

def _run_config_for_dataloaders():
    return {
        'runtime': {'seed': 0},
        'dataset': {
            'V': None, 'L': None, 'N_facts': None,
            'kwargs': {'num_key_repeats': 1, 'num_query_repeats': 1},
            'batch_size': 8,
            'split_size': {'train': None, 'val': 16, 'test': 16},
        },
        'training': {'max_num_steps': 5},
    }


def test_build_dataloaders_fills_dims_and_derives_train_size():
    run_config = _run_config_for_dataloaders()
    dims = MqarDimensions(V=64, L=32, N_facts=4, D=16, N=8, Lambda=1, M=1)
    dataloaders = build_dataloaders(dims, run_config)
    dataset_config = run_config['dataset']
    assert dataset_config['V'] == 64 and dataset_config['L'] == 32 and dataset_config['N_facts'] == 4
    # train split defaults to max_num_steps * batch_size
    assert dataset_config['split_size']['train'] == 5 * 8
    assert len(dataloaders['train']) == 5


def test_build_dataloaders_maps_repeat_dims_into_generator_kwargs():
    run_config = _run_config_for_dataloaders()
    dims = MqarDimensions(V=64, L=32, N_facts=4, D=16, N=8, Lambda=1, M=1, N_k=3, N_q=2)
    build_dataloaders(dims, run_config)
    assert run_config['dataset']['kwargs']['num_key_repeats'] == 3
    assert run_config['dataset']['kwargs']['num_query_repeats'] == 2
