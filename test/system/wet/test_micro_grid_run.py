"""Full training pipeline, end to end on one GPU: initialize_grid_run +
execute_grid_run over test/config/micro_grid.json5 (a 2x2 D-N grid of tiny
models), exercising the real mp-spawn worker/logger machinery. Asserts the
artifact contract: a complete accuracy grid saved with no NaN cells - exactly
what a hung/dead worker would violate."""
import shutil
from pathlib import Path

import json5
import numpy as np
import pytest

from experiments.grid_runs import execute_grid_run
from experiments.grid_runs_utils import initialize_grid_run
from utils.common import test_dir, test_results_dir

MICRO_GRID_CONFIG_PATH = test_dir / 'fixtures' / 'micro_grid.json5'


@pytest.fixture()
def clean_experiment_dir():
    experiment_dir = test_results_dir / 'micro_grid'
    yield experiment_dir
    shutil.rmtree(experiment_dir, ignore_errors=True)


@pytest.mark.cpu_ok
def test_micro_grid_trains_and_saves_complete_grid(training_device, clean_experiment_dir):
    run_config = json5.loads(MICRO_GRID_CONFIG_PATH.read_text())
    run_config['parallel']['devices_to_use'] = [training_device]

    grid_run_name = initialize_grid_run(run_config)
    run_results_dir = Path(run_config['io']['run_results_dir'])
    assert run_results_dir.is_dir()
    assert run_results_dir.parent == clean_experiment_dir

    execute_grid_run(grid_run_name, run_config)

    # artifact contract
    saved_files = {p.name for p in run_results_dir.iterdir()}
    assert 'run_config.json5' in saved_files
    assert 'accuracy_grid__val.npy' in saved_files

    grid = np.load(run_results_dir / 'accuracy_grid__val.npy')
    assert grid.shape == (1, 2, 2), 'expected (seeds=1, D-axis=2, N-axis=2)'
    assert not np.isnan(grid).any(), \
        f'all 4 cells must complete (hung/dead-worker contract): {grid}'
    assert ((grid >= 0.0) & (grid <= 1.0)).all(), grid

    # train/test grids are logged too (train from batch stats, test at final eval)
    for split in ('train', 'test'):
        split_path = run_results_dir / f'accuracy_grid__{split}.npy'
        assert split_path.is_file(), f'missing accuracy_grid__{split}.npy'
