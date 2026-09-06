"""The full multiprocessing grid machinery (spawn workers + logger + queues +
grid saving) exercised end-to-end WITHOUT any training or GPU:

- a mamba_theory grid computes closed-form accuracies per cell, so the whole
  parent/worker/logger pipeline runs on cpu workers in seconds;
- a grid whose every point is infeasible (N > D) enqueues zero tasks and must
  start up and shut down cleanly rather than wait on results that don't exist.
"""
import shutil
from pathlib import Path

import json5
import numpy as np
import pytest

from experiments.grid_runs import execute_grid_run
from experiments.grid_runs_utils import initialize_grid_run
from utils.common import test_dir, test_results_dir

MICRO_GRID_CONFIG_PATH = test_dir / 'fixtures' / 'micro_grid.json5'


def _theory_run_config(experiment_name: str) -> dict:
    run_config = json5.loads(MICRO_GRID_CONFIG_PATH.read_text())
    run_config['runtime']['experiment_name'] = experiment_name
    run_config['model'] = {'class': 'mamba_theory', 'variant': 'theory', 'task_type': 'MQAR'}
    run_config['parallel']['devices_to_use'] = ['cpu']
    return run_config


@pytest.fixture()
def clean_experiment_dirs():
    yield
    for name in ('machinery_theory_grid', 'machinery_all_skipped'):
        shutil.rmtree(test_results_dir / name, ignore_errors=True)


def test_theory_grid_runs_full_machinery_on_cpu(clean_experiment_dirs):
    run_config = _theory_run_config('machinery_theory_grid')

    grid_run_name = initialize_grid_run(run_config)
    execute_grid_run(grid_run_name, run_config)

    run_results_dir = Path(run_config['io']['run_results_dir'])
    grid = np.load(run_results_dir / 'accuracy_grid__val.npy')
    assert grid.shape == (1, 2, 2)
    assert not np.isnan(grid).any(), f'every feasible cell must report: {grid}'
    assert ((grid >= 0.0) & (grid <= 1.0)).all(), grid


def test_all_skipped_grid_shuts_down_cleanly(clean_experiment_dirs):
    run_config = _theory_run_config('machinery_all_skipped')
    run_config['grid']['D'] = 'np.array([8, 16])'
    run_config['grid']['N'] = 'np.array([32, 64])'  # N > D everywhere -> all skipped

    grid_run_name = initialize_grid_run(run_config)
    execute_grid_run(grid_run_name, run_config)  # zero tasks: must return, not hang

    # no task ever logged, so no accuracy grid is written - only the run config
    run_results_dir = Path(run_config['io']['run_results_dir'])
    assert not (run_results_dir / 'accuracy_grid__val.npy').exists()
    assert (run_results_dir / 'run_config.json5').is_file()
