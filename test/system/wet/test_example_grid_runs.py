"""End-to-end runs of the two __example example configs (4x4 D-N grid, one GPU,
16 workers = one wave), with the real regime-0 recipes. Asserts the artifact
contract, the capacity-skip NaN pattern, the byte-identical archived config,
and the descriptive wandb project name."""
import shutil
from pathlib import Path

import json5
import numpy as np
import pytest

from experiments.grid_runs import execute_grid_run
from experiments.grid_runs_utils import initialize_grid_run
from utils.common import test_results_dir

PROJECT_DIR = Path(__file__).resolve().parents[3]

# D rows x N cols; cells with N > D are capacity-skipped and stay NaN
D_AXIS = [16, 32, 48, 64]
N_AXIS = [8, 16, 24, 32]
EXPECTED_NAN = np.array([[n > d for n in N_AXIS] for d in D_AXIS])

SMALL_CONFIGS = {
    'MQAR__D_N__linear_trained__example': 'MQAR, linear, trained',
    'MQAR__D_N__full_trained__example': 'MQAR, full, trained',
}

_PARAMS = [
    pytest.param(name, descriptors, marks=pytest.mark.cpu_ok) if 'linear' in name
    else pytest.param(name, descriptors)
    for name, descriptors in SMALL_CONFIGS.items()
]


@pytest.mark.parametrize('name,descriptors', _PARAMS, ids=list(SMALL_CONFIGS))
def test_example_grid_trains_end_to_end(training_device, name, descriptors):
    if 'full' in name:
        pytest.importorskip('mamba_ssm')
        if training_device == 'cpu':
            pytest.skip('full Mamba (mamba_ssm CUDA kernels) requires a GPU')

    config_path = PROJECT_DIR / 'config' / f'{name}.json5'
    run_config = json5.loads(config_path.read_text())
    run_config['runtime']['experiment_name'] = name
    run_config['runtime']['is_test'] = True
    run_config['wandb']['activate'] = False
    run_config['parallel']['devices_to_use'] = [training_device]

    # keep the config's real recipe on disk, but shorten the run for the test:
    # enough steps to exercise the full pipeline, not enough to converge
    run_config['training'].update(
        max_num_steps=120, evaluate_any_num_steps=40, save_logs_any_num_steps=40,
        scheduler={'num_warmup_steps': 20, 'num_steps_at_max': 20,
                   'decay_type': 'cosine', 'num_decay_steps': 80, 'decay_factor': 0.0},
    )

    try:
        grid_run_name = initialize_grid_run(run_config, config_path=config_path)
        assert grid_run_name.startswith(f"{name} | {descriptors} | ")

        run_results_dir = Path(run_config['io']['run_results_dir'])
        assert (run_results_dir / 'run_config.json5').read_bytes() == config_path.read_bytes()

        execute_grid_run(grid_run_name, run_config)

        grid = np.load(run_results_dir / 'accuracy_grid__val.npy')
        assert grid.shape == (1, 4, 4), grid.shape
        nan_mask = np.isnan(grid[0])
        np.testing.assert_array_equal(nan_mask, EXPECTED_NAN,
                                      err_msg='NaNs must appear exactly at capacity-skipped (N > D) cells')
        values = grid[0][~nan_mask]
        assert ((values >= 0.0) & (values <= 1.0)).all(), values
    finally:
        shutil.rmtree(test_results_dir / name, ignore_errors=True)
