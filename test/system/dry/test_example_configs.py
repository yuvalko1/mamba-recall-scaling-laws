"""The __example example configs: initialize_grid_run must archive a BYTE-IDENTICAL
copy of the source config file (runtime-filled fields live only in memory), and
the wandb project name must follow the descriptive format."""
import shutil
from pathlib import Path

import json5
import pytest

from experiments.grid_runs_utils import initialize_grid_run
from utils.common import test_results_dir

PROJECT_DIR = Path(__file__).resolve().parents[3]

SMALL_CONFIGS = {
    'MQAR__D_N__linear_trained__example': 'MQAR, linear, trained',
    'MQAR__D_N__full_trained__example': 'MQAR, full, trained',
}


@pytest.mark.parametrize('name,descriptors', SMALL_CONFIGS.items(), ids=list(SMALL_CONFIGS))
def test_saved_config_is_byte_identical_and_name_is_descriptive(name, descriptors):
    config_path = PROJECT_DIR / 'config' / f'{name}.json5'
    run_config = json5.loads(config_path.read_text())

    # runtime-only mutations - must NOT leak into the archived config.
    # distinct experiment name: the wet example test trains under test_results_dir/<name>,
    # and this test's cleanup must never be able to touch that directory
    experiment_name = f'{name}__contract'
    run_config['runtime']['experiment_name'] = experiment_name
    run_config['runtime']['is_test'] = True
    run_config['wandb']['activate'] = False

    try:
        grid_run_name = initialize_grid_run(run_config, config_path=config_path)

        saved = Path(run_config['io']['run_results_dir']) / 'run_config.json5'
        assert saved.read_bytes() == config_path.read_bytes(), \
            'archived run_config.json5 must be a byte-identical copy of the source config'

        expected_prefix = f"{experiment_name} | {descriptors} | D, N, V=512, L=64, Nf=16, \u039b=1, M=1 | "
        assert grid_run_name.startswith(expected_prefix), grid_run_name
        assert len(grid_run_name) <= 128, grid_run_name
    finally:
        shutil.rmtree(test_results_dir / experiment_name, ignore_errors=True)
