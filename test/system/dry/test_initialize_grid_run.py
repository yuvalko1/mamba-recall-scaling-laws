"""initialize_grid_run with runtime.is_test routes results under test/results/ and
persists the run config; wandb auth is only demanded when wandb is activated."""
import shutil
from pathlib import Path

import json5
import pytest

from experiments.grid_runs_utils import _ensure_wandb_authenticated, initialize_grid_run
from utils.common import test_results_dir

EXPERIMENT_NAME = 'test_initialize_grid_run'


def _run_config(wandb_activate=False):
    return {
        'runtime': {'experiment_name': EXPERIMENT_NAME, 'seed': 0, 'is_test': True},
        'model': {'class': 'mamba_tiny', 'variant': 'simplified'},
        'wandb': {'activate': wandb_activate},
        'io': {'run_results_dir': None, 'time_zone': None},
        'grid': {
            'V': 64, 'L': 32, 'N_facts': 4, 'Lambda': 1, 'M': 1,
            'D': 'np.array([16, 32])', 'N': 'np.array([8, 16])',
            'x_axis': 'D', 'y_axis': 'N',
        },
    }


@pytest.fixture()
def clean_experiment_dir():
    experiment_dir = test_results_dir / EXPERIMENT_NAME
    yield experiment_dir
    shutil.rmtree(experiment_dir, ignore_errors=True)


def test_creates_test_results_dir_and_saves_config(clean_experiment_dir):
    run_config = _run_config()
    grid_run_name = initialize_grid_run(run_config)

    run_results_dir = Path(run_config['io']['run_results_dir'])
    assert run_results_dir.is_dir()
    assert run_results_dir.parent == clean_experiment_dir, \
        'is_test must route under test/results/<experiment_name>/'

    # dir name encodes model, axes and constants (N_k/N_q hidden at default 1)
    assert run_results_dir.name.startswith('mamba_tiny_simplified__D_N_')
    assert 'V64' in run_results_dir.name and 'Nf4' in run_results_dir.name

    saved = json5.loads((run_results_dir / 'run_config.json5').read_text())
    assert saved['grid'] == run_config['grid']
    assert saved['io']['run_results_dir'] == str(run_results_dir)
    # wandb project name: <experiment> | <task>, <paper-variant>, <trained?> | ...
    assert grid_run_name.startswith(f"{EXPERIMENT_NAME} | MQAR, linear, trained | ")


def test_wandb_activation_requires_credentials(monkeypatch, tmp_path, clean_experiment_dir):
    # no key anywhere -> activated wandb must fail fast (not hang at wandb.login)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    empty_netrc = tmp_path / 'netrc'
    empty_netrc.write_text('')
    monkeypatch.setenv('NETRC', str(empty_netrc))
    with pytest.raises(RuntimeError, match='wandb'):
        initialize_grid_run(_run_config(wandb_activate=True))

    # with a key present the check passes
    monkeypatch.setenv('WANDB_API_KEY', 'dummy')
    _ensure_wandb_authenticated()
    initialize_grid_run(_run_config(wandb_activate=True))
