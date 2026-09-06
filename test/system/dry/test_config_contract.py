"""Contract tests over every committed run config: they must parse, carry the
sections the pipeline reads, and describe a feasible grid. This is what makes
`config/**/*.json5` trustworthy for release without launching anything."""
from dataclasses import fields
from pathlib import Path

import json5
import numpy as np
import pytest

from experiments.grid_runs_utils import get_axes_from_grid_config
from mqar import MqarDimensions

PROJECT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_DIR / 'config'

RUN_CONFIG_PATHS = sorted(CONFIG_DIR.rglob('*.json5'))
CONFIG_IDS = [str(p.relative_to(CONFIG_DIR)) for p in RUN_CONFIG_PATHS]

REQUIRED_SECTIONS = ('runtime', 'seeds', 'model', 'dataset',
                     'wandb', 'grid_options', 'grid')
KNOWN_MODEL_CLASSES = ('mamba_ssm', 'mamba_tiny', 'mamba_theory')
DIM_NAMES = {f.name for f in fields(MqarDimensions)}

# the release rule set while preparing reproducibility: trained run configs never
# save models - except the mech-interp config, which must save the best model per
# cell (feeds the mech-interp figures) but no per-evaluation checkpoints.
# flag semantics (src/experiments/train.py): save_model_checkpoints -> best+final model;
# save_model_at_evaluation -> a checkpoint at every evaluation.
MECH_INTERP_CONFIG_NAME = 'MQAR__D_N__linear_trained__mech_interp.json5'


def _load(path: Path) -> dict:
    return json5.loads(path.read_text())


def test_some_configs_exist():
    assert len(RUN_CONFIG_PATHS) > 20


@pytest.mark.parametrize('config_path', RUN_CONFIG_PATHS, ids=CONFIG_IDS)
def test_config_parses_with_required_sections(config_path):
    run_config = _load(config_path)
    for section in REQUIRED_SECTIONS:
        assert section in run_config, f'missing section {section!r}'
    assert run_config['model']['class'] in KNOWN_MODEL_CLASSES
    # forward-only configs: closed-form theory, or designed weights (evaluation only)
    if run_config['runtime'].get('should_train', True):
        assert 'training' in run_config, "trainable config must carry a 'training' section"
    else:
        is_theory = run_config['model']['class'] == 'mamba_theory'
        is_designed = run_config['model'].get('kwargs', {}).get('set_designed_mqar_weights', False)
        assert is_theory or is_designed, \
            'should_train: false is only for theory or designed-weights configs'


@pytest.mark.parametrize('config_path', RUN_CONFIG_PATHS, ids=CONFIG_IDS)
def test_grid_axes_are_valid(config_path):
    grid_config = _load(config_path)['grid']
    assert grid_config['x_axis'] in DIM_NAMES
    assert grid_config['y_axis'] in DIM_NAMES
    grid_axes, grid_constants = get_axes_from_grid_config(grid_config)
    for axis in (grid_axes.x.axis, grid_axes.y.axis):
        assert isinstance(axis, np.ndarray) and axis.size >= 1
        assert axis.dtype.kind == 'i'
        assert (axis > 0).all()
    # every non-axis grid key that names a dimension must be an int or None
    for key, value in grid_constants.as_dict().items():
        assert key in DIM_NAMES
        assert value is None or isinstance(value, int), (key, value)
    # scaling rules carry their required companion keys
    if grid_config.get('scale_N_with_Lambda', False):
        assert grid_config.get('Lambda_N'), 'scale_N_with_Lambda needs Lambda_N'


@pytest.mark.parametrize('config_path', RUN_CONFIG_PATHS, ids=CONFIG_IDS)
def test_training_and_dataset_sections_are_sane(config_path):
    run_config = _load(config_path)
    training = run_config.get('training')
    if training is not None:
        assert training['max_num_steps'] >= 1
        assert training['optimizer']['learning_rate'] > 0
    dataset = run_config['dataset']
    if 'batch_size' in dataset:  # theory configs skip the dataloader path entirely
        assert dataset['batch_size'] >= 1
        # train may be omitted (derived from max_num_steps * batch_size at runtime)
        assert {'val', 'test'} <= set(dataset['split_size']) <= {'train', 'val', 'test'}
    seeds = run_config['seeds']
    assert seeds['n_seeds'] >= 1


@pytest.mark.parametrize('config_path', RUN_CONFIG_PATHS, ids=CONFIG_IDS)
def test_initialize_run_name_and_archived_config(config_path):
    """Every config must initialize cleanly: a wandb project name within the
    128-char limit and in the descriptive format, and a byte-identical archived
    run_config.json5."""
    import shutil
    from experiments.grid_runs_utils import initialize_grid_run
    from utils.common import test_results_dir

    run_config = _load(config_path)
    run_config['runtime']['experiment_name'] = config_path.stem
    run_config['runtime']['is_test'] = True
    run_config['wandb']['activate'] = False
    try:
        grid_run_name = initialize_grid_run(run_config, config_path=config_path)
        assert len(grid_run_name) <= 128, grid_run_name
        assert grid_run_name.startswith(f'{config_path.stem} | '), grid_run_name
        saved = Path(run_config['io']['run_results_dir']) / 'run_config.json5'
        assert saved.read_bytes() == config_path.read_bytes()
    finally:
        shutil.rmtree(test_results_dir / config_path.stem, ignore_errors=True)


@pytest.mark.parametrize('config_path', RUN_CONFIG_PATHS, ids=CONFIG_IDS)
def test_no_unintended_checkpoint_saving(config_path):
    run_config = _load(config_path)
    training = run_config.get('training', {})
    saves_best_model = training.get('save_model_checkpoints', False)
    saves_per_evaluation = training.get('save_model_at_evaluation', False)

    if not run_config['runtime'].get('should_train', True):
        return  # forward-only (theory/designed) runs are exempt from the rule

    if config_path.name == MECH_INTERP_CONFIG_NAME:
        assert saves_best_model, 'the mech-interp config must save the best model per cell'
        assert not saves_per_evaluation, 'mech-interp must not save per-evaluation checkpoints'
    else:
        assert not saves_best_model and not saves_per_evaluation, \
            'only the mech-interp config may save models'
