"""Referential integrity of scripts/figures/config.json5: every active entry must
point at an existing committed bundle with a run_config and accuracy grid, so the
figure scripts reproduce the paper figures from a fresh clone."""
from pathlib import Path

import json5
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[3]
FIGURES_CONFIG_PATH = PROJECT_DIR / 'scripts' / 'figures' / 'config.json5'

_config = json5.loads(FIGURES_CONFIG_PATH.read_text())
GRID_ENTRIES = [(name, entry) for name, entry in _config['grids'].items() if entry is not None]


def test_config_has_expected_sections():
    assert 'defaults' in _config and 'grids' in _config and 'mech_interp' in _config
    assert len(GRID_ENTRIES) >= 20


@pytest.mark.parametrize('name,entry', GRID_ENTRIES, ids=[n for n, _ in GRID_ENTRIES])
def test_grid_entry_points_at_complete_bundle(name, entry):
    settings = {**_config.get('defaults', {}), **entry}
    run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']
    assert run_dir.is_dir(), f'{name}: bundle missing: {run_dir}'
    assert (run_dir / 'run_config.json5').is_file(), f'{name}: bundle has no run_config.json5'
    split = settings.get('split', 'val')
    assert (run_dir / f'accuracy_grid__{split}.npy').is_file(), \
        f'{name}: bundle has no accuracy_grid__{split}.npy'


def test_bundle_configs_are_byte_identical_to_config_files():
    """saved_runs_data holds exactly one final bundle per family (only what the
    paper + appendix figures need), and each bundle's archived run_config.json5
    is a byte-identical copy of the family's config/ file."""
    config_files = {p.stem: p for p in (PROJECT_DIR / 'config').rglob('*.json5')}
    checked = 0
    for family_dir in sorted((PROJECT_DIR / 'saved_runs_data').iterdir()):
        if not family_dir.is_dir():
            continue
        config_path = config_files.get(family_dir.name)
        assert config_path is not None, f'{family_dir.name}: bundle family without a config file'
        bundles = [d for d in family_dir.iterdir() if (d / 'run_config.json5').is_file()]
        assert len(bundles) == 1, f'{family_dir.name}: expected exactly one final bundle, found {len(bundles)}'
        assert (bundles[0] / 'run_config.json5').read_bytes() == config_path.read_bytes(), \
            f'{family_dir.name}/{bundles[0].name}: archived config differs from {config_path}'
        checked += 1
    assert checked >= 20, f'only {checked} families checked - sweep looks broken'


def test_mech_interp_entry_is_consistent_with_its_bundle():
    # the mech_interp run itself lives under results/ (checkpoints are not archived),
    # but the same run's grid bundle is committed and referenced from `grids`
    mech = _config['mech_interp']
    assert {'run', 'cells', 'plot_kwargs'} <= set(mech)
    grids_entry = _config['grids'].get('MQAR__D_N__linear_trained__mech_interp')
    assert grids_entry is not None, 'mech-interp grid bundle entry missing from grids'
    assert grids_entry['run'] == mech['run'], \
        'mech_interp.run and its grids entry point at different runs'


def test_mech_interp_cells_are_distinct_with_distinct_seeds():
    cells = _config['mech_interp']['cells']
    assert len(cells) >= 1
    for cell in cells:
        assert {'D', 'N', 'seed'} <= set(cell)
    assert len({(c['D'], c['N']) for c in cells}) == len(cells), 'duplicate (D, N) cell'
    assert len({c['seed'] for c in cells}) == len(cells), 'input-sequence seeds must differ'
    # the paper-body cell (first) stays the Fig-3 cell
    assert (cells[0]['D'], cells[0]['N']) == (64, 32)
