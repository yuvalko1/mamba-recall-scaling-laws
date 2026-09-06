"""End-to-end: ALL figure scripts run headless against the committed
saved_runs_data bundles, into a tmp dir (the committed figures/ tree is never
touched - asserted explicitly). The produced set is pinned EXACTLY: per-script
counts, body/appendix split, and (for the config-driven scripts) the exact
filename sets derived from figure_scripts/config.json5."""
import importlib
import sys
from pathlib import Path

import json5
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[3]
FIGURE_SCRIPTS_DIR = PROJECT_DIR / 'scripts' / 'figures'

for extra_path in (FIGURE_SCRIPTS_DIR, FIGURE_SCRIPTS_DIR / 'appendix'):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# import every script up front so all modules that bind save_figure_to_png exist
# before the redirect patches them
SCRIPT_NAMES = [
    'make_grid_figures',
    'make_scaling_curve_figures',
    'make_mech_interp_figures',
    'make_large_vocab_figures',
    'make_repeats_figures',
    'make_seeds_analysis_figures',
    'make_circuit_ablation_figure',
]
SCRIPTS = {name: importlib.import_module(name) for name in SCRIPT_NAMES}

# pinned inventory: (body, appendix) PNGs per script - update deliberately when
# the figure set changes
EXPECTED_COUNTS = {
    'make_grid_figures': (11, 10),
    'make_scaling_curve_figures': (3, 8),
    'make_mech_interp_figures': (3, 24),
    'make_large_vocab_figures': (0, 2),
    'make_repeats_figures': (0, 1),
    'make_seeds_analysis_figures': (0, 8),
    'make_circuit_ablation_figure': (0, 1),
}
EXPECTED_TOTAL, EXPECTED_BODY, EXPECTED_APPENDIX = 71, 17, 54


def _figures_snapshot():
    return {p: p.stat().st_mtime for p in (PROJECT_DIR / 'figures').rglob('*.png')}


@pytest.fixture(scope='module')
def figure_inventory(tmp_path_factory):
    """Run every figure script once, redirected into a tmp dir; yields
    {script: [relative png paths]} and asserts the repo figures/ was untouched."""
    from utils import plot as plot_module
    from utils.common import figures_dir

    tmp_root = tmp_path_factory.mktemp('figures')
    original = plot_module.save_figure_to_png
    produced: dict[str, list[str]] = {name: [] for name in SCRIPT_NAMES}
    current = {'script': None}

    def redirected(fig, save_dir, save_name, **kwargs):
        save_dir = Path(save_dir)
        try:
            relative = save_dir.relative_to(figures_dir)
        except ValueError:
            relative = Path(save_dir.name)
        original(fig, tmp_root / relative, save_name, **kwargs)
        produced[current['script']].append(str(relative / f'{save_name}.png'))

    patched = []
    for module in list(sys.modules.values()):
        # __dict__ access on purpose: getattr would trigger lazy-module imports
        module_vars = getattr(module, '__dict__', None)
        if module_vars is not None and module_vars.get('save_figure_to_png') is original:
            module_vars['save_figure_to_png'] = redirected
            patched.append(module_vars)

    repo_figures_before = _figures_snapshot()
    try:
        for name, module in SCRIPTS.items():
            current['script'] = name
            if name == 'make_mech_interp_figures':
                module.main(argv=[])  # don't let argparse read pytest's argv
            else:
                module.main()
    finally:
        for module_vars in patched:
            module_vars['save_figure_to_png'] = original

    assert _figures_snapshot() == repo_figures_before, \
        'figure scripts must never touch the committed figures/ tree during tests'
    # every recorded file actually exists on disk
    for name, files in produced.items():
        for f in files:
            assert (tmp_root / f).is_file(), (name, f)
    return produced


def _split(files):
    body = sorted(f for f in files if not f.startswith('appendix'))
    appendix = sorted(f for f in files if f.startswith('appendix'))
    return body, appendix


@pytest.mark.parametrize('script', SCRIPT_NAMES)
def test_per_script_counts_are_exact(figure_inventory, script):
    body, appendix = _split(figure_inventory[script])
    assert (len(body), len(appendix)) == EXPECTED_COUNTS[script], \
        f'{script}: body={body} appendix={appendix}'


def test_total_counts_are_exact(figure_inventory):
    all_files = [f for files in figure_inventory.values() for f in files]
    body, appendix = _split(all_files)
    assert len(set(all_files)) == len(all_files), 'duplicate figure paths across scripts'
    assert (len(all_files), len(body), len(appendix)) == \
        (EXPECTED_TOTAL, EXPECTED_BODY, EXPECTED_APPENDIX)


def test_grid_figures_match_config_exactly(figure_inventory):
    config = json5.loads((FIGURE_SCRIPTS_DIR / 'config.json5').read_text())
    expected = set()
    for name, entry in config['grids'].items():
        if entry is None or not entry.get('grid_figure', True):
            continue
        subdir = entry.get('grid_subdir', '')
        expected.add(str(Path(subdir) / f'{name}.png') if subdir else f'{name}.png')
    assert set(figure_inventory['make_grid_figures']) == expected


def test_mech_interp_figures_match_cells_exactly(figure_inventory):
    config = json5.loads((FIGURE_SCRIPTS_DIR / 'config.json5').read_text())
    panels = ['hidden_state__H_ideal', 'hidden_state__H_noisy', 'hidden_state__H_projected',
              'attention_maps__alpha_ideal', 'attention_maps__alpha_noisy',
              'convolution__x_Gvv_xi', 'convolution__xi_Gkq_xi',
              'invariant_operators__Gkq', 'invariant_operators__Gvv']
    expected = set()
    for i, cell in enumerate(config['mech_interp']['cells']):
        tag = f"D{cell['D']}_N{cell['N']}"
        for panel in panels:
            body = (i == 0) and panel.startswith('hidden_state')
            name = f'mech_interp__{tag}__{panel}.png'
            expected.add(name if body else f'appendix/{name}')
    assert set(figure_inventory['make_mech_interp_figures']) == expected


def test_curve_figures_are_curves_with_expected_split(figure_inventory):
    files = figure_inventory['make_scaling_curve_figures']
    assert all(f.endswith('__acc_curve__sigma_inv.png') for f in files), files
    assert 'MQAR__D_N__linear_trained__regime_2__acc_curve__sigma_inv.png' in files


def test_make_all_figures_sh_in_sync_with_pinned_inventory():
    """figure_scripts/make_all_figures.sh runs every maker and checks the same
    pinned totals as this file - guard both against drifting."""
    import re
    sh = (FIGURE_SCRIPTS_DIR / 'make_all_figures.sh').read_text()

    sh_scripts = re.findall(r'^\s+((?:appendix/)?make_\w+\.py)$', sh, flags=re.MULTILINE)
    assert [Path(s).stem for s in sh_scripts] == SCRIPT_NAMES

    assert int(re.search(r'^EXPECTED_BODY=(\d+)$', sh, flags=re.MULTILINE).group(1)) == EXPECTED_BODY
    assert int(re.search(r'^EXPECTED_APPENDIX=(\d+)$', sh, flags=re.MULTILINE).group(1)) == EXPECTED_APPENDIX
