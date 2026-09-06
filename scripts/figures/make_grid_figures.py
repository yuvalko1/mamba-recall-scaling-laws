"""
Generic accuracy-grid figure maker.

Renders every entry of `grids` in scripts/figures/config.json5 into <root>/figures/<name>.png.
Each entry points at a saved run bundle (by default under <root>/saved_runs_data/, the
committed release artifacts; set root: 'results' for local, non-archived runs).

Usage (from anywhere):
    python scripts/figures/make_grid_figures.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt

from experiments.grid_runs_utils import get_axes_from_grid_config
import numpy as np

from theory.approximate_scaling_laws import (
    calculate_scaling_coefficient, calculate_scaling_bias,
    evaluate_theoretical_variance, evaluate_theoretical_accuracy)
from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png, plot_accuracy_grid, _load_and_aggregate_accuracy_grid


def infer_task_and_variant(run_config: dict) -> tuple[str, str]:
    task_type = 'AR' if run_config['dataset']['kwargs'].get('reduce_to_AR', False) else 'MQAR'
    model_variant = 'full' if run_config['model']['class'] == 'mamba_ssm' else 'linear'
    return task_type, model_variant


def infer_model_type(run_config: dict) -> str:
    """'designed' (weights set analytically, forward-only) vs 'trained'."""
    # full-Mamba (mamba_ssm) configs carry no model.kwargs at all - never designed
    designed = run_config['model'].get('kwargs', {}).get('set_designed_mqar_weights', False)
    return 'designed' if designed else 'trained'


def infer_grid_type(grid_axes) -> str:
    """Grid type from the bundle's own axes, e.g. 'D_N', 'N_Lambda', 'N_M'."""
    return f'{grid_axes.x.name}_{grid_axes.y.name}'


def approximate_theory_grid(grid_axes, grid_constants, run_config):
    """The paper's 'theoretical' D-N panel: closed-form approximation
    sigma^2 = a*N_f/(ND) + 1/D, accuracy = Phi(1/sigma - m_V), masked at N > D.
    Exactly reproduces the notebook computation (figures__single_layer.ipynb)."""
    task_type, model_variant = infer_task_and_variant(run_config)
    constants = grid_constants.as_dict()
    a, a_label = calculate_scaling_coefficient(task_type, 'trained', model_variant)
    b, _ = calculate_scaling_bias(constants['V'])
    nn, dd = np.meshgrid(grid_axes.y.axis, grid_axes.x.axis)
    _, sigma, _ = evaluate_theoretical_variance(
        D=dd, N_eff=nn, N_f=constants['N_facts'], a=a, a_label=a_label,
        approximate_large_N_f=False)
    y_fit, _ = evaluate_theoretical_accuracy(sigma=sigma, b=b)
    y_fit[nn > dd] = np.nan
    return y_fit


def load_bundle(run_dir: Path, split: str, seed_aggregation: str):
    run_config = json5.loads((run_dir / 'run_config.json5').read_text())
    grid_axes, grid_constants = get_axes_from_grid_config(run_config['grid'])
    accuracy_grid = _load_and_aggregate_accuracy_grid(
        run_dir / f'accuracy_grid__{split}.npy', seed_aggregation=seed_aggregation)
    return accuracy_grid, grid_axes, grid_constants, run_config


def main():
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    defaults = config.get('defaults', {})



    for name, entry in config['grids'].items():

        if entry is None:
            continue

        settings = {**defaults, **entry}

        # opt out of the grid PNG per entry with grid_figure: false (e.g. bundles used
        # only by other figure scripts); the curves script has its own `curve_figure` flag
        if not settings.get('grid_figure', True):
            continue

        run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']

        # font sizes matching the paper figure notebooks: D-N grids use 26/16
        # (figures__single_layer.ipynb grid section), multi-layer/head use 18/14
        configure_matplotlib(
            dpi=300,
            ax_labelsize=settings.get('ax_labelsize', 26),
            tick_labelsize=settings.get('tick_labelsize', 16),
        )

        if not run_dir.is_dir():
            print(f"SKIP {name}: run dir not found: {run_dir}")
            continue

        split = settings.get('split', 'val')
        if not (run_dir / f'accuracy_grid__{split}.npy').is_file():
            print(f"SKIP {name}: no accuracy_grid__{split}.npy yet (run just started?)")
            continue
        seed_aggregation = settings.get('seed_aggregation', 'max')
        accuracy_grid, grid_axes, grid_constants, run_config = load_bundle(run_dir, split, seed_aggregation)

        # derived approximate-theory panel (uses only the bundle's axes/constants)
        if settings.get('theory_approx', False):
            accuracy_grid = approximate_theory_grid(grid_axes, grid_constants, run_config)

        # paper style: math-mode (italic) axis labels, e.g. $D$ / $N$ / $\Lambda$
        MATH_LABELS = {'Lambda': r'$\Lambda$'}
        x_label = y_label = None
        if settings.get('math_labels', True):
            x_label = MATH_LABELS.get(grid_axes.x.name, f'${grid_axes.x.name}$')
            y_label = MATH_LABELS.get(grid_axes.y.name, f'${grid_axes.y.name}$')

        plot_accuracy_grid(
            accuracy_grid=accuracy_grid,
            grid_axes=grid_axes,
            grid_constants=grid_constants,
            run_config=run_config,
            split=split,
            seed_aggregation=seed_aggregation,
            show_title=settings.get('show_title', False),
            show_text=settings.get('show_text', False),
            show_cbar=settings.get('show_cbar', False),
            figsize=tuple(settings.get('figsize', [7, 5])),
            cmap=settings.get('cmap', 'inferno'),
            x_label=x_label,
            y_label=y_label,
            ticks_type=settings.get('ticks_type', 'quarter'),
            make_square=settings.get('make_square', True),
        )

        fig = plt.gcf()
        save_dir = figures_dir / settings.get('grid_subdir', '')  # '' -> figures/ (paper body)
        save_figure_to_png(fig, save_dir=save_dir, save_name=name, verbose=False)
        plt.close(fig)
        print(f"loaded: {settings.get('root', 'saved_runs_data')}/{settings['run']}")
        print(f"saved: {save_dir.relative_to(figures_dir.parent)}/{name}.png")
        print()


if __name__ == "__main__":
    main()
