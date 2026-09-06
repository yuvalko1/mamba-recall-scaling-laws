"""
Scaling-law collapse curves (the paper's `acc_curve__sigma_inv` figures).

Every entry of `grids` in scripts/figures/config.json5 that can be collapsed is rendered
into <root>/figures/<name>__acc_curve__sigma_inv.png: measured accuracy of every grid cell,
scattered against the theoretical inverse-noise scale 1/sigma, with the analytical
Phi(1/sigma + b) curve overlaid. Grid type (D_N / N_Lambda), task (AR/MQAR), model variant
(linear/full) and model type (trained/designed - which picks the theoretical coefficient a)
are all inferred from the bundle's run_config; entries with `curve_figure: false`, derived-theory
panels (`theory_approx`), and unsupported grid types (e.g. N_M) are skipped.
All the math is reused from src/ (theory + experiments modules); this script only
orchestrates data loading and plotting.

Usage (from anywhere):
    python scripts/figures/make_scaling_curve_figures.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.colors as mcolors

from experiments.scaling_laws_utils import flatten_D_N_grid_data, flatten_N_Lambda_grid_data
from theory.approximate_scaling_laws import (
    calculate_scaling_coefficient, calculate_scaling_bias, calculate_N_eff,
    evaluate_theoretical_variance, evaluate_theoretical_accuracy)
from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png
from make_grid_figures import load_bundle, infer_task_and_variant, infer_model_type, infer_grid_type


def main():
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    defaults = config.get('defaults', {})

    # curve font sizes matching the paper notebook (figures__single_layer.ipynb, curves section)
    configure_matplotlib(dpi=300, ax_labelsize=18, tick_labelsize=14)

    for name, entry in config['grids'].items():

        if entry is None:
            continue

        settings = {**defaults, **entry}

        # every grid can in principle be collapsed; opt out per entry with curve: false.
        # derived-theory panels have no measured accuracies to scatter - always skipped.
        if not settings.get('curve_figure', True) or settings.get('theory_approx', False):
            continue

        run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']

        if not run_dir.is_dir():
            print(f"SKIP {name}: run dir not found: {run_dir}")
            continue

        split = settings.get('split', 'val')
        if not (run_dir / f'accuracy_grid__{split}.npy').is_file():
            print(f"SKIP {name}: no accuracy_grid__{split}.npy yet (run just started?)")
            continue
        seed_aggregation = settings.get('seed_aggregation', 'max')
        accuracy_grid, grid_axes, grid_constants, run_config = load_bundle(run_dir, split, seed_aggregation)

        task_type, model_variant = infer_task_and_variant(run_config)
        model_type = settings.get('model_type', infer_model_type(run_config))
        grid_type = settings.get('grid_type', infer_grid_type(grid_axes))
        if grid_type not in ('D_N', 'N_Lambda'):
            continue  # no collapse mapping for this grid type
        constants = grid_constants.as_dict()
        V, N_f = constants['V'], constants['N_facts']

        a, a_label = calculate_scaling_coefficient(task_type, model_type, model_variant)
        b, _ = calculate_scaling_bias(V)

        # flatten the grid to per-cell points and map onto the theory's inverse-noise axis
        Lambda = None
        if grid_type == 'D_N':
            D, N, accuracy = flatten_D_N_grid_data(accuracy_grid, grid_axes, grid_constants)
            N_eff, N_eff_label = N, r'N'
        elif grid_type == 'N_Lambda':
            N, Lambda, accuracy = flatten_N_Lambda_grid_data(accuracy_grid, grid_axes, constants)
            D = constants['D']
            N_eff, N_eff_label = calculate_N_eff(N=N, Lambda=Lambda)
        else:
            raise ValueError(grid_type)

        _, sigma_data, sigma_inv_label = evaluate_theoretical_variance(
            D=D, N_eff=N_eff, N_f=N_f, a=a, a_label=a_label,
            N_eff_label=N_eff_label, approximate_large_N_f=False)
        x_data = 1 / sigma_data

        finite = np.isfinite(x_data) & np.isfinite(accuracy)
        x_axis = np.linspace(np.nanmin(x_data[finite]), np.nanmax(x_data[finite]), 100)
        y_theory, _ = evaluate_theoretical_accuracy(sigma=1 / x_axis, b=b)

        plt.figure()
        if Lambda is None:
            plt.scatter(x_data, accuracy, s=5, label=f'{model_type.title()}')
        else:
            # Lambda-colored scatter with a discrete colorbar, as in figures__multi_layer.ipynb
            cmap = plt.get_cmap('rainbow', int(Lambda.max()) + 1 - int(Lambda.min()))
            bounds = np.arange(Lambda.min() - 0.5, Lambda.max() + 1.5, 1)
            ticks = np.arange(Lambda.min(), Lambda.max() + 1)
            im = plt.scatter(x_data, accuracy, c=Lambda, s=25, cmap=cmap,
                             norm=mcolors.BoundaryNorm(bounds, cmap.N),
                             label=f'{model_type.title()} ({task_type})')
            cbar = plt.colorbar(im, ticks=ticks, boundaries=bounds)
            cbar.set_label(r'$\Lambda$', rotation=0, labelpad=15)
            plt.ylim([-0.05, 1.05])
        plt.plot(x_axis, y_theory, 'k', alpha=0.75, label='Theory')
        plt.xlabel(sigma_inv_label)
        plt.ylabel(r'$\text{Accuracy}$')
        plt.legend()

        fig = plt.gcf()
        save_name = f'{name}__acc_curve__sigma_inv'
        save_dir = figures_dir / settings.get('curve_subdir', '')  # '' -> figures/ (paper body)
        save_figure_to_png(fig, save_dir=save_dir, save_name=save_name, verbose=False)
        plt.close(fig)
        print(f"loaded: {settings.get('root', 'saved_runs_data')}/{settings['run']}")
        print(f"saved: {save_dir.relative_to(figures_dir.parent)}/{save_name}.png")
        print()


if __name__ == "__main__":
    main()
