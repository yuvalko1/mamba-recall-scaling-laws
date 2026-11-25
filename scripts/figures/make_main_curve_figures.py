"""
Main-figure variant of the scaling-law collapse curves (the bottom row of the paper's Fig. 1).

Same data and math as make_scaling_curve_figures.py, different layout: the axes box is placed at
the same horizontal fractions as the tight-cropped D-N grid PNGs (frame at 16.4% .. 97.3% of the
image width) so that grid and curve panels set at the same \\includegraphics width align exactly;
fonts are sized for a panel at ~0.31\\textwidth; the legend sits lower-right; the theory line of the
full model is labelled 'Theory*' (its coefficient a_full is set heuristically, see the paper).
Saved without tight cropping, as PNG and as vector PDF (the paper includes the PDF), under
<root>/figures/<name>__acc_curve__sigma_inv__main.{png,pdf} for the entries in MAIN_ENTRIES.

Usage (from anywhere):
    python scripts/figures/make_main_curve_figures.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts' / 'figures'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np

from experiments.scaling_laws_utils import flatten_D_N_grid_data
from theory.approximate_scaling_laws import (
    calculate_scaling_coefficient, calculate_scaling_bias,
    evaluate_theoretical_variance, evaluate_theoretical_accuracy)
from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png
from make_grid_figures import load_bundle, infer_task_and_variant, infer_model_type, infer_grid_type

# the three regime-(i) panels of Fig. 1 (AR linear, MQAR linear, MQAR full)
MAIN_ENTRIES = [
    'AR__D_N__linear_trained__regime_2',
    'MQAR__D_N__linear_trained__regime_2',
    'MQAR__D_N__full_trained__regime_2',
]
SUFFIX = '__acc_curve__sigma_inv__main'

DPI = 600                       # thin theory line and small text stay crisp in the PNG copy
FIGSIZE = (6.4, 5.8)
AXES_LEFT, AXES_RIGHT = 0.164, 0.973   # = frame fractions measured on the tight-cropped grid PNGs
AXES_BOTTOM, AXES_TOP = 0.29, 0.985    # bottom margin holds the sqrt-fraction x label
AX_LABELSIZE, X_LABELSIZE, TICK_LABELSIZE, LEGEND_FONTSIZE = 28, 36, 19, 20
THEORY_LINE = dict(color='0.2', alpha=0.8, lw=2.5)
SCATTER_SIZE = 6


def main(save_pdf=True):
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    defaults = config.get('defaults', {})

    configure_matplotlib(dpi=DPI, ax_labelsize=AX_LABELSIZE, tick_labelsize=TICK_LABELSIZE)
    matplotlib.rcParams.update({
        'legend.fontsize': LEGEND_FONTSIZE,
        # lighter frame, closer to the D-N grid panels next to them
        'axes.edgecolor': '0.35', 'axes.linewidth': 0.6,
        'xtick.color': '0.35', 'ytick.color': '0.35',
        'xtick.labelcolor': 'black', 'ytick.labelcolor': 'black',
        'legend.edgecolor': '0.6',
    })

    for name in MAIN_ENTRIES:
        settings = {**defaults, **config['grids'][name]}
        run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']
        if not run_dir.is_dir():
            print(f"SKIP {name}: run dir not found: {run_dir}")
            continue
        split = settings.get('split', 'val')
        accuracy_grid, grid_axes, grid_constants, run_config = load_bundle(
            run_dir, split, settings.get('seed_aggregation', 'max'))
        task_type, model_variant = infer_task_and_variant(run_config)
        model_type = settings.get('model_type', infer_model_type(run_config))
        assert settings.get('grid_type', infer_grid_type(grid_axes)) == 'D_N', name
        constants = grid_constants.as_dict()
        V, N_f = constants['V'], constants['N_facts']

        a, a_label = calculate_scaling_coefficient(task_type, model_type, model_variant)
        b, _ = calculate_scaling_bias(V)
        D, N, accuracy = flatten_D_N_grid_data(accuracy_grid, grid_axes, grid_constants)
        _, sigma_data, sigma_inv_label = evaluate_theoretical_variance(
            D=D, N_eff=N, N_f=N_f, a=a, a_label=a_label, N_eff_label=r'N', approximate_large_N_f=False)
        x_data = 1 / sigma_data
        finite = np.isfinite(x_data) & np.isfinite(accuracy)
        x_axis = np.linspace(np.nanmin(x_data[finite]), np.nanmax(x_data[finite]), 100)
        y_theory, _ = evaluate_theoretical_accuracy(sigma=1 / x_axis, b=b)

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes([AXES_LEFT, AXES_BOTTOM, AXES_RIGHT - AXES_LEFT, AXES_TOP - AXES_BOTTOM])
        ax.scatter(x_data, accuracy, s=SCATTER_SIZE, label=model_type.title())
        theory_label = 'Theory*' if model_variant == 'full' else 'Theory'
        ax.plot(x_axis, y_theory, label=theory_label, **THEORY_LINE)
        ax.set_xticks(np.arange(1, int(np.floor(ax.get_xlim()[1])) + 1))  # integer ticks up to the axis limit
        ax.set_xlabel(sigma_inv_label, fontsize=X_LABELSIZE)
        ax.set_ylabel(r'$\text{Accuracy}$')
        ax.legend(loc='lower right')

        save_name = f'{name}{SUFFIX}'
        save_figure_to_png(fig, save_dir=figures_dir, save_name=save_name, dpi=DPI, verbose=False,
                           tight=False, bbox_inches=None)
        if save_pdf:  # the dry tests redirect the PNG only, so they pass save_pdf=False
            fig.savefig(figures_dir / f'{save_name}.pdf', facecolor='white', bbox_inches=None)
        plt.close(fig)
        print(f"saved: figures/{save_name}.png" + (" + .pdf" if save_pdf else ""))


if __name__ == "__main__":
    main()
