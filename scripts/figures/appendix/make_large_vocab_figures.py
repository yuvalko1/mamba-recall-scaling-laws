"""
Large-vocabulary / long-context figures (the V/L sweep).

Renders the appendix graphs from the MQAR__V_L__linear_trained bundle
(V up to 50k x L in {1000,2000,3000,4000}, scale_Nf_with_L, D=150, N=75):
  - <name>__acc_vs_V.png:          per-L accuracy vs V (log-x)
  - <name>__phi_inv_acc_vs_mV.png: per-L Phi^-1(accuracy) vs m_V (~sqrt(2 log V)) with
    per-L linear fits (theory slope: 1); fitted slopes are printed per L

The bundle is read via the MQAR__V_L__linear_trained entry of the `grids` section in
scripts/figures/config.json5.

Usage (from anywhere):
    python scripts/figures/appendix/make_large_vocab_figures.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts' / 'figures'))  # for make_grid_figures

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

from theory.approximate_scaling_laws import m_V
from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png
from make_grid_figures import load_bundle

ENTRY = 'MQAR__V_L__linear_trained'


def main():
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    settings = {**config.get('defaults', {}), **config['grids'][ENTRY]}
    run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']

    split = settings.get('split', 'val')
    seed_aggregation = settings.get('seed_aggregation', 'max')
    accuracy_grid, grid_axes, grid_constants, run_config = load_bundle(run_dir, split, seed_aggregation)
    assert grid_axes.x.name == 'V' and grid_axes.y.name == 'L'

    configure_matplotlib(dpi=300, ax_labelsize=18, tick_labelsize=14)

    V_axis = grid_axes.x.axis

    fig1, ax1 = plt.subplots()  # accuracy vs V (log-x)
    fig0, ax0 = plt.subplots()  # Phi^-1(accuracy) vs m_V(V), with linear fits
    fit_label = 'Linear Fit'  # single legend entry shared by all per-L fit lines

    for i, L in enumerate(grid_axes.y.axis):

        f = m_V  # the theory's vocabulary bias scale, ~sqrt(2 log V)

        x = f(V_axis)
        y = accuracy_grid[:, i]
        z = norm.ppf(y)

        mask = np.isfinite(x) & np.isfinite(z)
        if mask.sum() < 2:
            print(f"L={int(L)}: only {mask.sum()} finished cells - skipping its fit")
            a = None
        else:
            v_fit = np.linspace(np.min(V_axis), np.max(V_axis), 100)
            x_fit = f(v_fit)
            a, b = np.polyfit(x[mask], z[mask], 1)
            print(f"L={int(L)}: slope = {a:.3f}  (intercept {b:.3f}, {mask.sum()} cells)")
            ax0.plot(x_fit, a * x_fit + b, color='gray', label=fit_label)
            fit_label = None

        ax0.scatter(x, z, label=f"L={int(L)}")
        ax1.scatter(V_axis, y, label=f"L={int(L)}")

    ax1.set_xlabel(r'$V$')
    ax1.set_ylabel(r'${\text{Accuracy}}$')
    ax1.legend(loc='lower left')
    ax1.set_xscale('log')

    ax0.set_xlabel(r'$m_V \approx \sqrt{2\log{V}}$')
    ax0.set_ylabel(r'$\Phi^{-1}({\text{Accuracy}})$')
    # two-column legend, filled column-major: rows (1000 | 3000), (2000 | 4000),
    # with the shared fit entry on its own bottom row
    handles, labels = ax0.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ['L=1000', 'L=2000', 'Linear Fit', 'L=3000', 'L=4000']
    if set(order) == set(labels):
        handles, labels = [by_label[l] for l in order], order
    ax0.legend(handles, labels, ncol=2, loc='upper right', columnspacing=1.0, fontsize=12, handletextpad=0.5)

    save_figure_to_png(fig1, save_dir=figures_dir / 'appendix', save_name=f'{ENTRY}__acc_vs_V')
    save_figure_to_png(fig0, save_dir=figures_dir / 'appendix', save_name=f'{ENTRY}__phi_inv_acc_vs_mV')
    plt.close(fig1)
    plt.close(fig0)


if __name__ == "__main__":
    main()
