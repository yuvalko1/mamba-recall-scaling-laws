"""
Seed-reliability analysis (appendix).

From the 5-seed MQAR__D_N__full_trained__regime_0 bundle:
  - <fam>__seed_grid__<i>.png:        the D-N accuracy grid of seed i alone (failed seeds
                                      appear as dark holes in the converged region)
  - <fam>__best_seed_grid.png:        the best-of-seeds D-N accuracy grid, same format
  - <fam>__convergence_rate_grid.png: per-cell fraction of seeds reaching accuracy >= 0.1
                                      (viridis)
  - <fam>__seed_reliability_rates.png: per accuracy-regime bin (binned by best-of-seeds
                                      accuracy), the per-seed convergence rate as bars
                                      (note the bins condition on a converged cell, so the
                                      rate has a mechanical 1/5 floor)

'Converged' = accuracy >= 0.1. The rates are also printed as a table.

Usage (from anywhere):
    python scripts/figures/appendix/make_seeds_analysis_figures.py
"""

import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np

from experiments.grid_runs_utils import get_axes_from_grid_config
from utils.common import figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png, plot_accuracy_grid

FAM = 'MQAR__D_N__full_trained__regime_0'
CONVERGED = 0.1
# quarter regime bins; the first starts at the convergence threshold, since cells with
# best-of-seeds < 0.1 have no converged seed by definition
BINS = [(0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def main():
    fam_dir = PROJECT_DIR / 'saved_runs_data' / FAM
    run_dir = sorted(fam_dir.glob('mamba*'), key=lambda p: p.stat().st_mtime)[-1]
    run_config = json5.loads((run_dir / 'run_config.json5').read_text())
    grid_axes, grid_constants = get_axes_from_grid_config(run_config['grid'])
    g = np.load(run_dir / 'accuracy_grid__val.npy')  # (seeds, D, N), NaN above capacity
    n_seeds = g.shape[0]
    print(f"bundle: {run_dir.name} ({n_seeds} seeds)")

    configure_matplotlib(dpi=300, ax_labelsize=18, tick_labelsize=14)

    # 1) per-seed grids + the best-of-seeds grid, paper D-N format (no colorbar:
    #    the paper row reuses the shared standalone colorbar strip)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        best_grid = np.nanmax(g, axis=0)
    panels = [(f'seed_grid__{i}', g[i], f'seed{i}') for i in range(n_seeds)]
    panels.append(('best_seed_grid', best_grid, 'max'))
    for name, grid, agg in panels:
        plot_accuracy_grid(
            accuracy_grid=grid, grid_axes=grid_axes, grid_constants=grid_constants,
            run_config=run_config, split='val', seed_aggregation=agg,
            show_title=False, show_text=False, show_cbar=False,
            figsize=(7, 5), cmap='inferno', x_label='$D$', y_label='$N$',
            ticks_type='quarter', make_square=True)
        fig = plt.gcf()
        save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name=f'{FAM}__{name}')
        plt.close(fig)

    # 2) per-cell convergence rate (fraction of seeds with acc >= CONVERGED), viridis
    tried = ~np.isnan(g).all(axis=0)
    rate = np.nanmean(g >= CONVERGED, axis=0).astype(float)
    rate[~tried] = np.nan
    plot_accuracy_grid(
        accuracy_grid=rate, grid_axes=grid_axes, grid_constants=grid_constants,
        run_config=run_config, split='val', seed_aggregation='rate',
        show_title=False, show_text=False, show_cbar=True,
        figsize=(7, 5), cmap='viridis', x_label='$D$', y_label='$N$',
        ticks_type='quarter', make_square=True)
    fig = plt.gcf()
    save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name=f'{FAM}__convergence_rate_grid')
    plt.close(fig)

    # 3) binned reliability rates: bin cells by best-of-seeds accuracy
    best = best_grid  # all-NaN slices (skipped N > D cells) already handled above
    labels, rate_1seed = [], []
    print("\n| Accuracy regime | per-seed convergence rate |")
    print("|---|---|")
    for lo, hi in BINS:
        cells = tried & (best > lo) & (best <= hi + 1e-12)
        if cells.sum() == 0:
            continue
        # per-seed convergence rate over all (cell, seed) pairs in the bin
        seeds_in_bin = g[:, cells]
        r1 = np.nanmean(seeds_in_bin >= CONVERGED)
        labels.append(f"({lo:g}, {hi:g}]")
        rate_1seed.append(float(r1))
        print(f"| ({lo:g}, {hi:g}] | {100*r1:.0f}% | ({int(cells.sum())} cells)")

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, rate_1seed, 0.55)
    plt.xticks(x, labels)
    plt.xlabel(r'$\text{Accuracy Regime (best of seeds)}$')
    plt.ylabel(r'$\text{Convergence Rate}$')
    plt.ylim([0, 1.05])
    fig = plt.gcf()
    save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name=f'{FAM}__seed_reliability_rates')
    plt.close(fig)


if __name__ == "__main__":
    main()
