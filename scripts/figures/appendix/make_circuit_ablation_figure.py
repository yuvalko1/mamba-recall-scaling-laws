"""
Circuit-minimality ablation bars (the paper's appendix ablation table as a figure).

Loads the six circuit_minimality_ablation__{base,A,B,C,D,E} bundles from
saved_runs_data/ (each a single V=128/L=64/Nf=16/D=64/N=16 cell trained with 5 seeds)
and plots mean accuracy per row with a std error bar - the table's 'MQAR Accuracy'
column (seed-averaged), as a simple bar plot labeled Base, A, ..., E.

Output: figures/appendix/circuit_minimality_ablation.png

Usage (from anywhere):
    python scripts/figures/appendix/make_circuit_ablation_figure.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np

from utils.common import figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png

ROWS = ['base', 'A', 'B', 'C', 'D', 'E']


def main():
    configure_matplotlib(dpi=300, ax_labelsize=18, tick_labelsize=14)

    labels, means, stds = [], [], []
    for row in ROWS:
        fam_dir = PROJECT_DIR / 'saved_runs_data' / f'circuit_minimality_ablation__{row}'
        runs = sorted(fam_dir.glob('mamba*'), key=lambda p: p.stat().st_mtime)
        if not runs:
            print(f"SKIP {row}: no bundle in {fam_dir}")
            continue
        grid_file = runs[-1] / 'accuracy_grid__val.npy'
        if not grid_file.is_file():
            print(f"SKIP {row}: no accuracy grid yet")
            continue
        values = np.load(grid_file).ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            print(f"SKIP {row}: no finished seeds yet")
            continue
        labels.append('Base' if row == 'base' else row)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
        print(f"{row}: {np.mean(values):.2f} +- {np.std(values):.2f}  ({values.size} seeds)")

    if not labels:
        print("nothing to plot yet")
        return

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4, color='tab:blue')
    plt.xticks(x, labels)
    plt.ylabel(r'$\text{Accuracy}$')
    plt.ylim([0, 1.05])

    fig = plt.gcf()
    save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name='circuit_minimality_ablation')
    plt.close(fig)


if __name__ == "__main__":
    main()
