"""
Repeated keys curves (appendix).

Renders accuracy vs N_k with one line per Lambda and the gray p = 1/N_k reference
of the single-layer guessing bound.

Reads the MQAR__Lambda_Nk__linear_trained entry of the `grids` section in
scripts/figures/config.json5; outputs land in figures/appendix/ as
<entry>__acc_vs_<axis>.png.

Usage (from anywhere):
    python scripts/figures/appendix/make_repeats_figures.py
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

from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png
from make_grid_figures import load_bundle

ENTRIES = {
    'MQAR__Lambda_Nk__linear_trained': dict(reference=True),   # gray p = 1/N_k guessing bound
}


def main():
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    defaults = config.get('defaults', {})

    configure_matplotlib(dpi=300, ax_labelsize=18, tick_labelsize=14)

    for name, opts in ENTRIES.items():
        entry = config['grids'].get(name)
        if entry is None:
            print(f"SKIP {name}: no config entry")
            continue
        settings = {**defaults, **entry}
        run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']
        split = settings.get('split', 'val')
        if not (run_dir / f'accuracy_grid__{split}.npy').is_file():
            print(f"SKIP {name}: no accuracy_grid__{split}.npy yet")
            continue

        accuracy_grid, grid_axes, grid_constants, run_config = load_bundle(
            run_dir, split, settings.get('seed_aggregation', 'max'))
        assert grid_axes.x.name == 'Lambda'
        rep_name = grid_axes.y.name  # 'N_k' or 'N_q'
        rep_axis = grid_axes.y.axis

        plt.figure()
        for i, lam in enumerate(grid_axes.x.axis):
            acc = accuracy_grid[i, :]
            plt.plot(rep_axis, acc, 'o-', label=rf'$\Lambda = {int(lam)}$')
        if opts['reference']:
            plt.plot(rep_axis, 1.0 / np.asarray(rep_axis, dtype=float), '*-',
                     color='gray', label=r'$p = \frac{1}{N_k}$')
        plt.xlabel(rf'${rep_name}$')
        plt.ylabel(r'$\text{Accuracy}$')
        plt.xticks(rep_axis)
        plt.ylim(0, 1.02)
        plt.legend()

        fig = plt.gcf()
        save_name = f"{name}__acc_vs_{rep_name.replace('_', '')}"
        save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name=save_name)
        plt.close(fig)
        print(f"saved: figures/appendix/{save_name}.png")


if __name__ == "__main__":
    main()
