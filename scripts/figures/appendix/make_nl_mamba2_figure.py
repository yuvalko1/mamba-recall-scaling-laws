"""
Zero-shot associative recall of the pretrained Mamba2 family vs. the number of facts (appendix).

Renders figures/appendix/nl_zero_shot_mamba2.png from the nl_zero_shot_mamba2 bundle: one curve per
model (Mamba2-370M ... 2.7B), accuracy averaged over the seeds (one random context per seed) as a
function of N_f, the number of key-value pairs in context. Seed-to-seed std is printed, not drawn.

The bundle is read via the `nl_zero_shot_mamba2` section of scripts/figures/config.json5
(grid shape: (n_seeds, n_models, n_N_facts); models and N_facts come from the archived run_config).

Usage (from anywhere):
    python scripts/figures/appendix/make_nl_mamba2_figure.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown
from matplotlib import pyplot as plt
import numpy as np

from experiments.nl_zero_shot_mamba2 import SPLIT, _parse_axis
from utils.common import figure_scripts_dir, figures_dir
from utils.plot import configure_matplotlib, save_figure_to_png

ENTRY = 'nl_zero_shot_mamba2'


def load_nl_zero_shot_bundle(run_dir: Path):
    run_config = json5.loads((run_dir / 'run_config.json5').read_text())
    accuracy_grid = np.load(run_dir / f'accuracy_grid__{SPLIT}.npy')  # (n_seeds, n_models, n_N_facts)
    n_facts_axis = _parse_axis(run_config['facts']['N_facts'])
    model_names = [m['name'] for m in run_config['models']]
    assert accuracy_grid.shape[1:] == (len(model_names), len(n_facts_axis)), accuracy_grid.shape
    return accuracy_grid, model_names, n_facts_axis, run_config


def main():
    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    settings = config[ENTRY]
    run_dir = PROJECT_DIR / settings.get('root', 'saved_runs_data') / settings['run']
    if not run_dir.is_dir():
        print(f"SKIP {ENTRY}: run dir not found: {run_dir}")
        return

    accuracy_grid, model_names, n_facts_axis, run_config = load_nl_zero_shot_bundle(run_dir)
    mean = accuracy_grid.mean(axis=0)
    std = accuracy_grid.std(axis=0)
    for k, name in enumerate(model_names):
        print(f"{name}: " + ' '.join(f"{int(n)}:{mean[k, j]:.2f}" for j, n in enumerate(n_facts_axis))
              + f"   (mean seed std {std[k].mean():.3f}, {accuracy_grid.shape[0]} seeds)")

    configure_matplotlib(dpi=300, ax_labelsize=settings.get('ax_labelsize', 18),
                         tick_labelsize=settings.get('tick_labelsize', 14))
    fig, ax = plt.subplots()  # the paper's curve-figure size (matplotlib default)
    for k, name in enumerate(model_names):
        ax.plot(n_facts_axis, mean[k], marker='o', ms=4, lw=2, label=name)
    ax.set_xlabel(r'$N_f$')
    ax.set_ylabel(r'${\text{Accuracy}}$')
    ax.set_xlim(0, int(n_facts_axis.max()) + 10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=12)

    save_figure_to_png(fig, save_dir=figures_dir / 'appendix', save_name=ENTRY)
    plt.close(fig)


if __name__ == "__main__":
    main()
