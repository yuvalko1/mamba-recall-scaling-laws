"""
Mech-interp figures: paper Fig. 3 (decoded hidden state) + the reversing appendix
figures (attention maps, convolution operators, invariant operators), repeated for
each configured grid cell of the mech_interp run.

Driven by the `mech_interp` section of scripts/figures/config.json5: the run
(relative to saved_runs_data/, whose committed bundle holds the run_config,
accuracy grids, and the configured cells' best-model checkpoints; set root:
'results' for a local run), a `cells` list of {D, N, mqar_seed}, and plot kwargs. All panels carry a __D<D>_N<N> name tag. The FIRST cell is the
paper-body one: its hidden-state panels land in figures/ (paper Fig 3); everything
else - the body cell's reversing panels and all panels of the other cells - lands
in figures/appendix/.

Usage (from anywhere):
    python scripts/figures/make_mech_interp_figures.py [--verbose]

--verbose restores the debug prints (recorder messages, per-run accuracy, shapes).
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import json5
import matplotlib

matplotlib.use('Agg')  # headless; figures are saved, never shown

from utils.common import set_seed, figure_scripts_dir, figures_dir
from utils.load import load_model_from_saved_run
from utils.plot import configure_matplotlib
from experiments.mech_interp_utils import (
    run_model_on_single_sequence,
    compute_hidden_states, plot_hidden_states,
    compute_attention_maps, plot_attention_map,
    compute_model_G_matrices, plot_G_matrices,
    compute_conv1d_effective_matrices, plot_conv1d_effective_matrices)


def make_cell_figures(run_dir: Path, cell: dict, plot_kwargs: dict, is_body: bool, verbose: bool = False):
    """The full mech-interp figure set for one grid cell (D, N) and one input-sequence
    seed. Every panel carries a __D<D>_N<N> tag in its name; the body cell's
    hidden-state panels (paper Fig 3) go to figures/, everything else to
    figures/appendix/."""
    D, N, seed = cell['D'], cell['N'], cell['seed']

    cell_tag = f'__D{D}_N{N}'
    appendix_dir = figures_dir / 'appendix'
    hidden_state_dir = figures_dir if is_body else appendix_dir

    # an absolute run dir bypasses the loader's hardcoded results/ root
    model, dataloaders, run_config, dims = load_model_from_saved_run(str(run_dir), x=D, y=N)
    model_class = run_config['model']['class']

    # Fig. 3: decoded hidden state (recurrent-mode recording)
    set_seed(seed)
    x_ids, _, _, records = run_model_on_single_sequence(model, dataloaders, ssm_mode='recurrent', verbose=verbose)
    H_ideal, H_noisy, H_noisy_projected = compute_hidden_states(x_ids, model, model_class, records, dims,
                                                                verbose=verbose)
    plot_hidden_states(
        H_ideal, H_noisy, H_noisy_projected, dims,
        **plot_kwargs,
        script_name=f'mech_interp{cell_tag}__hidden_state',
        save_dir=hidden_state_dir,
    )

    # reversing appendix: attention maps (attention-mode recording)
    set_seed(seed)
    x_ids, _, _, records = run_model_on_single_sequence(model, dataloaders, ssm_mode='attention', verbose=verbose)
    alpha_ideal, alpha_noisy = compute_attention_maps(x_ids, records, dims.V)
    plot_attention_map(alpha_ideal, **plot_kwargs, name='alpha_ideal',
                       script_name=f'mech_interp{cell_tag}__attention_maps', save_dir=appendix_dir)
    plot_attention_map(alpha_noisy, **plot_kwargs, name='alpha_noisy',
                       script_name=f'mech_interp{cell_tag}__attention_maps', save_dir=appendix_dir)

    # reversing appendix: effective convolution operators (reuses the attention-mode x_ids)
    G_kq, G_vv, G_E_tilde, G_E = compute_model_G_matrices(model=model, model_class=model_class)
    x_Gvv_xi, xi_Gkq_xi = compute_conv1d_effective_matrices(G_vv, G_kq, x_ids, dims)
    plot_conv1d_effective_matrices(
        x_Gvv_xi, xi_Gkq_xi, dims,
        **plot_kwargs,
        script_name=f'mech_interp{cell_tag}__convolution',
        save_dir=appendix_dir,
    )

    # reversing appendix: invariant operators (G matrices)
    # viridis: its non-black floor keeps the near-zero quadrant structure visible
    plot_G_matrices(
        G_kq, G_vv, dims.V,
        **{**plot_kwargs, 'cmap': 'viridis'},
        script_name=f'mech_interp{cell_tag}__invariant_operators',
        save_dir=appendix_dir,
        figsize=(8, 8),
        cp_line_color='gray',
        kv_line_color=None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render the mech-interp figure set per configured cell.')
    parser.add_argument('--verbose', action='store_true',
                        help='debug prints: recorder messages, per-run accuracy, tensor shapes')
    args = parser.parse_args(argv)  # None -> sys.argv; pass [] when calling in-process

    config = json5.loads((figure_scripts_dir / 'config.json5').read_text())
    settings = config['mech_interp']
    plot_kwargs = settings['plot_kwargs']

    # compact fonts: these dense panels render at ~1/3 text width
    configure_matplotlib(ax_labelsize=16, tick_labelsize=12)

    # like the `grids` entries: the committed bundle under saved_runs_data/ by
    # default (run_config, accuracy grids, and the configured cells' best models);
    # set root: 'results' to use a local, non-archived run
    root = settings.get('root', 'saved_runs_data')
    run_dir = PROJECT_DIR / root / settings['run']

    for i, cell in enumerate(settings['cells']):
        print(f"\nloaded: {root}/{settings['run']}  "
              f"(D={cell['D']}, N={cell['N']}, mqar_seed={cell['seed']})")
        make_cell_figures(run_dir, cell, plot_kwargs, is_body=(i == 0), verbose=args.verbose)


if __name__ == "__main__":
    main()
