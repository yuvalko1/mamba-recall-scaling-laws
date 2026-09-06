import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.plot import save_figure_to_png
from mamba_tiny.scans import compute_attention, compute_recurrence
from mqar import MqarDimensions
from mqar_zoology.associative_recall import IGNORED_TOKEN
from theory.simplified_linear_mamba import extract_model_weights, compute_G_matrices, compute_effective_operators
from utils import recorder
from utils.common import figures_dir
from utils.recorder import collect_model_recordings

import matplotlib.ticker as mticker


def _maybe_show():
    # show only on interactive backends (notebooks); Agg (figure scripts) just warns
    import matplotlib
    if 'agg' not in matplotlib.get_backend().lower():
        plt.show()



def run_model_on_single_sequence(model, dataloaders, ssm_mode='attention', split='val', device='cpu', print_ids=False,
                                 verbose=False):

    for batch in dataloaders[split]:
        break

    x_ids = batch.x_ids[0]


    x_ids = batch.x_ids[0].unsqueeze(0).to(device)
    y_true_ids = batch.y_true_ids[0].unsqueeze(0).to(device)

    model.eval()
    model.to(device)

    assert ssm_mode in ['recurrent', 'attention', 'selective_scan']

    model.layers[0].mixer.config['ssm_mode'] = ssm_mode
    recorder.VERBOSE = verbose  # quiet the recorder's informational prints too
    if verbose:
        print(f"setting SSM mode to '{ssm_mode}'\n")

    recorder.empty()

    recorder.enable_recording()

    # predict
    y_pred_logits = model(x_ids).logits  # (B, L, V)

    # collect predictions
    y_pred_ids = y_pred_logits.argmax(dim=2)
    y_correct = y_pred_ids.eq(y_true_ids)
    y_correct = y_correct.masked_select(y_true_ids.ne(IGNORED_TOKEN)).detach().cpu()

    y_response_ids = y_pred_ids.squeeze().detach().cpu().numpy()

    total_correct = int(y_correct.sum())

    accuracy = float(y_correct.to(float).mean())

    if print_ids:
        print(f"\n{x_ids = }")
        print("\ny_response_ids = \n", y_response_ids, "\n")
        print("\ny_response_gt = \n", y_true_ids, "\n")

    if verbose:
        print(f"\n{accuracy = }")
        print(f"{total_correct = } / {len(y_correct)}")

    # collect records
    records, records_order = collect_model_recordings(model, verbose=False)

    recorder.disable_recording()

    return x_ids, y_true_ids, y_pred_ids, records


def normalize(mat: np.ndarray, allow_flip: bool = True) -> np.ndarray:
    """
    Scale `mat` so that its largest entry becomes +1, and its smallest entry ≥ -1.
    If allow_flip and |min(mat)| > |max(mat)|, flip the sign of the matrix first.
    Pass allow_flip=False when the input's sign is already oriented against a
    reference (see compute_attention_maps) - the global-extreme heuristic here
    mis-orients when a positive noise outlier exceeds the signal magnitude.
    """
    max_val = mat.max()
    min_val = mat.min()

    # If the negative side dominates, flip everything
    if allow_flip and abs(min_val) > abs(max_val):
        mat = -mat
        max_val = -min_val  # after flip, this is the new maximum

    # Avoid division by zero
    if max_val == 0:
        return mat  # all zeros

    return mat / max_val


def compute_ideal_x_B_C(x_ids, V):

    # ideal weights

    I = torch.eye(V)
    Z = torch.zeros((V, V))


    # write/read/output (k/q/v) projectors
    S_B_ideal = torch.hstack([I, Z])  # (N, 2D)
    S_C_ideal = torch.hstack([Z, I])  # (N, 2D)

    # input dependent matrices

    E_oh = torch.eye(V)

    x_curr = E_oh[:, x_ids[0]]

    x_prev = torch.zeros_like(x_curr)
    x_prev[:, 1:] = x_curr[:, :-1]

    x_hat = torch.vstack([x_prev, x_curr])

    x_ssm_ideal = x_hat.T.unsqueeze(0)

    B_ideal = (S_B_ideal @ x_hat).T.unsqueeze(0)
    C_ideal = (S_C_ideal @ x_hat).T.unsqueeze(0)

    return x_ssm_ideal, B_ideal, C_ideal


def compute_ideal_attention_map(x_ids, V):

    x_ssm_ideal, B_ideal, C_ideal = compute_ideal_x_B_C(x_ids, V)

    alpha_ideal, y_ssm_ideal = (
        compute_attention(x=x_ssm_ideal, A=None, B=B_ideal, C=C_ideal))

    alpha_ideal = alpha_ideal[0]

    return alpha_ideal


def compute_attention_maps(x_ids, records, V, layer_idx=0, head_idx=0):

    caller_name = f'layers.{layer_idx}.heads.{head_idx}'

    alpha_ideal = compute_ideal_attention_map(x_ids, V)
    alpha_noisy = records[f'model.{caller_name}__attention_alpha'][0].cpu().numpy()

    # the circuit is sign-symmetric (flipping both B and C preserves the output),
    # so trained attention may come out negative at the recall positions;
    # orient it against the ideal reference (as make_positive_sign does for G)
    if float((alpha_noisy * np.asarray(alpha_ideal)).sum()) < 0:
        alpha_noisy = -alpha_noisy

    return alpha_ideal, alpha_noisy

def compute_ideal_hidden_state(x_ids, V):

    x_ssm_ideal, B_ideal, C_ideal = compute_ideal_x_B_C(x_ids, V)

    h_ideal, y_ssm_ideal = (
        compute_recurrence(x=x_ssm_ideal, A=None, B=B_ideal, C=C_ideal))

    h_ideal = h_ideal[0]

    return h_ideal


def compute_hidden_states(x_ids, model, model_class, records, dims: MqarDimensions, layer_idx=0, head_idx=0,
                          verbose=False):

    caller_name = f'layers.{layer_idx}.heads.{head_idx}'

    V = dims.V
    D = dims.D
    N_facts = dims.N_facts

    if verbose:
        print(f"{V=}, {D=}")

    max_t = int(N_facts * 2)  # length of context part

    Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out = compute_model_projection_matrices(model, model_class)

    if verbose:
        print(f"{Pi_q_in.shape=}, {Pi_v_out.shape=}")

    h_ideal = compute_ideal_hidden_state(x_ids, V)[max_t]
    h_noisy = records[f'model.{caller_name}__h_ssm'][0].cpu().numpy()[max_t]
    h_noisy_projected = Pi_v_out.numpy() @ h_noisy.T @ Pi_q_in.numpy()

    if verbose:
        print(f"{h_ideal.shape=}, {h_noisy.shape=}, {h_noisy_projected.shape=}")

    H_ideal = h_ideal[:, V:].T
    H_noisy = h_noisy[:, D:].T
    H_noisy_projected = h_noisy_projected[:, V:]

    if verbose:
        print(f"{H_ideal.shape=}, {H_noisy.shape=}, {H_noisy_projected.shape=}")

    return H_ideal, H_noisy, H_noisy_projected


def plot_hidden_states(
        H_ideal, H_noisy, H_noisy_projected, dims: MqarDimensions,
        cmap='inferno', vmin=0, vmax=1, no_title=True, no_cbar=True,
        save=False, script_name='', save_dir=None,
):

    V = dims.V

    add_dim_label = False

    x_labels = [r"$k$", r"$k'$", r"$k''$"]
    y_labels = [r"$v$", r"$v'$", r"$v''$"]

    # pixel-identical panels: the label strings differ in size ($v$ vs $v''$,
    # $k$ vs $k''$), so both tight_layout (axes placement) and bbox_inches='tight'
    # (final crop) would give each panel its own geometry. Instead every panel
    # gets (a) the SAME fixed axes rectangle - no tight_layout - and (b) two
    # invisible points pinning the crop's left/bottom edges at fixed axes
    # fractions just beyond the widest labels. All (V, V) panels then come out
    # pixel-identical and stack 1:1.
    AXES_RECT = [0.26, 0.22, 0.70, 0.70]  # figure fractions: left, bottom, w, h
    # per panel: label extents differ ($v$ vs $v''$, one- vs two-digit ticks)
    CROP_ANCHOR_LEFT = {'H_ideal': -0.1769, 'H_noisy': -0.17, 'H_projected': -0.1892}
    CROP_ANCHOR_BOTTOM = {'H_ideal': -0.11, 'H_noisy': -0.1552, 'H_projected': -0.12}
    titles = [r"$H$", r"$H'$", r"$H''$"]
    names = ['H_ideal', 'H_noisy', 'H_projected']

    H_matrices = [
        H_ideal,
        H_noisy,
        H_noisy_projected,
    ]

    for M, x_label, y_label, title, name in zip(H_matrices, x_labels, y_labels, titles, names):
        M = np.asarray(M)  # torch tensors upset matplotlib's array coercion
        dims = np.array(M.shape)

        # vmax=None must autoscale to the DISPLAYED data: the (V, V) matrices are
        # cropped below to the value x key quadrant, so a global max sitting in an
        # off-view quadrant must not set the color scale (it washes out the view)
        panel_vmax = vmax
        if panel_vmax is None and M.shape == (V, V):
            panel_vmax = np.asarray(M)[V // 2:, :V // 2].max()

        fig, ax = plt.subplots(figsize=(6, 6))  # bigger figure
        ax.set_position(AXES_RECT)  # fixed geometry (see comment above)

        im = ax.imshow(
            M, cmap=cmap, interpolation="none",
            vmin=vmin,
            vmax=panel_vmax,
            origin="lower",
            extent=(0, dims[1], 0, dims[0]),
            aspect=1,
        )

        if M.shape == (V, V):
            V_k = V // 2
            ax.set_xlim(0, V_k)  # key
            ax.set_ylim(V_k, V)  # value
            dims //= 2
        else:
            ax.set_xlim(0, dims[1])  # key
            ax.set_ylim(0, dims[0])  # value

        if add_dim_label:
            x_label = f"{x_label}\n$[V_k]$"
            y_label = f"{y_label}\n$[V_v]$"

        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel(y_label, rotation=0, labelpad=16)

        if not no_title:
            ax.set_title(title, pad=10)

        # Sharper ticks and frame for print
        ax.tick_params(axis="both", which="major", length=4, width=1.2)

        ax.xaxis.set_major_locator(mticker.MultipleLocator(dims[1] / 4))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(dims[0] / 4))

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # invisible crop anchors (see comment above the label lists)
        ax.plot([CROP_ANCHOR_LEFT[name], 0.5], [0.5, CROP_ANCHOR_BOTTOM[name]],
                transform=ax.transAxes, linestyle='none',
                marker='.', markersize=0.1, alpha=0, clip_on=False)

        if save:
            save_figure_to_png(fig, save_dir=(save_dir or figures_dir), save_name=f'{script_name}__{name}',
                               tight=False)  # fixed geometry; tight_layout would resize per-label
        _maybe_show()
        plt.close(fig)


def plot_attention_map(
        alpha,
        no_title=True,
        no_cbar=True,
        cmap='inferno',
        vmin=0, vmax=1,
        save=False, name='attention', script_name='', save_dir=None,
        figsize=(6, 6),
):

    # data
    t_axis = np.arange(0, alpha.shape[0])
    alpha = normalize(alpha, allow_flip=False)  # sign already oriented against the ideal

    # styling parameters

    # create figure & axis
    fig, ax = plt.subplots(figsize=figsize)

    # plot
    im = ax.pcolormesh(
        t_axis, t_axis,
        alpha.T,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )

    if not no_cbar:
        plt.colorbar(im, ax=ax)

    # title and labels
    if not no_title:
        ax.set_title(r"SSM Attention $\alpha$")

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\tau$')

    ax.set_aspect('equal')


    fig.tight_layout()
    if save:
        save_figure_to_png(fig, save_dir=(save_dir or figures_dir), save_name=f'{script_name}__{name}')
    _maybe_show()
    plt.close(fig)


def compute_model_projection_matrices(model, model_class):

    # extract trained model weights
    E, P_in, W, S_B, S_C, P_out = extract_model_weights(
        model=model, model_class=model_class,
        print_output_shapes=False,
    )

    # compute effective transformation matrices
    Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out = compute_effective_operators(E, P_in, W, S_B, S_C, P_out)

    return Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out


def compute_model_G_matrices(model, model_class, make_positive_sign=True):

    # compute effective transformation matrices
    Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out = compute_model_projection_matrices(model=model, model_class=model_class)
    G_kq, G_vv = compute_G_matrices(
        Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out)

    # # slice

    # slice
    V = G_kq.shape[1] // 2
    G_E_tilde = G_kq[:V, V:]  # prev→curr block
    G_E = G_vv[:, V:]  # curr→curr block

    # if required, invert signs
    if make_positive_sign:

        half_V = V // 2

        sign_G_E_tilde = np.sign(float(torch.diag(G_E_tilde)[:half_V].mean()))
        sign_G_E = np.sign(float(torch.diag(G_E)[half_V:].mean()))


        G_kq = sign_G_E_tilde * G_kq
        G_vv = sign_G_E * G_vv
        G_E_tilde = sign_G_E_tilde * G_E_tilde
        G_E = sign_G_E * G_E

    return G_kq, G_vv, G_E_tilde, G_E


def plot_G_matrices(
        G_kq, G_vv, V,
        cmap='inferno', cp_line_color='w', kv_line_color='gray',
        vmin=0, vmax=None,
        figsize=(12, 12),
        no_title=False,
        no_cbar=True,
        save=False, script_name='', save_dir=None,
):

    # G_kq is (2V, 2V)
    # G_vv is (V, 2V)

    G_matrices = [G_kq, G_vv]
    G_names = ['Gkq', 'Gvv']

    # # estimate signs
    #


    ncols = len(G_names)
    # fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=figsize, sharey=True)


    for G, name in zip(G_matrices, G_names):

        fig = plt.figure(figsize=figsize)

        if isinstance(G, torch.Tensor):
            G = G.cpu().numpy()

        # G *= sign

        ax = plt.gca()
        ax.imshow(
            G.T,
            interpolation='none',
            cmap=cmap,
            origin='upper',  # (0,0) at top-left
            extent=(0, (2*V if name == 'Gkq' else V), 2*V, 0),
            vmin=vmin,
            vmax=vmax,
            aspect=1,
        )
        if not no_title:
            ax.set_title(name)

        if kv_line_color is not None:
            ax.axvline(0.5 * V, color=kv_line_color, linestyle="-", alpha=0.5)
            ax.axhline(0.5 * V, color=kv_line_color, linestyle="-", alpha=0.5)
            ax.axvline(1.5 * V, color=kv_line_color, linestyle="-", alpha=0.5)
            ax.axhline(1.5 * V, color=kv_line_color, linestyle="-", alpha=0.5)

        ax.axvline(V, color=cp_line_color, linestyle="-", alpha=0.75)
        ax.axhline(V, color=cp_line_color, linestyle="-", alpha=0.75)

        # # Axis labels

        # Sub-labels for halves
        half_ax_labels = ['prev', 'curr']

        if name == 'Gvv':
            ax.set_xlim([0, V])
            ax.set_aspect(1)
            ax.set_xticks([V / 2])
            ax.set_yticks([V / 2, 3 * V / 2])
            ax.set_xticklabels([r'$t$'])
            ax.set_yticklabels([r'${\tau-1}$', r'$\tau$'])


        elif name == 'Gkq':
            ax.set_xlim([0, 2 * V])
            ax.set_xticks([V/2, 3*V/2])
            ax.set_yticks([V / 2, 3 * V / 2])
            ax.set_xticklabels([r'${\tau-1}$', r'$\tau$'])
            ax.set_yticklabels([r'${t-1}$', r'$t$'])

        ax.tick_params(axis='y', labelrotation=90)

        # Optional: adjust tick label positions
        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis='y', left=True, right=False, labelleft=True)

        plt.tight_layout()
        if save:
            save_figure_to_png(fig, save_dir=(save_dir or figures_dir), save_name=f'{script_name}__{name}')
        _maybe_show()
        plt.close(fig)


def compute_conv1d_effective_matrices(
        G_vv, G_kq, x_ids,
        dims: MqarDimensions,
):

    V = dims.V

    E_oh = torch.eye(V)

    x_curr = E_oh[:, x_ids[0]]

    x_prev = torch.zeros_like(x_curr)
    x_prev[:, 1:] = x_curr[:, :-1]

    x_hat = torch.vstack([x_prev, x_curr])

    xi_tau = x_hat
    xi_t = x_hat
    x_t = x_curr

    x_Gvv_xi = x_t.T @ G_vv @ xi_tau
    xi_Gkq_xi = xi_t.T @ G_kq.T @ xi_tau

    return x_Gvv_xi, xi_Gkq_xi


def plot_conv1d_effective_matrices(
        x_Gvv_xi, xi_Gkq_xi,
        dims: MqarDimensions,
        figsize=(6, 6),
        no_title=True,
        no_cbar=True,
        cmap='inferno',
        vmin=0, vmax=1,
        save=False, script_name='', save_dir=None,
):

    N_facts = dims.N_facts

    max_t = int(N_facts * 2) - 1  # length of context part

    M_arrays = [x_Gvv_xi, xi_Gkq_xi]
    names = ['x_Gvv_xi', 'xi_Gkq_xi']

    lims = np.array([0 - 0.5, max_t + 0.5])

    for M, name in zip(M_arrays, names):
        M = np.asarray(M)  # torch tensors upset matplotlib's array coercion

        fig = plt.figure(figsize=figsize)
        plt.imshow(M.T, interpolation='none', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlim(lims)
        plt.ylim(lims[::-1])

        plt.tight_layout()
        if save:
            save_figure_to_png(fig, save_dir=(save_dir or figures_dir), save_name=f'{script_name}__{name}')
        _maybe_show()
        plt.close(fig)

