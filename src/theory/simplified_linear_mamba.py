from copy import copy

import numpy as np
import torch
from torch import nn

from utils.common import print_array_shapes


def extract_model_weights(model: nn.Module, model_class: str = 'mamba_tiny', print_output_shapes=False):

    match model_class:

        case 'mamba_tiny':

            E = model.embedding.weight.cpu().detach().T
            P_in = model.layers[0].mixer.in_proj_x.weight.cpu().detach()
            W = model.layers[0].mixer.conv1d.weight.cpu().squeeze().detach()
            S_B = model.layers[0].mixer.S_B.weight.cpu().detach()
            S_C = model.layers[0].mixer.S_C.weight.cpu().detach()
            P_out = model.layers[0].mixer.out_proj_y.weight.cpu().detach()

        case _:
            raise NotImplementedError

    if print_output_shapes:
        print_array_shapes(['E', 'P_in', 'W', 'S_B', 'S_C', 'P_out'])

    return E, P_in, W, S_B, S_C, P_out


def compute_effective_operators(E, P_in, W, S_B, S_C, P_out, print_output_shapes=False):

    # make sure similar type
    vals = [E, P_in, W, S_B, S_C, P_out]
    t0 = type(vals[0])
    assert all(type(v) is t0 for v in vals), "all inputs matrices must share the same type"

    # choose np or torch (both supported)
    if t0 == np.ndarray:
        m = np
    elif t0 == torch.Tensor:
        m = torch
    else:
        raise TypeError(f"unsupported input matrix type {t0}")


    # compute effective matrices

    W_p = W[:, 0]  # prev
    W_c = W[:, 1]  # curr

    W_p_diag = m.diag(W_p)
    W_c_diag = m.diag(W_c)

    E_hat_p_in = W_p_diag @ P_in @ E
    E_hat_c_in = W_c_diag @ P_in @ E

    E_hat_in = m.hstack([E_hat_p_in, E_hat_c_in])

    Pi_v_in = copy(E_hat_in)
    Pi_v_out = E.T @ P_out

    Pi_k_in = S_B @ E_hat_in
    Pi_q_in = S_C @ E_hat_in

    if print_output_shapes:
        print_array_shapes(['Pi_q_in', 'Pi_k_in', 'Pi_v_in', 'Pi_v_out'])

    return Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out


def compute_G_matrices(Pi_q_in, Pi_k_in, Pi_v_in, Pi_v_out):

    G_kq = Pi_k_in.T @ Pi_q_in
    G_vv = Pi_v_out @ Pi_v_in

    return G_kq, G_vv


