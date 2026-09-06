from copy import copy

import numpy as np
import torch

from utils.common import print_array_shapes


def create_designed_model_weights(
        V: int, D: int, N: int,
        print_output_shapes: bool = False,
):

    assert (V >= D) and (D >= N) and (N >= 2)

    # dims
    V_k = V // 2
    V_v = V - V_k

    # baseline E, F
    E = create_scaled_embedding_matrix(V=V, M=D)  # (D, V)
    F = create_F(D=D, N=N)  # (N, D)

    # in/out embeddings
    E_in = copy(E)  # (D, V)
    E_out = E.T  # (V, D)

    # helpers
    I_D = torch.eye(D)
    Z_D = torch.zeros((D, D))
    Z_ND = torch.zeros((N, D))
    I_D_col = torch.ones(D)
    Z_D_col = torch.zeros(D)

    # in/out projections
    P_in = torch.hstack([I_D, I_D]).T  # (2D, D)
    P_out = torch.hstack([Z_D, I_D])  # (D, 2D)

    # conv1d weights
    W_p = torch.hstack([I_D_col, Z_D_col])
    W_c = torch.hstack([Z_D_col, I_D_col])
    W = torch.vstack([W_p, W_c]).T

    # write/read (k/q) projectors
    S_B = torch.hstack([F, Z_ND])  # (N, 2D)
    S_C = torch.hstack([Z_ND, F])  # (N, 2D)

    # print
    if print_output_shapes:
        print_array_shapes(['E_in', 'P_in', 'W', 'S_B', 'S_C', 'P_out', 'E_out'])

    return E_in, P_in, W, S_B, S_C, P_out, E_out


def create_scaled_embedding_matrix(V: int, M: int):
    """Random gaussian embedding matrix (M, V) with unit-norm token columns."""
    assert (V >= M) and (M >= 2)

    G = torch.randn(V, M)
    row_norms = torch.linalg.norm(G, dim=1)
    E = G / row_norms.unsqueeze(1)

    return E.T


def create_F(D: int, N: int):
    """
    Create F ∈ R^{N×D} such that F F^T = (D/N) I_N.
    """
    assert N <= D, "Must have N <= D"

    # 1. Random orthogonal matrix U ∈ R^{D×D}
    G = torch.randn(D, D)
    Q, _ = torch.linalg.qr(G)  # orthonormal columns

    # 2. Select first N rows of Q^T => first N columns of Q
    U = Q[:, :N].T   # shape (N, D)

    # 3. Scale so F F^T = (D/N) I_N
    scale = np.sqrt(D / N)
    F = scale * U

    return F
