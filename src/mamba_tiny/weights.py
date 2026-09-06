import torch
from torch import nn

from theory.designed_weights import create_designed_model_weights


def _freeze(x):
    for p in x.parameters():
        p.requires_grad = False


def configure_mamba_block_weights(self, **weights_config):

    # notations
    D = self.args.d_model
    D_in = self.args.d_inner
    N = self.args.d_state
    device = self.device
    args = self.args
    
    # set

    freeze_weights = weights_config.get('freeze_initialized_weights', False)
    requires_grad = not freeze_weights

    if self.n_heads == 1:
        D_in = self.args.d_inner
        P = D_in
    else:
        P = self.P
    
    if (init_value := weights_config.get('init_A', None)) is not None:
        values = torch.full((P, N), fill_value=float(init_value), device=device)
        self.A = nn.Parameter(values, requires_grad=requires_grad)

    if (init_value := weights_config.get('init_D', None)) is not None:
        values = torch.full((P,), fill_value=float(init_value), device=device)
        self.D = nn.Parameter(values, requires_grad=requires_grad)

    if (init_value := weights_config.get('init_W_heads', None)) is not None:
        values = torch.full((P, self.n_heads), fill_value=float(init_value), device=device)
        self.W_heads = nn.Parameter(values, requires_grad=requires_grad)

    if weights_config.get("init_P_in__identity_identity", False):
        assert D_in % D == 0
        assert args.expand == 2  # required
        I = torch.eye(D, device=device)
        W = torch.vstack([I, I])  # shape (D_in, D)
        with torch.no_grad():
            self.in_proj_x.weight.copy_(W)
        if freeze_weights:
            _freeze(self.in_proj_x)

    if weights_config.get("init_P_out__identity_zeros", False):
        assert D_in % D == 0
        assert args.expand == 2  # required
        I = torch.eye(D, device=device)
        Z = torch.zeros((D, D), device=device)
        W = torch.vstack([Z, I]).T
        with torch.no_grad():
            self.out_proj_y.weight.copy_(W)
        if freeze_weights:
            _freeze(self.out_proj_y)

    if weights_config.get("init_W_conv__identity_shift", False):
        # assert self.conv1d.bias is None
        assert args.expand == 2
        O_col = torch.ones((D,))
        Z_col = torch.zeros((D,))
        W_v = torch.concat([Z_col, O_col])
        W_k = torch.concat([O_col, Z_col])
        W_conv = torch.stack([W_k, W_v]).T.unsqueeze(1)
        with torch.no_grad():
            self.conv1d.weight.copy_(W_conv)
        if freeze_weights:
            _freeze(self.conv1d)


def set_mamba_designed_MQAR_weights(model: nn.Module):
    """ note: doesn't freeze weights; only sets them """

    device = model.device

    V = model.V
    D = model.D
    N = model.N

    D_in = int(2 * D)

    E_in, P_in, W, S_B, S_C, P_out, E_out = create_designed_model_weights(V=V, D=D, N=N)

    A = torch.ones((D_in, N))
    D_col = torch.zeros(D_in)


    with torch.no_grad():

        model.embedding.weight.copy_(E_in.T)
        model.lm_head.weight.copy_(E_out)

        model.layers[0].mixer.in_proj_x.weight.copy_(P_in)

        model.layers[0].mixer.conv1d.weight.copy_(W.unsqueeze(1))

        model.layers[0].mixer.A.copy_(A)
        model.layers[0].mixer.S_B.weight.copy_(S_B)
        model.layers[0].mixer.S_C.weight.copy_(S_C)
        model.layers[0].mixer.D.copy_(D_col)

        model.layers[0].mixer.out_proj_y.weight.copy_(P_out)