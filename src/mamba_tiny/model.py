"""
Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_tiny.scans import selective_scan
from mamba_tiny.weights import configure_mamba_block_weights, set_mamba_designed_MQAR_weights

from utils import recorder


@dataclass
class ModelArgs:
    vocab_size: int
    d_model: int
    d_state: int
    d_conv: int
    expand: int
    n_heads: int
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    pad_vocab_size_multiple: int = 1
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'logcumsumexp'  # 'cumsum'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


def _get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, **kwargs):
        """Full Mamba model."""
        super().__init__()

        self.config = kwargs

        if self.config.get('skip_biases', False):
            args.bias = False
            args.conv_bias = False

        self.args = args

        # notations
        V = self.args.vocab_size
        D = self.args.d_model
        N = self.args.d_state


        # model dimensions
        self.V = V  # vocab size
        self.D = D  # embedding size
        self.N = N  # state size
        self.n_heads = args.n_heads  # number of "attention" heads
        self.n_layers = args.n_layers  # number of layers

        self.device = _get_default_device()

        self.embedding = nn.Embedding(V, D)
        self.layers = nn.ModuleList([ResidualBlock(args, layer_idx=n, **self.config) for n in range(self.n_layers)])

        if not self.config.get('skip_normalizations', False):
            self.norm_f = RMSNorm(D)

        # Tie output projection to embedding weights. See "Weight Tying" paper
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embedding.weight

        if self.config.get('set_designed_mqar_weights', False):
            set_mamba_designed_MQAR_weights(self)


    def forward(self, input_ids, num_last_tokens=0):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """

        x = self.embedding(input_ids)
        recorder.record(self, x_input_ids=input_ids, x_input_embeddings=x)

        E_in = self.embedding.weight
        E_out = self.lm_head.weight

        recorder.record(
            self, E_in=E_in, E_out=E_out,
        )


        for i, layer in enumerate(self.layers):
            recorder.record(self, **{f"layer_{i}_x_input": x})
            x = layer(x)
            recorder.record(self, **{f"layer_{i}_y_output": x})

        if not self.config.get('skip_normalizations', False):
            x = self.norm_f(x)
            recorder.record(self, x_normalized=x)

        x = self.lm_head(x)
        recorder.record(self, y_output_logits=x)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])

        recorder.record(self, y_all_tokens_logits=x)

        if num_last_tokens > 0:
            x = x[:, -num_last_tokens:]

        recorder.record(self, y_last_token_logits=x)

        x = CausalLMOutput(logits=x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, **config):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()

        self.args = args
        self.config = config
        self.mixer = MambaBlock(args, layer_idx=layer_idx, **config)
        
        # notation
        D = args.d_model
        
        if not self.config.get("skip_normalizations", False):
            self.norm = RMSNorm(D)
        
    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        x_residual = x
        recorder.record(self, block_x_residual=x_residual) 
        if not self.config.get("skip_normalizations", False):
            x = self.norm(x)
            recorder.record(self, x_normalized=x) 

        y_mixer_output = self.mixer(x)
        recorder.record(self, block_y_mixer_output=y_mixer_output)

        if not self.config.get("skip_residual_connection", False):
            y_output = y_mixer_output + x_residual
            recorder.record(self, block_y_residual_added_output=y_output)
        else:
            y_output = y_mixer_output

        recorder.record(self, block_y_output=y_output)

        return y_output


class MambaBlock(nn.Module):

    def __init__(self, args: ModelArgs, layer_idx: int, **config):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()

        self.args = args
        self.config = config

        self.layer_idx = layer_idx  # for debug/recording


        self.d_inner = args.d_inner
        self.d_state = args.d_state

        # notations
        D = args.d_model
        D_in = args.d_inner
        N = args.d_state

        self.n_heads = args.n_heads  # number of "attention" heads
        self.N = N


        self.device = _get_default_device()

        self.in_proj_x = nn.Linear(D, D_in, bias=args.bias)
        self.out_proj_y = nn.Linear(D_in, D, bias=args.bias)

        if not self.config.get("skip_gating", False):
            self.in_proj_z = nn.Linear(D, D_in, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=D_in,
            out_channels=D_in,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=D_in,
            padding=args.d_conv - 1,
        )

        # originally:
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        # dt_proj projects Δ from dt_rank to d_in

        self.is_multi_headed = "head_pattern" in self.config.keys()

        if self.is_multi_headed:
            assert self.config["head_pattern"] in ['MQA', 'MKA', 'MVA', 'MHA']

        # single head (original, renamed)
        if not self.is_multi_headed:
            assert self.n_heads == 1
            self.S_B = nn.Linear(D_in, N, bias=False)  # per-head B (key)
            self.S_C = nn.Linear(D_in, N, bias=False)

        # multi-headed
        else:

            # W_contract -> W_x (paper)
            # W_heads -> W_y (paper)

            H = self.n_heads
            assert D_in % H == 0
            P = D_in // H
            self.P = P
            self.W_heads = nn.parameter.Parameter(torch.randn(D_in, P, H), requires_grad=True)

            print(f'{self.config["head_pattern"]=}')

            # multi-head patterns
            if self.config["head_pattern"] in ["MHA", "MKA"]:
                self.S_B = nn.Linear(D_in, H * N, bias=False)  # per-head B (key)
            else:  # "MQA", "MVA"
                self.S_B = nn.Linear(D_in, 1 * N, bias=False)  # shared B (key)
            if self.config["head_pattern"] in ["MHA", "MQA"]:
                self.S_C = nn.Linear(D_in, H * N, bias=False)  # per-head C (query)
            else:  # "MQA", "MVA"
                self.S_C = nn.Linear(D_in, 1 * N, bias=False)  # shared C (query)

            if self.config["head_pattern"] in ["MQA", "MKA"]:
                self.W_contract = nn.parameter.Parameter(torch.randn(P, H), requires_grad=True)  # shared X (value)
            else:  # "MHA", "MVA"
                self.W_contract = nn.parameter.Parameter(torch.randn(P, H, D_in), requires_grad=True)  # shared X (value)

        # discretization
        if not self.config.get("skip_discretization", False):
            self.x_proj_to_delta = nn.Linear(D_in, D_in, bias=True)

        # init
        if self.config.get("init_A", None) is None:
            print('self.config.get("init_A", None) is None')
            A = repeat(torch.arange(1, N + 1), 'n -> d n', d=D_in).to(self.device)
            self.A_log = nn.Parameter(torch.log(A), requires_grad=True)
            self.A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        # init
        if self.config.get("init_D", None) is None:
            print('self.config.get("init_D", None) is None')
            P = self.P if self.n_heads > 1 else self.args.d_inner  # single-head: P was undefined
            self.D = nn.Parameter(torch.ones(P))


        configure_mamba_block_weights(self, **self.config)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        x_in = x
        recorder.record(self, x_in=x_in)
        
        (b, l, d) = x.shape
        
        x = self.in_proj_x(x_in)  # shape (b, l, d_in)
        
        recorder.record(self, x_in_proj=x)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        recorder.record(self, x_conv1d=x)
        
        if not self.config.get("skip_non_linearities", False):
            
            x = F.silu(x)  # original
            
            recorder.record(self, x_conv1d_act=x)

        y = self.ssm(x)

        recorder.record(self, y_ssm_output=y)

        # z stuff
        if not self.config.get("skip_gating", False):
            
            z = self.in_proj_z(x_in)  # shape (b, l, d_in)
            recorder.record(self, z=z)
            
            if not self.config.get("skip_non_linearities", False):
                z = F.silu(z)
                recorder.record(self, z_silu=z)
            
            y = y * z
            recorder.record(self, y_times_z=y)
        
        y = self.out_proj_y(y)
        recorder.record(self, y_out_proj=y)

        return y

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        if self.config.get('init_A', None) is not None:
            A = self.A.float()
        else:
            A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        
        D = self.D.float()

        # # just for recording

        if not self.config.get("skip_discretization", False):
            delta = self.x_proj_to_delta(x)  # (b, l, n)
            recorder.record(self, delta_proj=delta)
            delta = F.softplus(delta)  # (b, l, d_in)
            recorder.record(self, delta_softplus=delta)
        else:
            delta_scale = 1
            delta = delta_scale * torch.ones_like(x)  # (b, l, d_in)

        # original (single-head)
        if not self.is_multi_headed:
            assert self.n_heads == 1
            B = self.S_B(x)  # (b, l, n)
            C = self.S_C(x)  # (b, l, n)
            y = selective_scan(
                x, delta, A, B, C, D, mode=self.args.scan_mode, config=self.config,
                caller_name=f'layers.{self.layer_idx}.heads.{0}',
            )  # (b, l, d)

        # multi-head patterns
        else:

            # split x into heads
            b, L, D_in = x.shape
            H = self.n_heads
            P = self.P
            N = self.N

            # X (value)
            if self.config["head_pattern"] in ["MKA", "MQA"]:  # -> shared X
                # reshape (D,) to (P, H)
                x_heads_raw = x.view(b, L, H, P).permute(0, 1, 3, 2).contiguous()  # (b, l, p, h)
                x_contract = torch.einsum("ph,blph->blp", self.W_contract, x_heads_raw)  # (b, l, p)
                x_heads = x_contract.unsqueeze(-1).repeat(1, 1, 1, H)  # (b, l, p, h)
            else:  # "MHA", "MVA" -> non-shared X
                x_heads = torch.einsum("phd,bld->blph", self.W_contract, x)  # (b, l, p, h)

            # B (key)
            if self.config["head_pattern"] in ["MHA", "MKA"]:  # -> per-head B
                B = self.S_B(x)  # (b, l, h*n)
                B_heads = B.view(b, L, H, N).permute(0, 1, 3, 2).contiguous()  # (b, l, n, h)
            else: # "MQA", "MVA" -> shared B
                B = self.S_B(x)  # (b, l, n)
                B_heads = B.unsqueeze(-1).repeat(1, 1, 1, H)  # (b, l, n, h)

            # C (query)
            if self.config["head_pattern"] in ["MHA", "MQA"]:  # -> per-head C
                C = self.S_C(x)  # (b, l, h*n)
                C_heads = C.view(b, L, H, N).permute(0, 1, 3, 2).contiguous()  # (b, l, n, h)
            else:  # "MKA", "MVA" -> shared C
                C = self.S_C(x)  # (b, l, n)  # shared C (key)
                C_heads = C.unsqueeze(-1).repeat(1, 1, 1, H)  # (b, l, n, h)

            # heads
            y_heads = []
            for h in range(self.n_heads):
                x_h = x_heads[:, :, :, h]  # (b, l, p)
                B_h = B_heads[:, :, :, h]  # (b, l, n)
                C_h = C_heads[:, :, :, h]  # (b, l, n)
                y_h = selective_scan(
                    x_h, delta, A, B_h, C_h, D,
                    mode=self.args.scan_mode,
                    config=self.config,
                    caller_name=f'layers.{self.layer_idx}.heads.{h}'
                )  # (b, l, p)
                y_heads.append(y_h)
            y_heads = torch.stack(y_heads, dim=3)  # (b, l, p, h)
            y = torch.einsum("dph,blph->bld", self.W_heads, y_heads)


        return y  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]


class RMSNorm(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):

        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
