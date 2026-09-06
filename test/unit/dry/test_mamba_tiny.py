import pytest
import torch

from mamba_tiny.model import Mamba, MambaBlock, ModelArgs, RMSNorm
from mamba_tiny.scans import compute_attention, compute_recurrence, selective_scan


def _args(**overrides):
    base = dict(vocab_size=64, d_model=8, d_state=4, d_conv=2, expand=2,
                n_heads=1, n_layers=1)
    base.update(overrides)
    return ModelArgs(**base)


# ---------- ModelArgs ----------

def test_post_init_derives_inner_dim_and_dt_rank():
    args = _args(d_model=24, expand=2)
    assert args.d_inner == 48
    assert args.dt_rank == 2  # ceil(24 / 16)
    args = _args(dt_rank=7)
    assert args.dt_rank == 7


def test_post_init_pads_vocab():
    args = _args(vocab_size=50, pad_vocab_size_multiple=8)
    assert args.vocab_size == 56
    args = _args(vocab_size=64, pad_vocab_size_multiple=8)
    assert args.vocab_size == 64


# ---------- construction / forward on CPU ----------

def test_cpu_forward_shapes_and_finite(cpu_default_device):
    torch.manual_seed(0)
    model = Mamba(_args(n_layers=2))
    input_ids = torch.randint(0, 64, (3, 16))
    logits = model(input_ids).logits
    assert logits.shape == (3, 16, 64)
    assert torch.isfinite(logits).all()


def test_single_head_default_init_D_regression(cpu_default_device):
    # regression: with init_D absent and n_heads == 1, the default-D branch used the
    # multi-head-only P attribute (UnboundLocalError); it must fall back to d_inner
    block = MambaBlock(_args(), layer_idx=0)
    assert block.D.shape == (block.args.d_inner,)
    assert torch.all(block.D == 1.0)


@pytest.mark.parametrize('head_pattern', ['MQA', 'MKA', 'MVA', 'MHA'])
def test_multi_head_patterns_forward(cpu_default_device, head_pattern):
    # multi-head is only wired for the paper's simplified attention-mode variant
    # (the selective_scan path would mix per-head P dims with per-D_in delta)
    torch.manual_seed(0)
    model = Mamba(
        _args(d_model=8, n_heads=2),
        head_pattern=head_pattern,
        ssm_mode='attention', compute_attention=True,
        skip_normalizations=True, skip_non_linearities=True, skip_gating=True,
        skip_biases=True, skip_discretization=True, skip_residual_connection=True,
        init_A=1, init_D=0, freeze_initialized_weights=True,
    )
    logits = model(torch.randint(0, 64, (2, 12))).logits
    assert logits.shape == (2, 12, 64)
    assert torch.isfinite(logits).all()


def test_paper_kwargs_init_A_and_D_frozen(cpu_default_device):
    # the paper's simplified variant fixes A=1 and D=0 (no learned D bias), frozen
    block = MambaBlock(_args(), layer_idx=0,
                       init_A=1, init_D=0, freeze_initialized_weights=True)
    assert torch.all(block.A == 1.0) and not block.A.requires_grad
    assert torch.all(block.D == 0.0) and not block.D.requires_grad
    assert block.A.shape == (block.args.d_inner, block.args.d_state)


def test_skip_biases_kwarg_disables_biases(cpu_default_device):
    model = Mamba(_args(), skip_biases=True)
    block = model.layers[0].mixer
    assert block.conv1d.bias is None
    assert block.in_proj_x.bias is None


def test_rmsnorm_is_scale_invariant_per_row():
    norm = RMSNorm(d_model=8)
    x = torch.randn(4, 8)
    torch.testing.assert_close(norm(x), norm(3.0 * x), rtol=1e-4, atol=1e-5)


# ---------- scans ----------

def _scan_inputs(L=8, d_in=6, n=4, batch=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, L, d_in)
    dt = torch.ones(batch, L, d_in)
    A = -torch.rand(d_in, n)  # decaying states, as in the default parameterization
    B = torch.randn(batch, L, n)
    C = torch.randn(batch, L, n)
    D = torch.randn(d_in)
    return x, dt, A, B, C, D


def test_scan_modes_agree_on_short_sequences():
    # the two scan implementations are algebraically equivalent; numerical drift
    # grows with sequence length, so compare on a short one
    x, dt, A, B, C, D = _scan_inputs(L=8)
    y_cumsum = selective_scan(x, dt, A, B, C, D, config={}, mode='cumsum')
    y_logcumsumexp = selective_scan(x, dt, A, B, C, D, config={}, mode='logcumsumexp')
    torch.testing.assert_close(y_cumsum, y_logcumsumexp, rtol=1e-3, atol=1e-4)


def test_attention_matches_recurrence_for_identity_A():
    # both compute_attention and compute_recurrence assume A acts as identity;
    # their SSM outputs must then coincide exactly
    x, _, A, B, C, _ = _scan_inputs(L=8)
    _, y_attention = compute_attention(x, A, B, C)
    _, y_recurrence = compute_recurrence(x, A, B, C)
    torch.testing.assert_close(y_attention, y_recurrence, rtol=1e-4, atol=1e-5)


def test_attention_alpha_is_causal():
    x, _, A, B, C, _ = _scan_inputs(L=8)
    alpha, _ = compute_attention(x, A, B, C)
    upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert (alpha[:, upper] == 0).all()
