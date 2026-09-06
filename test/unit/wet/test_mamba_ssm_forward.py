import pytest
import torch

from mqar import MqarDimensions
from utils.config import available_mamba_architectures, get_model_from_config


def test_full_mamba_forward_on_gpu(free_gpu):
    # the real mamba_ssm CUDA kernels, via the same construction path the grid runner uses
    if 'mamba_ssm' not in available_mamba_architectures:
        pytest.skip('mamba_ssm not importable in this environment')

    device = f'cuda:{free_gpu}'
    dims = MqarDimensions(V=64, L=32, N_facts=4, D=32, N=16, Lambda=1, M=1)
    model_config = {
        'class': 'mamba_ssm',
        'variant': 'full',
        'args': {
            'vocab_size': None, 'd_model': None, 'n_layer': None,
            'ssm_cfg': {'d_state': None},
        },
        'kwargs': {},
    }
    model = get_model_from_config(dims=dims, model_config=model_config).to(device)
    input_ids = torch.randint(0, 64, (2, 32), device=device)
    logits = model(input_ids).logits
    assert logits.shape[0] == 2 and logits.shape[1] == 32
    assert logits.shape[2] >= 64  # mamba_ssm pads vocab internally
    assert torch.isfinite(logits).all()
