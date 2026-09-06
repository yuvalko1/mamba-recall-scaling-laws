import pytest
import torch
import torch.nn.functional as F

from mamba_tiny.model import Mamba, ModelArgs
from mqar.generators import generate_mqar_batch
from utils.common import set_seed

IGNORED_TOKEN = -100


def _tiny_model(device):
    args = ModelArgs(vocab_size=64, d_model=16, d_state=8, d_conv=2, expand=2,
                     n_heads=1, n_layers=1)
    return Mamba(args).to(device)


def _loss_on_batch(model, batch, device):
    logits = model(batch.x_ids.to(device)).logits
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.y_true_ids.to(device).reshape(-1),
        ignore_index=IGNORED_TOKEN,
    )


@pytest.mark.cpu_ok
def test_forward_backward_finite_on_gpu(training_device):
    device = training_device
    set_seed(0)
    model = _tiny_model(device)
    batch = generate_mqar_batch(V=64, L=32, N_facts=4, batch_size=8, seed=0)
    loss = _loss_on_batch(model, batch, device)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name


@pytest.mark.cpu_ok
def test_short_training_reduces_loss_on_fixed_batch(training_device):
    device = training_device
    set_seed(0)
    model = _tiny_model(device)
    batch = generate_mqar_batch(V=64, L=32, N_facts=4, batch_size=32, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    initial_loss = _loss_on_batch(model, batch, device).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = _loss_on_batch(model, batch, device)
        loss.backward()
        optimizer.step()
    final_loss = _loss_on_batch(model, batch, device).item()

    assert final_loss < initial_loss, (initial_loss, final_loss)
    assert final_loss == final_loss  # not NaN
