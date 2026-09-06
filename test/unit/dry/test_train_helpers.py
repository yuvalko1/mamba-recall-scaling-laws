import math

import pytest
import torch

from experiments.train import (
    Accumulator, _construct_scheduler_from_config, _safe_path_name,
    atomic_write_bytes, calculate_grad_norm)


def _make_optimizer(lr=1.0):
    model = torch.nn.Linear(4, 4)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _lr_at_steps(scheduler_config, steps, lr=1.0):
    optimizer = _make_optimizer(lr)
    scheduler = _construct_scheduler_from_config(optimizer, scheduler_config)
    lrs = {}
    for step in range(max(steps) + 1):
        if step in steps:
            lrs[step] = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
    return lrs


def test_scheduler_warmup_flat_cosine_phases():
    config = {'num_warmup_steps': 10, 'num_steps_at_max': 20,
              'decay_type': 'cosine', 'num_decay_steps': 40, 'decay_factor': 0.0}
    lrs = _lr_at_steps(config, steps=[0, 5, 10, 29, 30, 50, 70, 100])
    assert lrs[0] == 0.0                       # warmup start
    assert lrs[5] == pytest.approx(0.5)        # mid warmup
    assert lrs[10] == 1.0                      # warmup done -> flat at max
    assert lrs[29] == 1.0                      # end of flat phase
    assert lrs[50] == pytest.approx(0.5)       # cosine midpoint (frac = 0.5)
    assert lrs[70] == pytest.approx(0.0, abs=1e-9)   # decay finished
    assert lrs[100] == pytest.approx(0.0, abs=1e-9)  # flat at floor thereafter


def test_scheduler_linear_and_exp_interpolation():
    linear = {'num_warmup_steps': 0, 'num_steps_at_max': 0,
              'decay_type': 'linear', 'num_decay_steps': 10, 'decay_factor': 0.5}
    lrs = _lr_at_steps(linear, steps=[0, 5, 10])
    assert lrs[0] == 1.0
    assert lrs[5] == pytest.approx(0.75)
    assert lrs[10] == pytest.approx(0.5)

    exp = {**linear, 'interpolation': 'exp'}
    lrs = _lr_at_steps(exp, steps=[5])
    assert lrs[5] == pytest.approx(math.sqrt(0.5))


def test_scheduler_noop_configs_return_none():
    optimizer = _make_optimizer()
    assert _construct_scheduler_from_config(optimizer, None) is None
    assert _construct_scheduler_from_config(optimizer, {}) is None
    # nothing to schedule: no warmup, no decay, factor 1.0
    assert _construct_scheduler_from_config(
        optimizer, {'num_warmup_steps': 0, 'num_decay_steps': 0, 'decay_factor': 1.0}) is None


def test_scheduler_rejects_bad_configs():
    optimizer = _make_optimizer()
    with pytest.raises(ValueError):
        _construct_scheduler_from_config(optimizer, {'num_warmup_steps': -1, 'decay_factor': 0.5})
    with pytest.raises(ValueError):
        _construct_scheduler_from_config(optimizer, {'num_decay_steps': 10, 'decay_factor': 1.5})


def test_calculate_grad_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    assert calculate_grad_norm(model).item() == 0.0  # no grads yet
    with torch.no_grad():
        model.weight.grad = torch.tensor([[3.0, 4.0]])
    assert calculate_grad_norm(model).item() == pytest.approx(5.0)


def test_accumulator_defaults():
    accumulator = Accumulator()
    assert accumulator.correct == 0 and accumulator.total == 0


def test_atomic_write_bytes(tmp_path):
    target = tmp_path / 'sub' / 'file.bin'
    atomic_write_bytes(b'payload', str(target))
    assert target.read_bytes() == b'payload'
    # no temp files left behind
    assert [p.name for p in target.parent.iterdir()] == ['file.bin']
    # overwrite works
    atomic_write_bytes(b'new', str(target))
    assert target.read_bytes() == b'new'


def test_safe_path_name():
    assert _safe_path_name('D=0016, N=0008, seed=1 @ cuda:0-0xabcd') == 'D_0016_N_0008_seed_1_cuda_0_0xabcd'
    assert _safe_path_name('***') == 'default'
