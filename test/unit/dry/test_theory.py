import numpy as np
import pytest

from mqar import MqarDimensions
from theory.approximate_scaling_laws import (
    calculate_N_eff, calculate_scaling_bias, m_V, predict_theoretical_recall_accuracy)


def _dims(**overrides):
    base = dict(V=512, L=64, N_facts=16, D=64, N=16, Lambda=1, M=1)
    base.update(overrides)
    return MqarDimensions(**base)


def test_m_V_grows_with_vocab():
    values = [m_V(v) for v in (64, 512, 4096, 50000)]
    assert all(np.isfinite(values))
    assert values == sorted(values)
    # expected max of V std gaussians: below the sqrt(2 ln V) leading term
    # (the second-order correction is negative), but of its magnitude
    assert 0.5 * np.sqrt(2 * np.log(50000)) < values[-1] < np.sqrt(2 * np.log(50000))


def test_scaling_bias_is_negative_m_V():
    b, label = calculate_scaling_bias(512)
    assert b == pytest.approx(-m_V(512))
    assert isinstance(label, str)


def test_N_eff_definition():
    N_eff, label = calculate_N_eff(N=np.array([8, 16]), Lambda=np.array([1, 2]))
    assert np.all(N_eff >= np.array([8, 16]))  # layers only add capacity
    assert isinstance(label, str)


@pytest.mark.parametrize('task_type', ['AR', 'MQAR'])
def test_predicted_accuracy_is_probability_and_monotone_in_N(task_type):
    accuracies = [
        predict_theoretical_recall_accuracy(dims=_dims(N=n), task_type=task_type)
        for n in (2, 8, 32, 64)
    ]
    for accuracy in accuracies:
        assert 0.0 <= accuracy <= 1.0
    assert accuracies == sorted(accuracies), 'recall should improve with state size N'
    # large state on a small task should essentially solve it
    assert accuracies[-1] > 0.5


def test_predicted_accuracy_rejects_unknown_task():
    with pytest.raises(ValueError):
        predict_theoretical_recall_accuracy(dims=_dims(), task_type='copy')
