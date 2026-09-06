import numpy as np
from scipy.stats import norm

from mqar import MqarDimensions


def calculate_scaling_coefficient(
        task_type: str,
        model_type: str,
        model_variant: str = 'linear',
):

    if model_type == 'trained' and model_variant == 'linear':
        if task_type == 'AR':
            a = 1.
            a_label = r''
        elif task_type == 'MQAR':
            a = 5. / 4.
            a_label = r'\frac{5}{4}'
        else:
            raise ValueError
    elif model_type == 'trained' and model_variant == 'full':
        # factor 1/2: an empirical observation; all other factors: theory-based
        if task_type == 'AR':
            a = 1. / 2.
            a_label = r'\frac{1}{2}'
        elif task_type == 'MQAR':
            a = 5. / 8.
            a_label = r'\frac{5}{8}'
        else:
            raise ValueError
    elif model_type == 'designed':
        if task_type == 'AR':
            a = 2.
            a_label = r'2'
        elif task_type == 'MQAR':
            a = 3.
            a_label = r'3'
        else:
            raise ValueError
    else:
        raise ValueError

    return a, a_label


def calculate_scaling_bias(V):
    b = -m_V(V)
    b_label = r'-m_V'
    return b, b_label


def calculate_N_eff(N, Lambda):
    N_eff = N * Lambda
    N_eff_label = r'\Lambda N'
    return N_eff, N_eff_label


def evaluate_theoretical_variance(
        D: np.ndarray,
        N_eff: np.ndarray,
        N_f: float,
        a: float=1,
        a_label: str=r'a',
        N_eff_label: str=r'N',
        approximate_large_N_f: bool = False,
):
    if approximate_large_N_f:
        var = a * N_f / (D * N_eff)
        sigma_inv_label = rf'$\sqrt{{\frac{{{N_eff_label} D}}{{{a_label} N_f + {N_eff_label}}}}}$'

    else:
        var = a * N_f / (D * N_eff) + 1 / D
        sigma_inv_label = rf'$\sqrt{{\frac{{{N_eff_label} D}}{{{a_label} N_f + {N_eff_label}}}}}$'

    sigma = np.sqrt(var)

    return var, sigma, sigma_inv_label


def evaluate_theoretical_accuracy(
    sigma: np.ndarray,
    b: float,
):
    z = 1 / sigma + b
    y = norm.cdf(z)

    return y, z


def predict_theoretical_recall_accuracy(
        dims: MqarDimensions,
        task_type: str,
        model_type: str = 'trained',
        model_variant: str = 'linear',
) -> float:
    """Closed-form recall accuracy of a single grid point: Phi(1/sigma - m_V)."""
    a, _ = calculate_scaling_coefficient(task_type=task_type, model_type=model_type, model_variant=model_variant)
    b, _ = calculate_scaling_bias(dims.V)
    N_eff, _ = calculate_N_eff(dims.N, dims.Lambda)
    _, sigma, _ = evaluate_theoretical_variance(D=dims.D, N_eff=N_eff, N_f=dims.N_facts, a=a)
    accuracy, _ = evaluate_theoretical_accuracy(sigma=sigma, b=b)
    return float(accuracy)


def m_V(V):
    """Typical maximum of V standard gaussians: sqrt(2 ln V) plus its second-order
    correction (see the "Mamba Recall Scaling Laws: Proofs" appendix)."""
    m = np.sqrt(2 * np.log(V)) - (np.log(np.log(V)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(V)))
    return m