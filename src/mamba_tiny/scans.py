
import torch
from torch.nn import functional as F

from utils import recorder



def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)


def selective_scan(x, dt, A, B, C, D, config: dict, mode='logcumsumexp', caller_name=''):

    A = A.to(x.device)
    y_ssm_from_attention = None

    ssm_mode = config.get('ssm_mode', 'selective_scan')

    if config.get('compute_attention', False) or ssm_mode == "attention":


        attention_alpha, y_ssm_from_attention = compute_attention(x, A, B, C)

        recorder.record(caller_name, attention_alpha=attention_alpha)

    match ssm_mode:

        case "selective_scan":

            # discretize
            
            dB = torch.einsum('bld,bln->bldn', dt, B)
            dA = torch.einsum('bld,dn->bldn', dt, A)

            dB_x = torch.einsum('bld,bldn->bldn', x, dB)
            dA = dA.clamp(min=-20)

            recorder.record(caller_name, B_bar=dB, B_bar_x=dB_x, A_bar=dA)

            padding = (0, 0, 0, 0, 1, 0)  # never change this
            dA_star = F.pad(dA[:, 1:], padding).cumsum(1)

            match mode:

                case 'cumsum':

                    dA_cumsum = dA_star.exp()
                    h_ssm = dB_x / (dA_cumsum + 1e-12)
                    h_ssm = h_ssm.cumsum(1) * dA_cumsum
                    y_ssm_raw = torch.einsum('bldn,bln->bld', h_ssm, C)

                    recorder.record(caller_name, h_ssm=h_ssm)

                case 'logcumsumexp':  # more numerically stable (Heisen sequence)

                    dB_x_log = complex_log(dB_x)
                    h_log = torch.logcumsumexp(dB_x_log - dA_star, 1) + dA_star
                    y_ssm_raw = torch.einsum('bldn,bln->bld', h_log.real.exp() * torch.cos(h_log.imag), C)

                    recorder.record(caller_name, h_log_ssm=h_log)

                case _:
                    raise ValueError(f"unknown {mode = }")

            recorder.record(caller_name, y_ssm_scan=y_ssm_raw)

        case "attention":

                if y_ssm_from_attention is None:
                    attention_alpha, y_ssm_from_attention = compute_attention(x, A, B, C)

                y_ssm_raw = y_ssm_from_attention

        case "recurrent":

            h_ssm, y_ssm_raw = compute_recurrence(x, A, B, C)
            recorder.record(caller_name, y_ssm_recurrent=y_ssm_raw, h_ssm=h_ssm)

        case _:
            raise ValueError(f"unknown {config['ssm_mode'] = }")

    # output
    y_ssm_plus_u_D = y_ssm_raw + x * D
    recorder.record(caller_name, y_ssm_plus_u_D=y_ssm_plus_u_D)

    return y_ssm_plus_u_D


def compute_recurrence(x, A, B, C):

    device = x.device

    # shapes
    batch_size, L, D_in = x.shape
    N = B.shape[2]

    # pre-allocate
    h = torch.zeros((batch_size, L, N, D_in), device=device)
    y = torch.zeros((batch_size, L, D_in), device=device)

    # carry h_t across time
    h_t = torch.zeros((batch_size, N, D_in), device=device)

    for t in range(L):
        # slice out time‑t for the whole batch
        x_t = x[:, t, :, None]  # (B, D_in, 1)
        B_t = B[:, t, :, None]  # (B, N, 1)
        C_t = C[:, t, :, None]  # (B, N, 1)


        #   A: (N, N),    h_t: (B, N, D_in)  →  (B, N, D_in)


        # instead, simply assume A is identity operation
        A_prod_h_t = h_t

        # --- write into memory:  h_t = A @ h_t + B_t @ x_t^T  ---
        B_t_prod_x_t = torch.matmul(B_t, x_t.transpose(-2, -1))  # (B, N, D_in)
        h_t = A_prod_h_t + B_t_prod_x_t  # (B, N, D_in)

        # --- read from memory:  y_t = h_t^T @ C_t  ---
        #   h_t^T: (B, D_in, N),  C_t: (B, N, 1)  → (B, D_in, 1)
        y_t = torch.matmul(h_t.transpose(1, 2), C_t).squeeze(-1)  # (B, D_in, 1)

        # save
        h[:, t] = h_t
        y[:, t, :] = y_t.squeeze()

    return h, y


def compute_attention(x, A, B, C):

    batch_size, L, D_in = x.shape

    # 1) compute K once: shape (B, L, L)
    K = torch.einsum('bln,btn->blt', C, B)

    # 2) build a (L, L) matrix of exponents (l-t)
    i = torch.arange(L, device=x.device)
    j = torch.arange(L, device=x.device)
    diff = i[:, None] - j[None, :]  # shape (L, L)
    mask = diff >= 0  # lower‐triangle including diagonal
    exponents = torch.where(mask, diff, torch.zeros_like(diff))  # negative diffs turned to 0



    # 3) broadcast to (B, L, L) and multiply by K
    alpha = K * mask

    # 4) compute y_from_alpha
    y_ssm_from_alpha = torch.einsum('btl,bld->btd', alpha, x)  # (B, L, D)

    return alpha, y_ssm_from_alpha
