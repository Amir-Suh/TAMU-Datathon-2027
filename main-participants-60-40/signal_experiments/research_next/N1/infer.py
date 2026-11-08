"""Standalone NumPy inference for the exported masked GRU and dilated TCN."""

import numpy as np
from scipy.special import expit, softmax

DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def residual(frames, weights, architecture, with_evidence):
    token = weights["payout.weight"][frames.token]
    kinds = weights["machine.weight"][frames.kind][:, None, :]
    machine = np.broadcast_to(kinds, (*token.shape[:2], kinds.shape[-1]))
    evidence = frames.evidence if with_evidence else np.zeros_like(frames.evidence)
    x = np.concatenate((token, machine, frames.context, evidence), axis=-1)
    if architecture == "gru":
        input_affine = x @ weights["encoder.weight_ih_l0"].T + weights["encoder.bias_ih_l0"]
        hidden_weight = weights["encoder.weight_hh_l0"].T
        bias = weights["encoder.bias_hh_l0"]
        state = np.zeros((len(x), hidden_weight.shape[0]), dtype=x.dtype)
        x = np.empty((*x.shape[:2], state.shape[-1]), dtype=state.dtype)
        for tick in range(x.shape[1]):
            reset_h, update_h, new_h = np.split(state @ hidden_weight + bias, 3, axis=-1)
            reset_i, update_i, new_i = np.split(input_affine[:, tick], 3, axis=-1)
            reset = expit(reset_i + reset_h)
            update = expit(update_i + update_h)
            proposed = np.tanh(new_i + reset * new_h)
            state = proposed + update * (state - proposed)
            x[:, tick] = state
    else:
        x = x @ weights["project.weight"].T + weights["project.bias"]
        for i, dilation in enumerate(DILATIONS):
            kernels = weights[f"encoder.{i}.weight"]
            h = x @ kernels[:, :, 1].T + weights[f"encoder.{i}.bias"]
            if dilation < x.shape[1]:
                h[:, dilation:] += x[:, :-dilation] @ kernels[:, :, 0].T
            x += np.maximum(h, 0)
    return x @ weights["head.weight"].T + weights["head.bias"]


def distribution(win_logit, size_logit, adjustment=None):
    if adjustment is not None:
        win_logit = win_logit + adjustment[..., 0]
        size_logit = size_logit + adjustment[..., 1:]
    win = expit(np.asarray(win_logit, dtype=float))
    mix = softmax(np.asarray(size_logit, dtype=float), axis=-1)
    return np.column_stack((1 - win, win[:, None] * mix))
