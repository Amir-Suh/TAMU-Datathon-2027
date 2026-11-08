"""Self-only causal convolutional win forecasts over point-in-time payout windows."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .causal import MISSING
from .schema import PAYOUTS


WINDOW = 40
CHANNELS = 16
KERNEL = 3
DILATIONS = (1, 2, 4, 8)


def history_windows(spins, machines, ticks):
    """Encode the 40 positions ending at t-1, with a separate observed flag."""
    positions = ticks[:, None] + np.arange(-WINDOW, 0)
    outcomes = np.where(positions >= 0, spins[machines[:, None], np.maximum(positions, 0)], MISSING)
    observed = outcomes != MISSING
    payout_flags = outcomes[..., None] == np.arange(len(PAYOUTS))
    features = np.concatenate([payout_flags, observed[..., None]], axis=2).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))


class CausalTemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(len(PAYOUTS) + 1 if index == 0 else CHANNELS,
                                              CHANNELS, KERNEL, dilation=dilation)
                                    for index, dilation in enumerate(DILATIONS))
        self.win = nn.Conv1d(CHANNELS, 1, 1)

    def sequence_logits(self, features):
        hidden = features
        for index, (layer, dilation) in enumerate(zip(self.layers, DILATIONS)):
            activated = F.relu(layer(F.pad(hidden, ((KERNEL - 1) * dilation, 0))))
            hidden = activated if index == 0 else hidden + activated
        return self.win(hidden).squeeze(1)

    def forward(self, features):
        return self.sequence_logits(features)[:, -1]


def with_win_head(base, win):
    """Replace the win probability and keep the conditional positive payout mix."""
    positive = base[:, 1:]
    sizes = positive / positive.sum(axis=1, keepdims=True)
    return np.column_stack([1 - win, win[:, None] * sizes])
