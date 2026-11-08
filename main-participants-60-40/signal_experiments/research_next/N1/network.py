"""Train and export small factored residual sequence models."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 16, 32, 64)
GRU_HIDDEN = 20
TCN_WIDTH = 12


class ResidualModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.payout = nn.Embedding(6, 6)
        self.machine = nn.Embedding(3, 3)
        n_input = 6 + 3 + 3 + 4
        if architecture == "gru":
            self.encoder = nn.GRU(n_input, GRU_HIDDEN, batch_first=True)
            width = GRU_HIDDEN
        else:
            self.project = nn.Linear(n_input, TCN_WIDTH)
            self.encoder = nn.ModuleList(nn.Conv1d(TCN_WIDTH, TCN_WIDTH, 2, dilation=d) for d in DILATIONS)
            width = TCN_WIDTH
        self.head = nn.Linear(width, 5)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, token, kind, context, evidence):
        machine = self.machine(kind)[:, None, :].expand(-1, token.shape[1], -1)
        x = torch.cat((self.payout(token), machine, context, evidence), dim=-1)
        if self.architecture == "gru":
            x = self.encoder(x)[0]
        else:
            x = self.project(x).transpose(1, 2)
            for layer, dilation in zip(self.encoder, DILATIONS):
                x = x + F.relu(layer(F.pad(x, (dilation, 0))))
            x = x.transpose(1, 2)
        return self.head(x)


def inputs(frames, with_evidence):
    evidence = frames.evidence if with_evidence else np.zeros_like(frames.evidence)
    return (torch.from_numpy(frames.token.astype(np.int64)),
            torch.from_numpy(frames.kind.astype(np.int64)),
            torch.from_numpy(frames.context), torch.from_numpy(evidence))


def fit(frames, architecture, with_evidence, seed, epochs, batch_size):
    torch.manual_seed(seed)
    model = ResidualModel(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    design = inputs(frames, with_evidence)
    labels = torch.from_numpy(frames.labels.astype(np.int64))
    win_offset = torch.from_numpy(frames.win_logit)
    size_offset = torch.from_numpy(frames.size_logit)
    loss_trace = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for index in torch.randperm(len(labels), generator=generator).split(batch_size):
            adjustment = model(*(value[index] for value in design))
            outcome = labels[index]
            known, won = outcome >= 0, outcome > 0
            win = win_offset[index] + adjustment[..., 0]
            size = size_offset[index] + adjustment[..., 1:]
            win_loss = F.binary_cross_entropy_with_logits(win[known], won[known].float(), reduction="sum")
            size_loss = F.cross_entropy(size[won], outcome[won] - 1, reduction="sum")
            n = int(known.sum())
            loss = (win_loss + size_loss) / n
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * n
            count += n
        loss_trace.append(total / count)
    return model, loss_trace


def export(model):
    return {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
