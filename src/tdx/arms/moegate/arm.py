"""Soft mixture of experts inside one sequence model (MoE gate).

Trunk: seq GRU (h=128, no cross-machine message) producing state h_t per machine per tick.
Experts: K=3 factored win/size heads (2-layer MLP 128 -> 64 -> 5).
Gate: g_t = softmax(W [h_t, short_flag, r_t]) over K experts.
Inference: numpy float64 (no torch imported at top-level).
Parity: < 1e-6 between torch and numpy float64.
"""
import json
from pathlib import Path
import numpy as np
from tdx.arm import Arm


def _masked(split):
    t = np.arange(split.y.shape[-1])[None, None, :]
    return np.where(t < split.cutoff[:, None, None], split.y, -1).astype(np.int64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_pair_stats(y):
    """Causal pair-rule statistics per machine at tick t.

    For lag pairs (1,3), (2,5), (4,7): running hit rate over all past scored ticks
    and over the last 40 ticks (scaled to [0,1], missing = 0.5).
    Returns array of shape [E, M, T+1, 6].
    At tick t, slice [:, :, t+1, :] contains statistics using observations strictly prior to t+1.
    """
    E, M, T = y.shape
    z = np.concatenate((y, np.full((E, M, 1), -1, y.dtype)), axis=2)
    obs = (z >= 0)
    win = (z > 0)

    def past_sum(a, width=None):
        cs = np.concatenate((np.zeros((*a.shape[:2], 1)), np.cumsum(a, axis=2)), axis=2)[..., :-1]
        if width is not None:
            cs = cs - np.concatenate((np.zeros((*a.shape[:2], width)), cs[..., :-width]), axis=2)[..., :T + 1]
        return cs

    def shift(a, k):
        out = np.zeros_like(a)
        out[..., k:] = a[..., :-k]
        return out

    stats = []
    for a_, b_ in ((1, 3), (2, 5), (4, 7)):
        valid = shift(obs, a_) & shift(obs, b_)
        match = (shift(win, a_) == shift(win, b_))
        scored = valid & obs
        hit = scored & (win == match)
        n_all, h_all = past_sum(scored), past_sum(hit)
        n_rec, h_rec = past_sum(scored, 40), past_sum(hit, 40)
        acc_all = np.divide(h_all, n_all, out=np.full(n_all.shape, 0.5, dtype=np.float64), where=n_all > 0)
        acc_rec = np.divide(h_rec, n_rec, out=np.full(n_rec.shape, 0.5, dtype=np.float64), where=n_rec > 0)
        stats.extend([acc_all, acc_rec])

    return np.stack(stats, axis=-1).astype(np.float64)


def forward_np(w, y, mt, steps, H, K, d_expert, factor=True):
    """Numpy float64 forward inference for MoeGateArm.

    y: [E, M, T] tokens (-1 missing)
    mt: [E, M] machine types
    steps: number of ticks to step forward
    Returns:
      p_mix: [E, M, steps, 5] mixture probabilities
      gates: [E, M, steps, K] gate probabilities
    """
    E, M, _ = y.shape
    h = np.zeros((E, M, H), dtype=np.float64)
    te = w['typ'][mt]
    wih, whh, bih, bhh = w['wih'], w['whh'], w['bih'], w['bhh']

    # Once a missing tick has appeared in the observed prefix, mark the
    # current input tick. This is causal even for historical training rows.
    flag = (np.cumsum(y < 0, axis=-1) > 0)[..., None].astype(np.float64)

    # causal pair statistics
    pair_stats = compute_pair_stats(y)

    out = []
    gates = []

    for t in range(steps):
        # Trunk GRUCell
        parts = [w['tok'][y[:, :, t] + 1], te]
        x = np.concatenate(parts, axis=-1)
        gi = x @ wih.T + bih
        gh = h @ whh.T + bhh
        r = _sigmoid(gi[..., :H] + gh[..., :H])
        z = _sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
        h = (1 - z) * n + z * h

        if K == 1 and d_expert == 0:
            o = h @ w['hw'].T + w['hb']
            if factor:
                wl = o[..., :1]
                sl = o[..., 1:] - o[..., 1:].max(-1, keepdims=True)
                lsm = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
                log_p = np.concatenate([-np.logaddexp(0, wl), -np.logaddexp(0, -wl) + lsm], -1)
                p_mix = np.exp(log_p)
            else:
                e = np.exp(o - o.max(-1, keepdims=True))
                p_mix = e / e.sum(-1, keepdims=True)
            g_t = np.ones((E, M, 1), dtype=np.float64)
            out.append(p_mix)
            gates.append(g_t)
        else:
            p_experts = []
            for k in range(K):
                if d_expert > 0:
                    eh = np.maximum(0.0, h @ w[f'e{k}_w1'].T + w[f'e{k}_b1'])
                    o_k = eh @ w[f'e{k}_w2'].T + w[f'e{k}_b2']
                else:
                    o_k = h @ w[f'e{k}_w'].T + w[f'e{k}_b']
                if factor:
                    wl = o_k[..., :1]
                    sl = o_k[..., 1:] - o_k[..., 1:].max(-1, keepdims=True)
                    lsm = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
                    log_p_k = np.concatenate([-np.logaddexp(0, wl), -np.logaddexp(0, -wl) + lsm], -1)
                    p_k = np.exp(log_p_k)
                else:
                    e = np.exp(o_k - o_k.max(-1, keepdims=True))
                    p_k = e / e.sum(-1, keepdims=True)
                p_experts.append(p_k)

            # Gate: g_t = softmax(W [h_t, short_flag, r_t])
            r_t = pair_stats[:, :, t + 1, :]
            v_t = np.concatenate([h, flag[:, :, t, :], r_t], axis=-1)
            g_logits = v_t @ w['gw'].T + w['gb']
            g_shift = g_logits - g_logits.max(axis=-1, keepdims=True)
            eg = np.exp(g_shift)
            g_t = eg / eg.sum(axis=-1, keepdims=True)

            # Mixture of PROBABILITY vectors: p_t = sum_k g_{t,k} p_{t,k}
            p_mix = np.zeros((E, M, 5), dtype=np.float64)
            for k in range(K):
                p_mix += g_t[..., k:k + 1] * p_experts[k]
            out.append(p_mix)
            gates.append(g_t)

    return np.stack(out, axis=2), np.stack(gates, axis=2)


class MoeGateArm(Arm):
    def __init__(self, K=3, h=128, d_expert=64, epochs=100, lr=2e-3, wd=1e-2, batch=32, seed=7,
                 transductive=True, w_cut=20.0, factor=True, lambda_bal=1.0, expert_drop_prob=0.1,
                 entropy_bonus=0.01, entropy_pct=0.20, device=None, smoke=False):
        self.p = dict(
            K=int(K),
            h=int(h),
            d_expert=int(d_expert),
            epochs=2 if smoke else int(epochs),
            lr=float(lr),
            wd=float(wd),
            batch=int(batch),
            seed=int(seed),
            transductive=bool(transductive),
            w_cut=float(w_cut),
            factor=bool(factor),
            lambda_bal=float(lambda_bal),
            expert_drop_prob=float(expert_drop_prob),
            entropy_bonus=float(entropy_bonus),
            entropy_pct=float(entropy_pct),
        )
        self.device = device
        self.smoke = smoke
        self.w = None
        self.parity = None

    def fit(self, rnd, spec):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        p = self.p
        dev = self.device if self.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(p['seed'])
        np.random.seed(p['seed'])

        Y, MT, CO, LAB = [], [], [], []
        for name in spec.tick_splits:
            s = rnd.splits[name]
            Y.append(_masked(s))
            MT.append(s.mtype.astype(np.int64))
            CO.append(s.cutoff.astype(np.int64))
            n = len(s.ids)
            lab = np.full((n, 8, 5), np.nan)
            if name in spec.soft_splits and s.target is not None:
                lab = s.target.astype(np.float64)
            elif name in spec.label_splits and s.label is not None:
                ok = s.label >= 0
                lab = np.where(ok[..., None], np.eye(5)[np.clip(s.label, 0, 4)], np.nan)
            LAB.append(lab)

        Y_np = np.concatenate(Y, axis=0)
        MT_np = np.concatenate(MT, axis=0)
        CO_np = np.concatenate(CO, axis=0)
        LAB_np = np.concatenate(LAB, axis=0)

        # Precompute causal pair statistics and short_flag
        R_np = compute_pair_stats(Y_np)
        FLAG_np = (np.cumsum(Y_np < 0, axis=-1) > 0)[..., None].astype(np.float32)

        Y_t = torch.tensor(Y_np, device=dev)
        MT_t = torch.tensor(MT_np, device=dev)
        CO_t = torch.tensor(CO_np, device=dev)
        LAB_t = torch.tensor(LAB_np, dtype=torch.float32, device=dev)
        R_t = torch.tensor(R_np, dtype=torch.float32, device=dev)
        FLAG_t = torch.tensor(FLAG_np, dtype=torch.float32, device=dev)
        has_cut = bool((~torch.isnan(LAB_t[..., 0])).any())

        H = p['h']
        K = p['K']
        d_exp = p['d_expert']

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.tok = nn.Embedding(6, 8)
                s.typ = nn.Embedding(4, 4)
                s.cell = nn.GRUCell(12, H)
                s.drop = nn.Dropout(0.1)
                s.K = K
                s.d_expert = d_exp
                if K == 1 and d_exp == 0:
                    s.head = nn.Linear(H, 5)
                    s.gate = None
                else:
                    if d_exp > 0:
                        s.exp_fc1 = nn.ModuleList([nn.Linear(H, d_exp) for _ in range(K)])
                        s.exp_fc2 = nn.ModuleList([nn.Linear(d_exp, 5) for _ in range(K)])
                    else:
                        s.exp_heads = nn.ModuleList([nn.Linear(H, 5) for _ in range(K)])
                    s.gate = nn.Linear(H + 7, K)

            def forward(s, y, mt, flag, r):
                B, M, T = y.shape
                dtype = s.cell.weight_ih.dtype
                h = torch.zeros(B, M, H, device=y.device, dtype=dtype)
                te = s.typ(mt)

                if s.K == 1 and s.d_expert == 0:
                    out = []
                    gates = []
                    for t in range(T):
                        parts = [s.tok(y[:, :, t] + 1), te]
                        x = torch.cat(parts, -1)
                        h = s.cell(x.reshape(B * M, -1), h.reshape(B * M, -1)).reshape(B, M, -1)
                        h_drop = s.drop(h)
                        o = s.head(h_drop)
                        if p['factor']:
                            wl = o[..., :1]
                            log_p = torch.cat([F.logsigmoid(-wl), F.logsigmoid(wl) + F.log_softmax(o[..., 1:], -1)], -1)
                            p_mix = torch.exp(log_p)
                        else:
                            p_mix = F.softmax(o, -1)
                        g_t = torch.ones(B, M, 1, device=y.device, dtype=dtype)
                        out.append(p_mix)
                        gates.append(g_t)
                    return torch.stack(out, 2), torch.stack(gates, 2)
                else:
                    # GRU Recurrence
                    h_seq = []
                    h_drop_seq = []
                    for t in range(T):
                        parts = [s.tok(y[:, :, t] + 1), te]
                        x = torch.cat(parts, -1)
                        h = s.cell(x.reshape(B * M, -1), h.reshape(B * M, -1)).reshape(B, M, -1)
                        h_seq.append(h)
                        h_drop_seq.append(s.drop(h))
                    H_seq = torch.stack(h_seq, dim=2)  # [B, M, T, H]
                    H_drop = torch.stack(h_drop_seq, dim=2)

                    p_experts = []
                    for k in range(s.K):
                        if s.d_expert > 0:
                            eh = F.relu(s.exp_fc1[k](H_drop))
                            o_k = s.exp_fc2[k](eh)
                        else:
                            o_k = s.exp_heads[k](H_drop)
                        if p['factor']:
                            wl = o_k[..., :1]
                            log_p_k = torch.cat([F.logsigmoid(-wl), F.logsigmoid(wl) + F.log_softmax(o_k[..., 1:], -1)], -1)
                            p_k = torch.exp(log_p_k)
                        else:
                            p_k = F.softmax(o_k, -1)
                        p_experts.append(p_k)
                    P_exp = torch.stack(p_experts, dim=3)  # [B, M, T, K, 5]

                    r_slice = r[:, :, 1:T + 1, :]
                    v_all = torch.cat([H_seq, flag, r_slice], -1)  # [B, M, T, H + 7]
                    g_logits = s.gate(v_all)
                    g_all = F.softmax(g_logits, dim=-1)  # [B, M, T, K]

                    g_mix = g_all
                    if s.training and s.K > 1:
                        drop_elem = (torch.rand(B, device=y.device) < p.get('expert_drop_prob', 0.1))
                        if drop_elem.any():
                            drop_k = torch.randint(0, s.K, (B,), device=y.device)
                            k_oh = F.one_hot(drop_k, num_classes=s.K).to(dtype).view(B, 1, 1, s.K)
                            keep_mask = 1.0 - (drop_elem.to(dtype).view(B, 1, 1, 1) * k_oh)
                            g_dropped = g_mix * keep_mask
                            g_mix = g_dropped / g_dropped.sum(dim=-1, keepdim=True).clamp(min=1e-12)

                    p_mix = (g_mix.unsqueeze(-1) * P_exp).sum(dim=3)  # [B, M, T, 5]
                    return p_mix, g_all

        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        N = Y_t.shape[0]
        entropy_limit_ep = int(p.get('entropy_pct', 0.20) * p['epochs'])

        for ep in range(p['epochs']):
            net.train()
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, p['batch']):
                b = perm[i:i + p['batch']]
                y = Y_t[b]
                lg, g_all = net(y, MT_t[b], FLAG_t[b], R_t[b])

                tgt = y[:, :, 1:]
                msk = tgt >= 0
                preds = lg[:, :, :-1]
                log_p = torch.log(preds.clamp(min=1e-12))
                num = F.nll_loss(log_p[msk], tgt[msk], reduction='sum')
                den = msk.sum().float()

                if has_cut:
                    idx = (CO_t[b] - 1).view(-1, 1, 1, 1).expand(-1, 8, 1, 5)
                    cl = torch.log(lg.gather(2, idx).squeeze(2).clamp(min=1e-12))
                    lab = LAB_t[b]
                    ok = ~torch.isnan(lab[..., 0])
                    num = num + p['w_cut'] * (-(lab[ok] * cl[ok]).sum(-1)).sum()
                    den = den + p['w_cut'] * ok.sum()
                loss = num / den

                # Load-balancing penalty: lambda_bal * sum_k (mean_gate_k - 1/K)^2
                if K > 1:
                    g_valid = g_all[:, :, :-1][msk]
                    if len(g_valid) > 0:
                        mean_gate = g_valid.mean(dim=0)
                        loss_bal = p.get('lambda_bal', 1.0) * ((mean_gate - 1.0 / K) ** 2).sum()
                        loss = loss + loss_bal

                        # Gate entropy bonus: -0.01 * H(g) for first 20% of epochs
                        if ep < entropy_limit_ep:
                            h_per_tick = -(g_valid * torch.log(g_valid.clamp(min=1e-12))).sum(dim=-1)
                            H_val = h_per_tick.mean()
                            loss = loss - p.get('entropy_bonus', 0.01) * H_val

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        net.double().eval()
        with torch.no_grad():
            ref, _ = net(Y_t[:3], MT_t[:3], FLAG_t[:3].double(), R_t[:3].double())
            ref_np = ref.cpu().numpy()

        sd = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
        w = dict(
            tok=sd['tok.weight'],
            typ=sd['typ.weight'],
            wih=sd['cell.weight_ih'],
            whh=sd['cell.weight_hh'],
            bih=sd['cell.bias_ih'],
            bhh=sd['cell.bias_hh'],
        )
        if K == 1 and d_exp == 0:
            w['hw'] = sd['head.weight']
            w['hb'] = sd['head.bias']
        else:
            w['gw'] = sd['gate.weight']
            w['gb'] = sd['gate.bias']
            for k in range(K):
                if d_exp > 0:
                    w[f'e{k}_w1'] = sd[f'exp_fc1.{k}.weight']
                    w[f'e{k}_b1'] = sd[f'exp_fc1.{k}.bias']
                    w[f'e{k}_w2'] = sd[f'exp_fc2.{k}.weight']
                    w[f'e{k}_b2'] = sd[f'exp_fc2.{k}.bias']
                else:
                    w[f'e{k}_w'] = sd[f'exp_heads.{k}.weight']
                    w[f'e{k}_b'] = sd[f'exp_heads.{k}.bias']

        self.w = w
        got, _ = forward_np(self.w, Y_np[:3], MT_np[:3], Y_np.shape[-1], H, K, d_exp, p['factor'])
        self.parity = float(np.abs(got - ref_np).max())
        if self.parity > 1e-9:
            raise RuntimeError(f'numpy/torch parity failed: {self.parity:.3e}')

    def predict(self, split):
        y = _masked(split)
        co = split.cutoff.astype(np.int64)
        p_all = self.step_outputs(y, split.mtype.astype(np.int64), int(co.max()))
        sel = p_all[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        pr = sel.clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)

    def step_outputs(self, y, mtype, steps):
        """Return unmasked float64 mixture probabilities for the first ``steps`` ticks."""
        probs, _ = forward_np(self.w, np.asarray(y), np.asarray(mtype), int(steps),
                              self.p['h'], self.p['K'], self.p['d_expert'], self.p['factor'])
        return np.asarray(probs, dtype=np.float64)

    def usage(self, split):
        """Mean gate weights per expert for each episode (shape: [E, K])."""
        y = _masked(split)
        co = split.cutoff.astype(np.int64)
        _, gates = forward_np(
            self.w, y, split.mtype.astype(np.int64), int(co.max()),
            self.p['h'], self.p['K'], self.p['d_expert'], self.p['factor']
        )
        # Select gate probabilities at the cutoff tick: shape [E, 8, K]
        sel_gates = gates[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        # Mean across machines per episode: shape [E, K]
        return sel_gates.mean(axis=1)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        meta = dict(self.p, flag='causal_v1')
        (d / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        p = json.loads((d / 'meta.json').read_text())
        if p.pop('flag', None) != 'causal_v1':
            raise ValueError("moegate model was trained with the non-causal flag; retrain before loading")
        a = cls(**p)
        with np.load(d / 'model.npz') as z:
            a.w = {k: z[k] for k in z.files}
        return a
