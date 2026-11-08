"""Exact-layout specialist arm for short_history episodes."""

import json
from pathlib import Path
import numpy as np

from tdx.arm import Arm
from .sampler import BUCKET_LUT, ShortCtxDataset, generate_batches, sample_batch


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_np(w, tokens, offsets, mtype, flag, H=64):
    """Pure numpy float64 inference for ShortCtxArm.

    Args:
        w: Dictionary of model weights.
        tokens: int array [B, 64] with payout classes in {0..4}.
        offsets: int array [B, 64] with tick offsets before cutoff.
        mtype: int array [B] with machine types in {0..3}.
        flag: int array [B] with short flags in {0, 1}.
        H: Hidden dimension of GRU.

    Returns:
        log_p: float64 array [B, 5] of log probabilities.
    """
    B = tokens.shape[0]
    b_idx = BUCKET_LUT[np.clip(offsets - 1, 0, len(BUCKET_LUT) - 1)]

    tok_emb = w['tok.weight'][tokens]
    off_emb = w['off.weight'][b_idx]
    typ_emb = w['typ.weight'][mtype][:, None, :]
    flg_emb = w['flg.weight'][flag][:, None, :]
    x = tok_emb + off_emb + typ_emb + flg_emb

    older = x[:, :24, :].mean(axis=1)
    recent = x[:, 24:, :]

    wih = w['cell.weight_ih']
    whh = w['cell.weight_hh']
    bih = w['cell.bias_ih']
    bhh = w['cell.bias_hh']

    h = np.zeros((B, H), dtype=np.float64)
    for t in range(40):
        xt = recent[:, t, :]
        gi = xt @ wih.T + bih
        gh = h @ whh.T + bhh
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h

    feat = np.concatenate([h, older], axis=-1)
    h_mlp = np.maximum(0.0, feat @ w['fc1.weight'].T + w['fc1.bias'])
    o = h_mlp @ w['fc2.weight'].T + w['fc2.bias']

    # Factored win/size head
    wl = o[:, :1]
    sl = o[:, 1:] - o[:, 1:].max(axis=-1, keepdims=True)
    lsm = sl - np.log(np.exp(sl).sum(axis=-1, keepdims=True))
    log_p = np.concatenate([-np.logaddexp(0.0, wl), -np.logaddexp(0.0, -wl) + lsm], axis=-1)
    return log_p


class ShortCtxArm(Arm):
    """Specialist arm trained on exact short_history layout with numpy float64 inference."""

    def __init__(self, d=64, h=64, mlp_h=64, epochs=20, lr=2e-3, wd=1e-2, batch_size=256,
                 dropout=0.1, transductive=True, seed=7, batches_per_epoch=250, exact_ratio=0.5,
                 device=None, smoke=False, **kwargs):
        self.smoke = bool(smoke)
        actual_epochs = 1 if self.smoke else int(epochs)
        actual_bpe = 10 if self.smoke else int(batches_per_epoch)
        actual_batch = 64 if self.smoke else int(batch_size)

        self.p = dict(
            d=int(d),
            h=int(h),
            mlp_h=int(mlp_h),
            epochs=actual_epochs,
            lr=float(lr),
            wd=float(wd),
            batch_size=actual_batch,
            dropout=float(dropout),
            transductive=bool(transductive),
            seed=int(seed),
            batches_per_epoch=actual_bpe,
            exact_ratio=float(exact_ratio),
        )
        self.device = device
        self.w = None
        self.parity = None

    def fit(self, rnd, spec):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        p = self.p
        dev = self.device if self.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(p['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(p['seed'])
        np.random.seed(p['seed'])

        # Build candidate dataset strictly from histories < cutoff
        dataset = ShortCtxDataset(rnd, spec)

        d = p['d']
        H = p['h']
        mlp_h = p['mlp_h']

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.H = H
                self.tok = nn.Embedding(5, d)
                self.off = nn.Embedding(64, d)
                self.typ = nn.Embedding(4, d)
                self.flg = nn.Embedding(2, d)
                self.cell = nn.GRUCell(d, H)
                self.fc1 = nn.Linear(H + d, mlp_h)
                self.drop = nn.Dropout(p['dropout'])
                self.fc2 = nn.Linear(mlp_h, 5)

            def forward(self, tokens, offsets, mtype, flag):
                lut_t = torch.tensor(BUCKET_LUT, device=offsets.device)
                b_idx = lut_t[torch.clamp(offsets - 1, 0, len(BUCKET_LUT) - 1)]
                x = self.tok(tokens) + self.off(b_idx) + self.typ(mtype)[:, None, :] + self.flg(flag)[:, None, :]
                older = x[:, :24, :].mean(dim=1)
                recent = x[:, 24:, :]
                B = tokens.shape[0]
                h = torch.zeros(B, self.H, device=tokens.device, dtype=x.dtype)
                for t in range(40):
                    h = self.cell(recent[:, t, :], h)
                feat = torch.cat([h, older], dim=-1)
                h_mlp = F.relu(self.fc1(feat))
                o = self.fc2(self.drop(h_mlp))
                wl = o[:, :1]
                sl = o[:, 1:]
                log_p = torch.cat([F.logsigmoid(-wl), F.logsigmoid(wl) + F.log_softmax(sl, dim=-1)], dim=-1)
                return log_p

        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        total_steps = p['epochs'] * p['batches_per_epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-5)

        rng = np.random.RandomState(p['seed'])

        for ep in range(p['epochs']):
            net.train()
            batch_gen = generate_batches(
                dataset,
                batch_size=p['batch_size'],
                n_batches=p['batches_per_epoch'],
                rng=rng,
                exact_ratio=p['exact_ratio'],
                include_flag=True,
            )
            for tokens, offsets, mtype, target, flag in batch_gen:
                tok_t = torch.tensor(tokens, dtype=torch.long, device=dev)
                off_t = torch.tensor(offsets, dtype=torch.long, device=dev)
                mt_t = torch.tensor(mtype, dtype=torch.long, device=dev)
                tgt_t = torch.tensor(target, dtype=torch.long, device=dev)
                flg_t = torch.tensor(flag, dtype=torch.long, device=dev)

                log_p = net(tok_t, off_t, mt_t, flg_t)
                loss = F.nll_loss(log_p, tgt_t)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                scheduler.step()

        # Parity check on double float
        net.double().eval()
        check_tokens, check_offsets, check_mtype, check_target, check_flag = sample_batch(
            dataset, batch_size=64, rng=np.random.RandomState(12345), include_flag=True
        )
        ct_t = torch.tensor(check_tokens, dtype=torch.long, device=dev)
        co_t = torch.tensor(check_offsets, dtype=torch.long, device=dev)
        cm_t = torch.tensor(check_mtype, dtype=torch.long, device=dev)
        cf_t = torch.tensor(check_flag, dtype=torch.long, device=dev)

        with torch.no_grad():
            ref = net(ct_t, co_t, cm_t, cf_t).cpu().numpy()

        self.w = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
        got = forward_np(self.w, check_tokens, check_offsets, check_mtype, check_flag, H=H)

        self.parity = float(np.abs(got - ref).max())
        if self.parity >= 1e-6:
            raise RuntimeError(f"Torch vs numpy parity failed: {self.parity:.3e}")

    def predict(self, split):
        """Inference in pure numpy float64.

        Builds the layout from split.y and split.cutoff for every machine:
        - If the machine has exactly 64 observed ticks before cutoff, use them as given
          (flag = 1).
        - Otherwise take the last 40 consecutive ticks + 24 random older observed ticks
          with a deterministic fixed seed (flag = 0).
        """
        E, M, T = split.y.shape
        co = split.cutoff.astype(np.int64)

        # Strictly mask ticks at or after cutoff
        y_masked = np.where(np.arange(T)[None, None, :] < co[:, None, None], split.y, -1)

        tokens_list = []
        offsets_list = []
        mtype_list = []
        flag_list = []

        for e in range(E):
            c_e = co[e]
            for m in range(M):
                y_row = y_masked[e, m]
                n_obs = int((y_row >= 0).sum())

                if n_obs == 64:
                    # Exact short history: use the 64 observed ticks as given
                    obs_ticks = np.where(y_row >= 0)[0]
                    tok = y_row[obs_ticks].astype(np.int8)
                    off = (c_e - obs_ticks).astype(np.int16)
                    flg = 1
                else:
                    # Long history: take last 40 consecutive ticks + 24 older observed ticks
                    rec_ticks = np.arange(c_e - 40, c_e, dtype=np.int16)
                    older_obs = np.where(y_row[:c_e - 40] >= 0)[0]
                    # Fixed deterministic seed per machine
                    fixed_rng = np.random.RandomState(42 + m)
                    if len(older_obs) >= 24:
                        chosen_older = np.sort(fixed_rng.choice(older_obs, size=24, replace=False))
                    else:
                        chosen_older = np.sort(fixed_rng.choice(older_obs, size=24, replace=True))
                    all_ticks = np.concatenate([chosen_older, rec_ticks])
                    tok = y_row[all_ticks].astype(np.int8)
                    off = (c_e - all_ticks).astype(np.int16)
                    flg = 0

                tokens_list.append(tok)
                offsets_list.append(off)
                mtype_list.append(split.mtype[e, m])
                flag_list.append(flg)

        tokens_arr = np.array(tokens_list, dtype=np.int8)
        offsets_arr = np.array(offsets_list, dtype=np.int16)
        mtype_arr = np.array(mtype_list, dtype=np.int64)
        flag_arr = np.array(flag_list, dtype=np.int64)

        log_p = forward_np(self.w, tokens_arr, offsets_arr, mtype_arr, flag_arr, H=self.p['h'])
        p = np.exp(log_p)
        p = np.clip(p, 1e-6, 1.0)
        p = p / p.sum(axis=-1, keepdims=True)

        return p.reshape(E, M, 5)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        (d / 'meta.json').write_text(json.dumps(self.p, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        p = json.loads((d / 'meta.json').read_text())
        arm = cls(**p)
        with np.load(d / 'model.npz') as z:
            arm.w = {k: z[k] for k in z.files}
        return arm
