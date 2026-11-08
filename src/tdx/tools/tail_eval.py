"""Evaluation tools for payout size, win part, and tail holdout validation."""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = str(min(12, int(os.environ.get(_var, "12"))))

import argparse
from dataclasses import replace
from pathlib import Path
import re
import sys
import numpy as np

from tdx.backtest import TAIL_SIGNAL_POLICY, truncate, make_pseudo_split
from tdx.data import load_round, with_answers
from tdx.fit import dev_spec
from tdx.paths import data_root
from tdx.run import arm_module, build_arm, fit_round


def compute_parts(p, outs):
    """Decompose next-spin forecast vector p into win part, size part, and total loss.
    Outcome class o in {0,1,2,3,4} for payouts 0,1,2,5,20.
    For forecast vector p:
      win part = -log(p0) if o==0 else -log(1-p0)
      size part (only for wins, o>0) = -log(p_o / (1-p0))
      total = -log(p_o) = win part + size part.
    Probabilities are clipped at 1e-6.
    Returns: wl, sl, tl
    """
    p = np.clip(p, 1e-6, 1.0)
    p0 = p[..., 0]
    pw = 1.0 - p0
    o = np.clip(outs, 0, 4)
    po = np.take_along_axis(p, o[..., None], -1)[..., 0]

    wl = np.where(outs == 0, -np.log(np.clip(p0, 1e-6, 1.0)), -np.log(np.clip(pw, 1e-6, 1.0)))
    sl = -np.log(np.clip(po / np.clip(pw, 1e-6, 1.0), 1e-6, 1.0))
    tl = -np.log(np.clip(po, 1e-6, 1.0))
    return wl, sl, tl


def compute_metrics(p, outs):
    """Compute summary metrics: n outcomes, n wins, win part, size part, total loss, Brier."""
    valid = outs >= 0
    win = valid & (outs > 0)
    wl, sl, tl = compute_parts(p, outs)

    n_outcomes = int(valid.sum())
    n_wins = int(win.sum())

    win_part = float(wl[valid].mean()) if n_outcomes > 0 else float('nan')
    size_part = float(sl[win].mean()) if n_wins > 0 else float('nan')
    total_loss = float(tl[valid].mean()) if n_outcomes > 0 else float('nan')

    o = np.clip(outs, 0, 4)
    onehot = np.eye(5)[o]
    brier_arr = ((p - onehot) ** 2).sum(axis=-1)
    brier = float(brier_arr[valid].mean()) if n_outcomes > 0 else float('nan')

    return {
        'n_outcomes': n_outcomes,
        'n_wins': n_wins,
        'win_part': win_part,
        'size_part': size_part,
        'total_loss': total_loss,
        'brier': brier,
    }


def paired_bootstrap(p_a, p_b, outs, B=2000, seed=0):
    """Paired episode bootstrap of size-part and total differences between model a and model b.
    Shape of p_a, p_b is [W, E, 8, 5], outs is [W, E, 8].
    Episodes are along axis 1.
    """
    valid = outs >= 0
    win = valid & (outs > 0)
    E = outs.shape[1]

    _, sl_a, tl_a = compute_parts(p_a, outs)
    _, sl_b, tl_b = compute_parts(p_b, outs)

    # Size part per episode
    na = (sl_a * win).sum(axis=(0, 2))
    da = win.sum(axis=(0, 2))
    nb = (sl_b * win).sum(axis=(0, 2))
    db = win.sum(axis=(0, 2))

    # Total loss per episode
    nta = (tl_a * valid).sum(axis=(0, 2))
    ntb = (tl_b * valid).sum(axis=(0, 2))
    dt = valid.sum(axis=(0, 2))

    rng = np.random.RandomState(seed)
    diff_size = []
    diff_total = []

    for _ in range(B):
        idx = rng.randint(0, E, size=E)
        d_size_den_a = da[idx].sum()
        d_size_den_b = db[idx].sum()
        if d_size_den_a > 0 and d_size_den_b > 0:
            diff_size.append(na[idx].sum() / d_size_den_a - nb[idx].sum() / d_size_den_b)

        d_tot_den = dt[idx].sum()
        if d_tot_den > 0:
            diff_total.append(nta[idx].sum() / d_tot_den - ntb[idx].sum() / d_tot_den)

    diff_size = np.array(diff_size)
    diff_total = np.array(diff_total)

    size_mean = float(np.mean(diff_size)) if len(diff_size) > 0 else 0.0
    size_ci = np.percentile(diff_size, [2.5, 97.5]) if len(diff_size) > 0 else np.array([0.0, 0.0])

    tot_mean = float(np.mean(diff_total)) if len(diff_total) > 0 else 0.0
    tot_ci = np.percentile(diff_total, [2.5, 97.5]) if len(diff_total) > 0 else np.array([0.0, 0.0])

    return {
        'size_mean': size_mean,
        'size_ci': size_ci,
        'total_mean': tot_mean,
        'total_ci': tot_ci,
    }


def compute_reliability_pwin(p, outs, n_bins=10):
    """Reliability table for predicted P(win): 10 equal-count bins."""
    valid = outs >= 0
    p = np.clip(p, 1e-6, 1.0)
    pw = 1.0 - p[..., 0]
    is_win = (outs > 0).astype(np.float64)

    pw_val = pw[valid]
    win_val = is_win[valid]

    if len(pw_val) == 0:
        return []

    order = np.argsort(pw_val)
    splits = np.array_split(order, n_bins)
    rows = []
    for k, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        rows.append({
            'bin': k + 1,
            'count': len(idx),
            'pred_pwin': float(np.mean(pw_val[idx])),
            'realized_win_rate': float(np.mean(win_val[idx])),
        })
    return rows


def compute_reliability_size(p, outs, n_bins=10):
    """Reliability table for size mix: 10 equal-count bins of predicted 1x share among wins."""
    valid = outs >= 0
    win = valid & (outs > 0)
    p = np.clip(p, 1e-6, 1.0)
    pw = np.maximum(1.0 - p[..., 0], 1e-9)
    p_1x_among_wins = p[..., 1] / pw
    is_1x = (outs == 1).astype(np.float64)

    p1x_win = p_1x_among_wins[win]
    act1x_win = is_1x[win]

    if len(p1x_win) == 0:
        return []

    order = np.argsort(p1x_win)
    splits = np.array_split(order, n_bins)
    rows = []
    for k, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        rows.append({
            'bin': k + 1,
            'count': len(idx),
            'pred_1x_share': float(np.mean(p1x_win[idx])),
            'realized_1x_share': float(np.mean(act1x_win[idx])),
        })
    return rows


NEW_ARMS = ('shortctx', 'moegate')
SEQLOSS_NAMES = {'slbase': 'ce', 'slbrier': 'brier', 'slcebrier': 'ce_brier2', 'slsize2': 'size2', 'slsize4': 'size4',
                 'slrec': 'rec64', 'sllen': 'lenw', 'slval': 'valw'}


def build_model(name, seed=0):
    """Factory to build an arm and its fit_spec for tail_eval and short_backtest."""
    if name == 'pooled_recent':
        module = arm_module('pooled')
        arm = build_arm(module, {'method': 'pooled_recent'})
        spec = dev_spec(False)
    elif name == 'm3':
        module = arm_module('m3flip')
        arm = build_arm(module, {})
        spec = dev_spec(False)
    elif name == 'gbdt':
        module = arm_module('gbdt')
        params = dict(module.VARIANTS['pair_l31'])
        params['threads'] = min(6, os.cpu_count() or 4)
        arm = build_arm(module, params)
        spec = dev_spec(True)
    elif name == 'gru_fac':
        from tdx.arms.seq import build as build_seq
        arm = build_seq(h=128, msg=False, epochs=100, transductive=True, factor=True, seed=seed)
        spec = dev_spec(True)
    elif name == 'gru_plain':
        from tdx.arms.seq import build as build_seq
        arm = build_seq(h=128, msg=False, epochs=100, transductive=True, factor=False, seed=seed)
        spec = dev_spec(True)
    elif name == 'tfm':
        module = arm_module('tfm2')
        params = dict(module.VARIANTS['conv9_alibi'])
        params.update(d=128, L=3, epochs=100, seed=seed)
        arm = build_arm(module, params)
        spec = dev_spec(True)
    elif name in SEQLOSS_NAMES:
        # loss study: model name -> tdx.arms.seqloss VARIANTS key (names are not prefixes of each other)
        module = arm_module('seqloss')
        params = dict(module.VARIANTS[SEQLOSS_NAMES[name]])
        params['seed'] = seed
        arm = build_arm(module, params)
        spec = dev_spec(True)
    elif name in NEW_ARMS:
        # research arms under development: model name == arm name, variant 'pilot' (must define VARIANTS['pilot'])
        module = arm_module(name)
        params = dict(module.VARIANTS['pilot'])
        params['seed'] = seed
        arm = build_arm(module, params)
        spec = dev_spec(bool(params.get('transductive', True)))
    else:
        raise ValueError(f"Unknown model name '{name}'. Must be one of: pooled_recent, m3, gbdt, gru_fac, gru_plain, tfm, {', '.join(NEW_ARMS)}")
    return arm, spec


def run_predict(model_name, seed, hold, out_dir, data=None, round_name='main'):
    """Subcommand predict implementation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tail_{model_name}_{seed}_h{hold}.npz"

    rnd = load_round(data_root(data, round_name), round_name, answers=False)
    rt, full = truncate(rnd, hold)

    arm, spec = build_model(model_name, seed)

    # Honest train labels for earlier cutoff
    tr_full, tr = full['train'], rt.splits['train']
    E = len(tr.ids)
    lab = tr_full.y[np.arange(E)[:, None], np.arange(8)[None, :], tr.cutoff[:, None]].astype(np.int8)
    splits = dict(rt.splits)
    splits['train'] = replace(tr, label=lab, condition=tr_full.condition)

    fitted_rt = fit_round(replace(rt, splits=splits), spec)
    # Supply dummy labels for any label_splits missing labels if arm expects it
    f_splits = dict(fitted_rt.splits)
    for s_name in spec.label_splits:
        s = f_splits[s_name]
        if s.label is None:
            f_splits[s_name] = replace(s, label=np.full(s.query_ids.shape, -1, dtype=np.int8))
    fitted_rt = replace(fitted_rt, splits=f_splits)
    arm.fit(fitted_rt, spec)

    val_t, val_f = rt.splits['validation'], full['validation']
    Ev = len(val_t.ids)
    preds, outs = [], []
    win_len = 30  # scored window is 30 ticks [cutoff-H, cutoff-H+30)
    for j in range(win_len):
        sj = make_pseudo_split(val_t, val_f, j)
        preds.append(arm.predict(sj))
        tick = val_t.cutoff + j
        outs.append(val_f.y[np.arange(Ev)[:, None], np.arange(8)[None, :], tick[:, None]])

    preds_arr = np.array(preds)
    outs_arr = np.array(outs)
    np.savez(out_path, preds=preds_arr, outs=outs_arr, signal_policy=TAIL_SIGNAL_POLICY)
    print(f"Saved {out_path} (preds: {preds_arr.shape}, outs: {outs_arr.shape})")
    return out_path


def find_tail_files(directory, model, hold):
    """Find matching npz files for a model and hold in directory, handling various naming conventions."""
    d = Path(directory)
    found = []
    for p in d.glob('*.npz'):
        stem = p.stem
        # e.g. tail_gru_fac_1_h30 or tail_gru_fac_1 or tail_m3_h60 or tail_m3
        if not stem.startswith(f"tail_{model}"):
            continue
        rest = stem[len(f"tail_{model}"):]
        m_hold = re.search(r'_h(\d+)$', rest)
        if m_hold:
            h = int(m_hold.group(1))
            if h == hold:
                found.append(p)
        else:
            if hold == 30:
                found.append(p)
    return sorted(list(set(found)))


def load_tail_dataset(in_dir, models, holds):
    """Load and aggregate predictions and outcomes across holds and models."""
    preds_by_model = {m: [] for m in models}
    outs_by_hold = []

    for h in holds:
        h_outs = None
        for m in models:
            files = find_tail_files(in_dir, m, h)
            if not files:
                raise FileNotFoundError(f"No npz files found for model '{m}' at hold {h} in '{in_dir}'")
            # Average over all seeds found for this model at hold h
            seed_preds = []
            for f in files:
                with np.load(f) as z:
                    if 'signal_policy' not in z or str(z['signal_policy']) != TAIL_SIGNAL_POLICY:
                        raise ValueError(f'{f}: regenerate forecasts with neutral pseudo-cutoff signal')
                    seed_preds.append(z['preds'])
                    if h_outs is None:
                        h_outs = z['outs']
            avg_pred = np.mean(seed_preds, axis=0)
            preds_by_model[m].append(avg_pred)
        outs_by_hold.append(h_outs)

    # Concatenate across holds along axis 0 (ticks / windows dimension)
    pooled_preds = {m: np.concatenate(preds_by_model[m], axis=0) for m in models}
    pooled_outs = np.concatenate(outs_by_hold, axis=0)
    return pooled_preds, pooled_outs


def run_report(in_dir, models, holds, baseline='pooled_recent', data=None, round_name='main'):
    """Subcommand report implementation."""
    preds, outs = load_tail_dataset(in_dir, models, holds)
    W, E, M, _ = list(preds.values())[0].shape

    print(f"## Tail Holdout Evaluation Report")
    print(f"Holds: {holds} | Windows: {len(holds)} | Ticks: {W} | Episodes: {E} | Machines: {M}")
    print()

    # Table a: Per model summary
    print("### Table a: Model Performance Comparison")
    print("| Model | Outcomes | Wins | Win Part | Size Part | Total Loss | Brier |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    metrics_by_model = {}
    for m in models:
        met = compute_metrics(preds[m], outs)
        metrics_by_model[m] = met
        print(f"| {m} | {met['n_outcomes']} | {met['n_wins']} | {met['win_part']:.4f} | {met['size_part']:.4f} | {met['total_loss']:.4f} | {met['brier']:.5f} |")
    print()

    # Table b: Paired bootstrap vs baseline
    if baseline in preds:
        print(f"### Table b: Paired Episode Bootstrap vs Baseline ({baseline}) (B=2000, 95% CI)")
        print("| Model | Size Mean Diff | Size 95% CI | Total Mean Diff | Total 95% CI |")
        print("|:---|---:|:---:|---:|:---:|")
        for m in models:
            if m == baseline:
                print(f"| {m} (baseline) | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] |")
                continue
            boot = paired_bootstrap(preds[m], preds[baseline], outs, B=2000, seed=0)
            sz_ci = f"[{boot['size_ci'][0]:.4f}, {boot['size_ci'][1]:.4f}]"
            tot_ci = f"[{boot['total_ci'][0]:.4f}, {boot['total_ci'][1]:.4f}]"
            print(f"| {m} | {boot['size_mean']:.4f} | {sz_ci} | {boot['total_mean']:.4f} | {tot_ci} |")
        print()
    else:
        print(f"Note: Baseline '{baseline}' not in evaluated models; skipping bootstrap comparison.")
        print()

    # Table c: Breakdowns
    rnd = load_round(data_root(data, round_name), round_name, answers=False)
    rnd_ans = with_answers(rnd)
    val_s = rnd.splits['validation']
    val_ans_s = rnd_ans.splits['validation']

    mtype_raw = val_s.mtype  # [E, 8]
    cond_raw = val_ans_s.condition  # [E]
    sig_raw = val_s.signal  # [E, 8]

    # Broadcast to [W, E, 8]
    mtype_full = np.broadcast_to(mtype_raw[None, :, :], outs.shape)
    cond_full = np.broadcast_to(cond_raw[None, :, None], outs.shape)
    sig_full = np.broadcast_to(sig_raw[None, :, :], outs.shape)

    valid = outs >= 0
    win = valid & (outs > 0)

    print("### Table c1: Breakdown by Machine Type (steady, balanced, jackpot)")
    print("| Model | Steady Win | Steady Size | Balanced Win | Balanced Size | Jackpot Win | Jackpot Size |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    mtype_map = {'steady': 0, 'balanced': 1, 'jackpot': 2}
    for m in models:
        wl, sl, _ = compute_parts(preds[m], outs)
        vals = []
        for mt_name in ('steady', 'balanced', 'jackpot'):
            mt_val = mtype_map[mt_name]
            v_mask = valid & (mtype_full == mt_val)
            w_mask = win & (mtype_full == mt_val)
            w_loss = wl[v_mask].mean() if v_mask.sum() > 0 else float('nan')
            s_loss = sl[w_mask].mean() if w_mask.sum() > 0 else float('nan')
            vals.extend([w_loss, s_loss])
        print(f"| {m} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} | {vals[4]:.4f} | {vals[5]:.4f} |")
    print()

    print("### Table c2: Breakdown by Validation Condition")
    cond_names = sorted(list(np.unique(cond_raw)))
    header_cond = " | ".join([f"{c} Win | {c} Size" for c in cond_names])
    print(f"| Model | {header_cond} |")
    sep_cond = " | ".join(["---: | ---:" for _ in cond_names])
    print(f"|:---| {sep_cond} |")
    for m in models:
        wl, sl, _ = compute_parts(preds[m], outs)
        c_vals = []
        for c in cond_names:
            v_mask = valid & (cond_full == c)
            w_mask = win & (cond_full == c)
            w_loss = wl[v_mask].mean() if v_mask.sum() > 0 else float('nan')
            s_loss = sl[w_mask].mean() if w_mask.sum() > 0 else float('nan')
            c_vals.append(f"{w_loss:.4f} | {s_loss:.4f}")
        print(f"| {m} | " + " | ".join(c_vals) + " |")
    print()

    print("### Table c3: Breakdown by Signal (-1, 0, 1)")
    print("| Model | Sig -1 Win | Sig -1 Size | Sig 0 Win | Sig 0 Size | Sig 1 Win | Sig 1 Size |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        wl, sl, _ = compute_parts(preds[m], outs)
        sig_vals = []
        for s_val in (-1, 0, 1):
            v_mask = valid & (sig_full == s_val)
            w_mask = win & (sig_full == s_val)
            w_loss = wl[v_mask].mean() if v_mask.sum() > 0 else float('nan')
            s_loss = sl[w_mask].mean() if w_mask.sum() > 0 else float('nan')
            sig_vals.extend([w_loss, s_loss])
        print(f"| {m} | {sig_vals[0]:.4f} | {sig_vals[1]:.4f} | {sig_vals[2]:.4f} | {sig_vals[3]:.4f} | {sig_vals[4]:.4f} | {sig_vals[5]:.4f} |")
    print()

    # Table d: Reliability tables
    print("### Table d1: Reliability Table of Predicted P(win) (10 equal-count bins)")
    for m in models:
        print(f"#### Model: {m}")
        rel_win = compute_reliability_pwin(preds[m], outs, n_bins=10)
        print("| Bin | Count | Mean Pred P(win) | Realized Win Rate |")
        print("|---:|---:|---:|---:|")
        for r in rel_win:
            print(f"| {r['bin']} | {r['count']} | {r['pred_pwin']:.4f} | {r['realized_win_rate']:.4f} |")
        print()

    print("### Table d2: Reliability Table of Size Mix: Predicted 1x Share among Wins (10 equal-count bins)")
    for m in models:
        print(f"#### Model: {m}")
        rel_sz = compute_reliability_size(preds[m], outs, n_bins=10)
        print("| Bin | Count | Mean Pred 1x Share | Realized 1x Share |")
        print("|---:|---:|---:|---:|")
        for r in rel_sz:
            print(f"| {r['bin']} | {r['count']} | {r['pred_1x_share']:.4f} | {r['realized_1x_share']:.4f} |")
        print()


def make_parser():
    p = argparse.ArgumentParser(description="Tail Holdout Evaluation Tool")
    sub = p.add_subparsers(dest='subcommand')

    # predict subcommand
    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--model', required=True, choices=['pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm', 'shortctx', 'moegate', 'slbase', 'slbrier', 'slcebrier', 'slsize2', 'slsize4', 'slrec', 'sllen', 'slval'])
    p_pred.add_argument('--seed', type=int, default=7)
    p_pred.add_argument('--hold', type=int, default=30)
    p_pred.add_argument('--out', required=True)
    p_pred.add_argument('--data')
    p_pred.add_argument('--round', default='main')

    # report subcommand
    p_rep = sub.add_parser('report')
    p_rep.add_argument('--dir', required=True)
    p_rep.add_argument('--models', required=True)
    p_rep.add_argument('--holds', default='30,60,90')
    p_rep.add_argument('--baseline', default='pooled_recent')
    p_rep.add_argument('--data')
    p_rep.add_argument('--round', default='main')

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Handle transparent shortcut if user provides flags directly
    if len(argv) > 0 and argv[0].startswith('--'):
        argv = ['predict'] + argv

    parser = make_parser()
    args = parser.parse_args(argv)

    if args.subcommand == 'predict':
        run_predict(args.model, args.seed, args.hold, args.out, data=args.data, round_name=args.round)
    elif args.subcommand == 'report':
        models_list = [m.strip() for m in args.models.split(',') if m.strip()]
        holds_list = [int(h.strip()) for h in args.holds.split(',') if h.strip()]
        run_report(args.dir, models_list, holds_list, baseline=args.baseline, data=args.data, round_name=args.round)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
