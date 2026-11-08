"""Real short_history backtest harness across pooled short episodes."""

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

from tdx.data import load_round, Split
from tdx.paths import data_root
from tdx.run import fit_round
from tdx.tools.tail_eval import (
    build_model,
    compute_parts,
    compute_metrics,
    paired_bootstrap,
)


def prepare_short_backtest_round(rnd):
    """Hide the last 8 observed ticks of every machine in short episodes (ticks cutoff-8..cutoff-1)
    by setting them to -1 and reducing cutoff by 8; other episodes keep their full histories.
    Honest train labels for train short episodes: the realized outcome at the new cutoff; unobserved -> -1.
    """
    trunc_splits = {}
    for s_name, s in rnd.splits.items():
        is_short = ((s.y >= 0).sum(axis=-1) == 64).all(axis=-1)
        new_cutoff = s.cutoff.copy()
        new_cutoff[is_short] -= 8

        new_y = s.y.copy()
        for e in np.where(is_short)[0]:
            new_y[e, :, new_cutoff[e]:] = -1

        new_n_obs = (new_y >= 0).sum(axis=-1).astype(s.n_obs.dtype)

        if s_name == 'train':
            new_label = s.label.copy()
            for e in np.where(is_short)[0]:
                c = new_cutoff[e]
                # Realized outcome at the new cutoff
                new_label[e, :] = np.where(s.y[e, :, c] >= 0, s.y[e, :, c], -1)
            trunc_splits[s_name] = replace(
                s,
                cutoff=new_cutoff,
                y=new_y,
                n_obs=new_n_obs,
                label=new_label,
            )
        else:
            trunc_splits[s_name] = replace(
                s,
                cutoff=new_cutoff,
                y=new_y,
                n_obs=new_n_obs,
                label=None,
                condition=None,
                target=None,
                reference=None,
            )

    return replace(rnd, splits=trunc_splits)


def collect_short_episodes(rnd):
    """Collect all short episodes across train, validation, and test splits (total 96 episodes)."""
    pooled_ids = []
    pooled_y = []
    pooled_mtype = []
    pooled_signal = []
    pooled_cutoff = []
    pooled_query_ids = []

    for s_name in ('train', 'validation', 'test'):
        s = rnd.splits[s_name]
        is_short = ((s.y >= 0).sum(axis=-1) == 64).all(axis=-1)
        for e in np.where(is_short)[0]:
            pooled_ids.append(f"{s_name}_{s.ids[e]}")
            pooled_y.append(s.y[e])
            pooled_mtype.append(s.mtype[e])
            pooled_signal.append(s.signal[e])
            pooled_cutoff.append(s.cutoff[e])
            pooled_query_ids.append(s.query_ids[e])

    return {
        'ids': pooled_ids,
        'y': np.stack(pooled_y, axis=0),  # [96, 8, 256]
        'mtype': np.stack(pooled_mtype, axis=0),  # [96, 8]
        'signal': np.stack(pooled_signal, axis=0),  # [96, 8]
        'cutoff': np.array(pooled_cutoff, dtype=np.int64),  # [96]
        'query_ids': np.stack(pooled_query_ids, axis=0),  # [96, 8]
    }


def run_short_predict(model_name, seed, out_dir, data=None, round_name='main'):
    """Fit model on short-truncated round and predict over 8 ticks (j=0..7) on pooled short episodes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"short_{model_name}_{seed}.npz"

    rnd = load_round(data_root(data, round_name), round_name, answers=False)
    short_pool = collect_short_episodes(rnd)
    rt = prepare_short_backtest_round(rnd)

    arm, spec = build_model(model_name, seed)

    fitted_rt = fit_round(rt, spec)
    # Supply dummy labels for any label_splits missing labels if arm expects it
    f_splits = dict(fitted_rt.splits)
    for s_name in spec.label_splits:
        s = f_splits[s_name]
        if s.label is None:
            f_splits[s_name] = replace(s, label=np.full(s.query_ids.shape, -1, dtype=np.int8))
    fitted_rt = replace(fitted_rt, splits=f_splits)
    arm.fit(fitted_rt, spec)

    # For j in 0..7 predict with pseudo cutoff (new cutoff + j)
    # new cutoff is original cutoff - 8
    orig_cutoffs = short_pool['cutoff']
    orig_y = short_pool['y']
    E_short = len(short_pool['ids'])

    preds, outs = [], []
    for j in range(8):
        pseudo_cutoff = orig_cutoffs - 8 + j
        y_j = orig_y.copy()
        for e in range(E_short):
            y_j[e, :, pseudo_cutoff[e]:] = -1
        n_obs_j = (y_j >= 0).sum(axis=-1).astype(np.int64)

        pseudo_split = Split(
            name='short_eval',
            ids=short_pool['ids'],
            y=y_j,
            mtype=short_pool['mtype'],
            signal=short_pool['signal'],
            cutoff=pseudo_cutoff,
            n_obs=n_obs_j,
            query_ids=short_pool['query_ids'],
        )

        pred_j = arm.predict(pseudo_split)
        out_j = np.zeros((E_short, 8), dtype=np.int8)
        for e in range(E_short):
            out_j[e, :] = orig_y[e, :, pseudo_cutoff[e]]

        preds.append(pred_j)
        outs.append(out_j)

    preds_arr = np.array(preds)  # [8, 96, 8, 5]
    outs_arr = np.array(outs)    # [8, 96, 8]
    np.savez(out_path, preds=preds_arr, outs=outs_arr)

    met = compute_metrics(preds_arr, outs_arr)
    print(f"short_backtest {model_name} (seed {seed}) saved to {out_path}")
    print(f"  n={met['n_outcomes']} wins={met['n_wins']} win_part={met['win_part']:.4f} size_part={met['size_part']:.4f} total={met['total_loss']:.4f} brier={met['brier']:.5f}")
    return out_path


def find_short_files(directory, model):
    """Find matching npz files for a model in directory."""
    d = Path(directory)
    found = []
    for p in d.glob('*.npz'):
        stem = p.stem
        # e.g. short_gru_fac_1 or short_pooled_recent_0 or short_pooled_recent
        if stem.startswith(f"short_{model}"):
            found.append(p)
    return sorted(list(set(found)))


def run_short_report(in_dir, models, baseline='pooled_recent'):
    """Subcommand report implementation for short_backtest."""
    preds_by_model = {}
    outs = None

    for m in models:
        files = find_short_files(in_dir, m)
        if not files:
            raise FileNotFoundError(f"No npz files found for model '{m}' in '{in_dir}'")
        seed_preds = []
        for f in files:
            z = np.load(f)
            seed_preds.append(z['preds'])
            if outs is None:
                outs = z['outs']
        preds_by_model[m] = np.mean(seed_preds, axis=0)

    W, E, M, _ = list(preds_by_model.values())[0].shape
    print(f"## Short History Backtest Report")
    print(f"Ticks per episode: {W} | Short Episodes: {E} | Machines: {M} | Total outcomes: {(outs >= 0).sum()}")
    print()

    # Table 1: Model performance comparison
    print("### Short History Model Performance Comparison")
    print("| Model | Outcomes | Wins | Win Part | Size Part | Total Loss | Brier |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        met = compute_metrics(preds_by_model[m], outs)
        print(f"| {m} | {met['n_outcomes']} | {met['n_wins']} | {met['win_part']:.4f} | {met['size_part']:.4f} | {met['total_loss']:.4f} | {met['brier']:.5f} |")
    print()

    # Table 2: Paired episode bootstrap vs baseline
    if baseline in preds_by_model:
        print(f"### Paired Episode Bootstrap vs Baseline ({baseline}) (B=2000, 95% CI)")
        print("| Model | Size Mean Diff | Size 95% CI | Total Mean Diff | Total 95% CI |")
        print("|:---|---:|:---:|---:|:---:|")
        for m in models:
            if m == baseline:
                print(f"| {m} (baseline) | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] |")
                continue
            boot = paired_bootstrap(preds_by_model[m], preds_by_model[baseline], outs, B=2000, seed=0)
            sz_ci = f"[{boot['size_ci'][0]:.4f}, {boot['size_ci'][1]:.4f}]"
            tot_ci = f"[{boot['total_ci'][0]:.4f}, {boot['total_ci'][1]:.4f}]"
            print(f"| {m} | {boot['size_mean']:.4f} | {sz_ci} | {boot['total_mean']:.4f} | {tot_ci} |")
        print()
    else:
        print(f"Note: Baseline '{baseline}' not in evaluated models; skipping bootstrap comparison.")
        print()


def make_parser():
    p = argparse.ArgumentParser(description="Short History Backtest Tool")
    sub = p.add_subparsers(dest='subcommand')

    # predict subcommand
    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--model', required=True, choices=['pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm', 'shortctx', 'moegate'])
    p_pred.add_argument('--seed', type=int, default=7)
    p_pred.add_argument('--out', required=True)
    p_pred.add_argument('--data')
    p_pred.add_argument('--round', default='main')

    # report subcommand
    p_rep = sub.add_parser('report')
    p_rep.add_argument('--dir', required=True)
    p_rep.add_argument('--models', required=True)
    p_rep.add_argument('--baseline', default='pooled_recent')

    # Top-level args for direct execution: python -m tdx.tools.short_backtest --model ...
    p.add_argument('--model', choices=['pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm', 'shortctx', 'moegate'])
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--out')
    p.add_argument('--data')
    p.add_argument('--round', default='main')

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # If first arg is not a subcommand (report/predict) but flags are provided
    if len(argv) > 0 and argv[0].startswith('--'):
        argv = ['predict'] + argv

    parser = make_parser()
    args = parser.parse_args(argv)

    if args.subcommand == 'predict':
        run_short_predict(args.model, args.seed, args.out, data=args.data, round_name=args.round)
    elif args.subcommand == 'report':
        models_list = [m.strip() for m in args.models.split(',') if m.strip()]
        run_short_report(args.dir, models_list, baseline=args.baseline)
    elif args.model and args.out:
        run_short_predict(args.model, args.seed, args.out, data=args.data, round_name=args.round)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
