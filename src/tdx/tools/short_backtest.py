"""Short-history backtest with episode-disjoint training and validation."""

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

from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.paths import data_root
from tdx.run import fit_round
from tdx.tools.tail_eval import (
    build_model,
    compute_metrics,
    paired_bootstrap,
)


WINDOW = 8
ARTIFACT_PREFIX = 'short_heldout_'


def prepare_short_backtest_round(rnd):
    """Give the fitter only original training episodes and labels."""
    train, validation = (rnd.splits[name] for name in ('train', 'validation'))
    if set(train.ids) & set(validation.ids):
        raise ValueError('Training and validation episode IDs overlap.')
    return fit_round(replace(rnd, splits={'train': train}), dev_spec(False))


def collect_short_episodes(rnd):
    """Select released short-history episodes from validation only."""
    split = rnd.splits['validation']
    legal = np.arange(split.y.shape[-1]) < split.cutoff[:, None, None]
    short = (((split.y >= 0) & legal).sum(axis=-1) == 64).all(axis=-1)
    if not short.any():
        raise ValueError('No short-history episodes in validation.')
    return replace(split, ids=[episode for episode, keep in zip(split.ids, short) if keep],
                   y=split.y[short], mtype=split.mtype[short], signal=split.signal[short],
                   cutoff=split.cutoff[short], n_obs=split.n_obs[short],
                   query_ids=split.query_ids[short], label=None, condition=None,
                   target=None, reference=None)


def short_windows(split):
    """Yield causal prefixes and sampled outcomes from validation history."""
    ticks = np.arange(split.y.shape[-1])
    for offset in range(WINDOW):
        cutoff = split.cutoff - WINDOW + offset
        y = np.where(ticks < cutoff[:, None, None], split.y, -1)
        prefix = replace(split, name='short_eval', y=y, cutoff=cutoff,
                         signal=np.zeros_like(split.signal), n_obs=(y >= 0).sum(axis=-1),
                         label=None, condition=None, target=None, reference=None)
        outcomes = np.take_along_axis(split.y, cutoff[:, None, None], axis=-1)[..., 0]
        yield prefix, outcomes


def write_short_predictions(path, predict, split):
    """Save forecasts keyed by held-out queries and pseudo-cutoffs."""
    preds, outs, cutoffs = [], [], []
    for prefix, outcomes in short_windows(split):
        pred = np.asarray(predict(prefix))
        if (pred.shape != (*outcomes.shape, 5) or not np.isfinite(pred).all()
                or np.any(pred < 0) or np.any(abs(pred.sum(axis=-1) - 1) >= 1e-6)):
            raise ValueError('Invalid short-history probability forecasts.')
        preds.append(pred)
        outs.append(outcomes)
        cutoffs.append(prefix.cutoff)
    np.savez_compressed(path, preds=np.asarray(preds), outs=np.asarray(outs),
                        cutoffs=np.asarray(cutoffs), query_ids=np.asarray(split.query_ids, dtype=str))


def run_short_predict(model_name, seed, out_dir, data=None, round_name='main'):
    """Fit on train and score eight sequential validation prefixes per machine."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ARTIFACT_PREFIX}{model_name}_{seed}.npz"

    rnd = load_round(data_root(data, round_name), round_name, answers=False)
    short = collect_short_episodes(rnd)
    arm, _ = build_model(model_name, seed)
    arm.fit(prepare_short_backtest_round(rnd), dev_spec(False))
    write_short_predictions(out_path, arm.predict, short)

    with np.load(out_path) as saved:
        met = compute_metrics(saved['preds'], saved['outs'])
    print(f"short_backtest {model_name} (seed {seed}) saved to {out_path}")
    print(f"  n={met['n_outcomes']} wins={met['n_wins']} win_part={met['win_part']:.4f} size_part={met['size_part']:.4f} total={met['total_loss']:.4f} brier={met['brier']:.5f}")
    return out_path


def find_short_files(directory, model):
    pattern = re.compile(rf"{ARTIFACT_PREFIX}{re.escape(model)}_-?\d+\.npz")
    return sorted(path for path in Path(directory).glob('*.npz') if pattern.fullmatch(path.name))


def run_short_report(in_dir, models, baseline='pooled_recent'):
    """Subcommand report implementation for short_backtest."""
    if not models:
        raise ValueError('At least one model is required.')
    preds_by_model = {}
    split_key = None

    for model in models:
        files = find_short_files(in_dir, model)
        if not files:
            raise FileNotFoundError(f"No held-out short backtest files for {model!r} in {in_dir!r}")
        seed_preds = []
        for file in files:
            with np.load(file) as saved:
                key = tuple(saved[name] for name in ('query_ids', 'cutoffs', 'outs'))
                if split_key is None:
                    split_key = key
                if any(not np.array_equal(a, b) for a, b in zip(key, split_key)):
                    raise ValueError(f"Held-out episode or outcome mismatch in {file}")
                seed_preds.append(saved['preds'])
        preds_by_model[model] = np.mean(seed_preds, axis=0)
    outs = split_key[2]

    W, E, M, _ = list(preds_by_model.values())[0].shape
    print(f"## Held-Out Short History Backtest Report")
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
    p_pred.add_argument('--model', required=True, choices=['pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm', 'shortctx', 'moegate', 'slbase', 'slbrier', 'slcebrier', 'slsize2', 'slsize4', 'slrec', 'sllen', 'slval'])
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
    p.add_argument('--model', choices=['pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm', 'shortctx', 'moegate', 'slbase', 'slbrier', 'slcebrier', 'slsize2', 'slsize4', 'slrec', 'sllen', 'slval'])
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
