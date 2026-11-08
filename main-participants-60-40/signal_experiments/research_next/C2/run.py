"""Test a shared polarity-change hazard against independent and shifted peers."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "C1"))
from run import (BASE, KIT, MISSING, Rule, filter_win, frozen_models, joint, lag_wins,
                 load_split, positive_mix, rate_grid, read_submission, write_submission)
from score import compare, sha256
from synthetic import generate, mask_history

MARGINAL = 1 / 96
HAZARD = 1 - np.sqrt(1 - MARGINAL)
SHIFT = 11


def episode_rows(split):
    episodes = np.unique(split.episode_ids)
    rows = np.stack([np.flatnonzero(split.episode_ids == ep) for ep in episodes])
    order = np.argsort(np.asarray(split.machine_ids)[rows], axis=1)
    rows = np.take_along_axis(rows, order, axis=1)
    if np.any(split.cutoff[rows] != split.cutoff[rows[:, :1]]):
        raise ValueError("Expected a common cutoff for all machines in each episode.")
    return rows


def donor_episodes(split, rows):
    """Map each episode to a different episode with similar observation support."""
    short = np.median((split.X[rows] != MISSING).sum(axis=-1), axis=1) <= 80
    donor = np.full(len(rows), -1, dtype=int)
    for indices in (np.flatnonzero(short), np.flatnonzero(~short)):
        if len(indices) > 1:
            donor[indices] = np.roll(indices, 1)
    return donor


def shared_win(model, split, control="aligned"):
    """Filter machine marginals with a per-tick shared-versus-independent mixture.

    Conditioning on an episode event couples histories. This prototype carries
    machine marginals to the next tick, avoiding an exponential joint state.
    """
    rows = episode_rows(split)
    donor = donor_episodes(split, rows) if control == "permuted" else None
    history = np.where(split.X != MISSING, split.X > 0, MISSING)
    if np.any((history != MISSING) & (np.arange(history.shape[1]) >= split.cutoff[:, None])):
        raise ValueError("Found post-cutoff history.")
    pairs = np.asarray(model.pairs)
    neutral = split.machine_type * 3 + 1
    rates = rate_grid(model, neutral, np.ones(1))
    actual = rate_grid(model, split.context, np.ones(1))
    state = np.broadcast_to(model.pair_prior[None, :, None, None] * .5,
                            (split.n, len(pairs), 2, 1)).copy()
    prior_log_odds = logit(HAZARD)
    log_ratio_trace = np.zeros((SHIFT, split.n))
    result = np.empty(split.n)
    dims = (1, 2, 3)
    latest = int(split.cutoff.max())
    for tick in range(latest + 1):
        flipped = state[:, :, ::-1]
        p_independent = (1 - HAZARD) * state + HAZARD * flipped if tick else state
        predictive = (1 - HAZARD) * p_independent + HAZARD * flipped if tick else state
        at_cutoff = split.cutoff == tick
        if at_cutoff.any():
            p_win = lag_wins(history, tick, pairs, actual, model.win_prior[split.context])
            result[at_cutoff] = (predictive * p_win).sum(axis=dims)[at_cutoff]
        if tick == latest:
            break
        p_win = lag_wins(history, tick, pairs, rates, model.win_prior[neutral])
        observed = history[:, tick, None, None, None]
        evidence = np.where(observed == MISSING, 1.0, np.where(observed == 1, p_win, 1 - p_win))
        local_post = p_independent * evidence
        local_mass = local_post.sum(axis=dims)
        if not tick:
            state = local_post / local_mass[:, None, None, None]
            continue
        reset_post = flipped * evidence
        reset_mass = reset_post.sum(axis=dims)
        own_ratio = np.log(reset_mass) - np.log(local_mass)
        matrix = own_ratio[rows]
        if control == "shifted":
            donor_matrix = log_ratio_trace[tick % SHIFT][rows]
        elif control == "permuted":
            donor_matrix = np.where((donor >= 0)[:, None], matrix[donor.clip(0)], 0)
        else:
            donor_matrix = matrix
        others = donor_matrix.sum(axis=1, keepdims=True) - donor_matrix
        probability = np.empty(split.n)
        probability[rows] = expit(prior_log_odds + matrix + others)
        p = probability[:, None, None, None]
        state = (1 - p) * local_post / local_mass[:, None, None, None] + p * reset_post / reset_mass[:, None, None, None]
        log_ratio_trace[tick % SHIFT] = own_ratio
    return result


def synthetic_comparison(m3, m4):
    report = {}
    for i, scenario in enumerate(("stable", "shared_flip", "staggered_flip")):
        for lead in (8, 16, 40) if scenario != "stable" else (16,):
            f = generate(scenario, episodes=12, seed=7150 + i*100 + lead, lead=lead)
            for label, split in (("full", f.split), ("sparse", mask_history(f.split, 24))):
                forecasts = {"m3": m3.predict_win(split), "m4": m4.predict_win(split),
                             "independent": filter_win(m3, split, Rule("polarity", 96))[0]}
                forecasts.update({name: shared_win(m3, split, name) for name in ("aligned", "shifted", "permuted")})
                report[f"{scenario}_{lead}_{label}"] = {
                    name: float(np.mean((values - (1 - split.targets[:, 0]))**2)) for name, values in forecasts.items()
                }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = frozen_models()
    m3, m4 = models["m3"], models["m4"]
    split = load_split(KIT / "public", "validation")
    split = replace(split, targets=None, reference=None, conditions=None, labels=None)
    raw = read_submission(BASE, split.query_ids)
    mix = positive_mix(np.array([raw[key] for key in split.query_ids]))
    win = {"m4_same_size": m4.predict_win(split),
           "independent": filter_win(m3, split, Rule("polarity", 96))[0]}
    win.update({name: shared_win(m3, split, name) for name in ("aligned", "shifted", "permuted")})
    forecasts = {}
    for name, values in win.items():
        path = args.out / f"{name}.csv"
        write_submission(path, dict(zip(split.query_ids, joint(values, mix))))
        forecasts[name] = path
    from run import subset
    reverse = np.arange(split.n - 1, -1, -1)
    flipped = shared_win(m3, subset(split, reverse))
    padded = shared_win(m3, replace(split, X=np.pad(split.X, ((0, 0), (0, 3)), constant_values=MISSING)))
    if not np.allclose(flipped[::-1], win["aligned"], rtol=0, atol=1e-14) or not np.array_equal(padded, win["aligned"]):
        raise ValueError("Shared filtering is order or padding dependent.")
    (args.out / "plan.json").write_text(json.dumps({
        "conditional_payout_mix": "Frozen hybrid in every arm.",
        "marginal_prior_hazard": MARGINAL, "shared_prior_hazard": HAZARD, "individual_prior_hazard": HAZARD,
        "shift_ticks": SHIFT, "signal": "Cutoff only.", "source_sha256": sha256(__file__),
        "checks": {"reverse_rows": True, "padding": True},
    }, indent=2) + "\n")
    for name, baseline in (("scores", BASE), ("vs_independent", forecasts["independent"])):
        scores = compare(KIT / "public/validation", baseline, forecasts, 60)
        (args.out / f"{name}.json").write_text(json.dumps(scores, indent=2) + "\n")
        print(name, {k: v["points"] for k, v in scores["models"].items()}, flush=True)
    synthetic = synthetic_comparison(m3, m4)
    (args.out / "synthetic_scores.json").write_text(json.dumps(synthetic, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
