"""Compare fixed-emission filters that can switch one lag-rule component."""
import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
EXPERIMENTS = SUITE.parent
KIT = EXPERIMENTS.parent
FROZEN = KIT / "m5_m6_repair/results/repaired-m5-m6"
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair"), str(EXPERIMENTS), str(SUITE)]
from models.common.data import MISSING, load_split
from score import compare, sha256
from slot_forecasting.io import read_submission, write_submission
from synthetic import generate, mask_history

BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"


@dataclass(frozen=True)
class Rule:
    component: str
    dwell: int = 96

    @property
    def hazard(self):
        return 0 if self.component == "fixed" else 1 / self.dwell


def frozen_models():
    path = FROZEN / "models.joblib"
    manifest = json.loads((FROZEN / "freeze.json").read_text())
    if sha256(path) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen model hash changed.")
    return joblib.load(path)


def positive_mix(probs):
    positive = probs[:, 1:]
    return positive / positive.sum(axis=1, keepdims=True)


def joint(win, mix):
    return np.column_stack((1 - win, win[:, None] * mix))


def transitions(state, component, pair_prior, strength_prior):
    if component == "polarity":
        return state[:, :, ::-1]
    if component == "pair":
        # Conditional reset distribution proportional to the prior on other pairs.
        source = state / (1 - pair_prior[None, :, None, None])
        return (source.sum(axis=1, keepdims=True) - source) * pair_prior[None, :, None, None]
    if component == "strength":
        return state.sum(axis=3, keepdims=True) * strength_prior
    return state


def rate_grid(model, context, strength):
    low = logit(np.clip(model.lo[context], 1e-6, 1 - 1e-6))
    high = logit(np.clip(model.hi[context], 1e-6, 1 - 1e-6))
    center = ((low + high) / 2)[:, None]
    half_width = ((high - low) / 2)[:, None] * strength
    return expit(center - half_width), expit(center + half_width)


def lag_wins(history, tick, pairs, rates, default):
    a, b = (tick - pairs).T
    lhs, rhs = history[:, a.clip(0)], history[:, b.clip(0)]
    available = (b >= 0)[None, :] & (lhs >= 0) & (rhs >= 0)
    parity = (lhs != rhs).astype(int)
    favorable = parity[:, :, None, None] == np.arange(2)[None, None, :, None]
    low, high = rates
    return np.where(available[:, :, None, None],
                    np.where(favorable, high[:, None, None, :], low[:, None, None, :]),
                    default[:, None, None, None])


def filter_win(model, split, rule, *, prior=None, trace=False):
    """Filter with constant hazard; marginalizing geometric run length is exact here."""
    pairs = np.asarray(model.pairs)
    pair_prior = model.pair_prior if prior is None else np.asarray(prior)
    strength = np.array([.50, .80, 1.0]) if rule.component == "strength" else np.array([1.0])
    strength_prior = np.array([.10, .20, .70]) if len(strength) > 1 else np.ones(1)
    history = np.where(split.X != MISSING, split.X > 0, MISSING)
    if np.any((history != MISSING) & (np.arange(history.shape[1]) >= split.cutoff[:, None])):
        raise ValueError("History must end before cutoff.")
    shape = (split.n, len(pairs), 2, len(strength))
    state = np.broadcast_to(pair_prior[None, :, None, None] * .5 * strength_prior, shape).copy()
    age = np.zeros_like(state)
    result = np.empty(split.n)
    pair_result = np.empty((split.n, len(pairs)))
    direction_result = np.empty((split.n, 2))
    expected_age = np.empty(split.n)
    reset_trace = np.zeros_like(history, dtype=float) if trace else None
    neutral_context = split.machine_type * 3 + 1
    rates = rate_grid(model, neutral_context, strength)
    actual_rates = rate_grid(model, split.context, strength)
    dims = (1, 2, 3)
    for tick in range(int(split.cutoff.max()) + 1):
        h = rule.hazard if tick else 0
        retained = (1 - h) * state
        reset = h * transitions(state, rule.component, pair_prior, strength_prior) if h else 0
        age = (1 - h) * (age + state) if tick else age
        pred = retained + reset
        ready = split.cutoff == tick
        if ready.any():
            p_win = lag_wins(history, tick, pairs, actual_rates, model.win_prior[split.context])
            result[ready] = (pred * p_win).sum(axis=dims)[ready]
            pair_result[ready] = pred.sum(axis=(2, 3))[ready]
            direction_result[ready] = pred.sum(axis=(1, 3))[ready]
            expected_age[ready] = age.sum(axis=dims)[ready]
        if tick == split.cutoff.max():
            break
        p_win = lag_wins(history, tick, pairs, rates, model.win_prior[neutral_context])
        observed = history[:, tick, None, None, None]
        likelihood = np.where(observed == MISSING, 1.0, np.where(observed == 1, p_win, 1 - p_win))
        posterior = pred * likelihood
        mass = posterior.sum(axis=dims, keepdims=True)
        if trace and h:
            reset_trace[:, tick] = ((reset * likelihood) / mass).sum(axis=dims)
        state = posterior / mass
        age = age * likelihood / mass
    if not np.allclose(pair_result.sum(axis=1), 1, atol=1e-10):
        raise ValueError("State mass drifted.")
    return result, pair_result, direction_result, expected_age, reset_trace


def subset(split, rows):
    return replace(split, query_ids=np.asarray(split.query_ids)[rows].tolist(),
                   episode_ids=split.episode_ids[rows],
                   machine_ids=np.asarray(split.machine_ids)[rows].tolist(),
                   machine_type=split.machine_type[rows], signal=split.signal[rows],
                   cutoff=split.cutoff[rows], X=split.X[rows], labels=None,
                   targets=None, reference=None, conditions=None)


def synthetic_runs(model):
    report = {}
    rules = {"fixed": Rule("fixed"), "polarity_32": Rule("polarity", 32),
             "polarity_96": Rule("polarity", 96), "pair_32": Rule("pair", 32),
             "pair_96": Rule("pair", 96), "strength_96": Rule("strength", 96)}
    scenarios = ("stable", "flip", "ramp", "pair_swap", "double_flip")
    for i, scenario in enumerate(scenarios):
        lead_times = (16,) if scenario == "stable" else (8, 16, 40)
        for lead in lead_times:
            f = generate(scenario, episodes=12, seed=5710 + i*100 + lead, lead=lead)
            for tag, s in (("full", f.split), ("sparse", mask_history(f.split, 24))):
                row = {"m3": {"p0_mse": float(np.mean((model.predict_win(s) - (1 - s.targets[:, 0]))**2))}}
                pair_truth = np.array([model.pairs.index(tuple(p)) for p in f.pairs])
                for name, rule in rules.items():
                    pw, pp, dd, age, reset = filter_win(model, s, rule, trace=True)
                    ticks = np.arange(reset.shape[1])[None, :]
                    last = ticks >= s.cutoff[:, None] - 16
                    observed = (s.X != MISSING) & (ticks < s.cutoff[:, None])
                    alarm = (reset >= .15) & observed
                    after = ticks >= f.change[:, None]
                    first = np.min(np.where(alarm & after, ticks - f.change[:, None], np.inf), axis=1)
                    finite = np.isfinite(first)
                    row[name] = {
                        "p0_mse": float(np.mean((pw - (1 - s.targets[:, 0]))**2)),
                        "pair_accuracy": float(np.mean(pp.argmax(axis=1) == pair_truth)),
                        "polarity_accuracy": float(np.mean(dd.argmax(axis=1) == f.polarity)),
                        "mean_run_age": float(age.mean()),
                        "mean_recent_reset_peak": float(np.max(np.where(last, reset, 0), axis=1).mean()),
                        "alarm_rate_within_16_of_change": float(np.mean(first <= 16)) if scenario != "stable" else None,
                        "median_alarm_delay_when_detected": float(np.median(first[finite])) if finite.any() else None,
                        "stable_alarm_rate_after_warmup": float(np.mean(np.any(alarm & (ticks >= 24), axis=1)))
                        if scenario == "stable" else None,
                    }
                report[f"{scenario}_lead{lead}_{tag}"] = row
    fit = generate("stable", episodes=24, seed=40721, rare=True).split
    pair_post = model._infer(fit)["ind"]["post"].sum(axis=(0, 2, 3))
    prior = .9 * pair_post / pair_post.sum() + .1 / len(model.pairs)
    f = generate("stable", episodes=12, seed=40722, rare=True)
    report["synthetic_shift_prior"] = {"train_episodes": 24, "held_episodes": 12}
    for tag, s in (("full", f.split), ("sparse", mask_history(f.split, 24))):
        pp_true = np.array([model.pairs.index(tuple(p)) for p in f.pairs])
        for name, probs in (("carry_main", model.pair_prior), ("refit_prior", prior)):
            pw, pp, *_ = filter_win(model, s, Rule("pair", 96), prior=probs)
            report["synthetic_shift_prior"][f"{tag}_{name}"] = {
                "p0_mse": float(np.mean((pw - (1 - s.targets[:, 0]))**2)),
                "pair_accuracy": float(np.mean(pp.argmax(axis=1) == pp_true)),
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = frozen_models()["m3"]
    val = load_split(KIT / "public", "validation")
    data = replace(val, labels=None, targets=None, reference=None, conditions=None)
    by_id = read_submission(BASE, val.query_ids)
    mix = positive_mix(np.array([by_id[key] for key in val.query_ids]))
    rules = {"fixed": Rule("fixed"),
             **{f"{component}_{dwell}": Rule(component, dwell)
                for component in ("polarity", "pair", "strength") for dwell in (32, 96)}}
    forecasts = {}
    reports = {}
    for name, rule in rules.items():
        pw, pp, _, age, _ = filter_win(model, data, rule)
        forecasts[name] = args.out / f"{name}.csv"
        write_submission(forecasts[name], dict(zip(val.query_ids, joint(pw, mix))))
        reports[name] = {
            "hazard": rule.hazard, "mean_run_age": float(age.mean()),
            "mean_top_pair_confidence": float(pp.max(axis=1).mean()),
        }
    sample = subset(data, np.array([0, 20, 180]))
    order = np.array([2, 0, 1])
    normal, *_ = filter_win(model, sample, Rule("pair", 96))
    permuted, *_ = filter_win(model, subset(sample, order), Rule("pair", 96))
    padded, *_ = filter_win(model, replace(sample, X=np.pad(sample.X, ((0, 0), (0, 9)), constant_values=MISSING)),
                            Rule("pair", 96))
    if not np.array_equal(normal[order], permuted) or not np.array_equal(normal, padded):
        raise ValueError("Row order or missing padding changed the predictions.")
    info = {
        "fit": "Frozen train-fitted M3 high/low and pair prior. Constant-hazard states with fixed emissions; no shared updates.",
        "signal": "Neutral at historical ticks and supplied signal at the cutoff only.",
        "input_forecast": str(BASE.relative_to(KIT)),
        "input_forecast_sha256": sha256(BASE), "source_sha256": sha256(__file__),
        "variants": reports, "row_order_and_padding": "passed",
    }
    (args.out / "plan.json").write_text(json.dumps(info, indent=2) + "\n")
    report = compare(KIT / "public/validation", BASE, forecasts, 60)
    (args.out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, score in report["models"].items():
        print(name, score["points"], score["delta_R"], flush=True)
    if not args.skip_synthetic:
        (args.out / "synthetic_scores.json").write_text(json.dumps(synthetic_runs(model), indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
