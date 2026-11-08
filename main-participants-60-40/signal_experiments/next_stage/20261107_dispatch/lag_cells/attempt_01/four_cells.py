"""Chronological four-cell Beta–Bernoulli E3 win heads."""
from dataclasses import dataclass, replace
from itertools import product
from math import log
import csv
import json
from time import perf_counter

import numpy as np
from scipy.special import betaln, expit, logit

from replay import DATA, OUT, ROOT, load_w2, trace

W2 = load_w2()
W1 = W2.w1
CELL_PARITY = np.array([0, 1, 1, 0])
CELL = np.arange(4)
DIRECTION = np.arange(2)
A_XOR = 2
B_XOR = 1
POOL_STRENGTH = 200
PAIR_STRENGTH = 150
HIGH_PAIR_MASS = 0.9
ARMS = ("E3Frozen", "E3Full", "E3Shared")
SPEC = {
    "raw_cell_order": ["00", "01", "10", "11"],
    "production_preflip": "XOR raw code with 2, flipping lag t-a",
    "train_only_sensitivity": "XOR raw code with 1, flipping lag t-b",
    "direction": ["match_high", "differ_high"],
    "pool_pseudocount": POOL_STRENGTH, "pair_pseudocount": PAIR_STRENGTH,
    "S0": "Inherited train-fitted M3 kappa in each fold",
    "shared_episode_prior": 0.3,
    "held_shifts": W1.SHIFTS, "earlier_signal": 0, "high_pair_mass": HIGH_PAIR_MASS,
    "unavailable": "M3 context-prior beta evidence over incomplete observed triples and matched M3 all-history cutoff fallback",
    "null": "M3 likelihood and prediction",
    "comparison": {"E3Frozen": "new posterior cell means with M3 hypothesis weights",
                   "E3Full": "same predictions with full four-cell evidence",
                   "E3Shared": "same full evidence plus separate episode sharing"},
    "selection": "Prespecified fixed primary XOR and pseudo-counts. The B-XOR sensitivity stays in grouped train.",
}


@dataclass
class Rates:
    pooled: np.ndarray
    pair_delta: np.ndarray
    fit_n: np.ndarray
    fit_k: np.ndarray


def causal_prefix(split):
    ticks = np.arange(split.X.shape[1])
    return replace(split, X=np.where(ticks < split.cutoff[:, None], split.X, W1.MISSING))


def counts(w, cutoff, pair, grid, flip_xor):
    raw_n, raw_k, current = W1.ordered_counts(w, cutoff, pair, grid)
    before_n = raw_n[:, :1] - raw_n
    before_k = raw_k[:, :1] - raw_k
    to_raw = CELL ^ flip_xor
    return raw_n + before_n[..., to_raw], raw_k + before_k[..., to_raw], current, raw_n, raw_k


def parent_rates(model):
    high = CELL_PARITY[None, :] == DIRECTION[:, None]
    return np.where(high[None], model.hi[:, None, None], model.lo[:, None, None])


def offset_fit(base, n, k, strength):
    """Optimize the common logit shift over contexts with a centered pseudo-binomial prior."""
    weight = n.sum(axis=0, keepdims=True)
    proportions = (n + 1) / (weight + n.shape[0])
    augmented_n = n + strength * proportions
    augmented_k = k + strength * proportions * expit(base)
    lower = np.full(n.shape[1:], -20.)
    upper = np.full(n.shape[1:], 20.)
    for _ in range(50):
        mid = (lower + upper) / 2
        derivative = (augmented_k - augmented_n * expit(base + mid)).sum(axis=0)
        lower = np.where(derivative > 0, mid, lower)
        upper = np.where(derivative > 0, upper, mid)
    return (lower + upper) / 2


def fit_rates(model, train, flip_xor, original):
    w = W1.w0.win_history(train)
    post = original["ind"]["post"]
    fit_n = np.zeros((len(model.win_prior), len(model.pairs), 2, 4))
    fit_k = np.zeros_like(fit_n)
    for pi, pair in enumerate(model.pairs):
        n, k, _, _, _ = counts(w, train.cutoff, pair, model.tau_grid, flip_xor)
        for ctx in np.unique(train.context):
            rows = train.context == ctx
            fit_n[ctx, pi] = np.einsum("idt,itc->dc", post[rows, pi], n[rows])
            fit_k[ctx, pi] = np.einsum("idt,itc->dc", post[rows, pi], k[rows])
    base = logit(parent_rates(model))
    pooled = offset_fit(base, fit_n.sum(axis=1), fit_k.sum(axis=1), POOL_STRENGTH)
    pair_delta = offset_fit((base + pooled)[:, None], fit_n, fit_k, PAIR_STRENGTH)
    return Rates(pooled, pair_delta, fit_n, fit_k)


def core(model, split, fit, flip_xor, original, *, audit=False):
    w = W1.w0.win_history(split)
    obs, won = w >= 0, w == 1
    ctx = split.context
    s0 = model.params["kappa"]
    mu = expit(logit(parent_rates(model))[:, None] + fit.pooled + fit.pair_delta)
    u_alpha = (s0 * model.win_prior[ctx])[:, None]
    u_beta = s0 - u_alpha
    llp = np.empty_like(original["core"]["llp"])
    pred = np.empty_like(llp)
    parent_post = original["ind"]["post"]
    fallback = original["core"]["pred"]
    audit_data = [] if audit else None
    for pi, pair in enumerate(model.pairs):
        n, k, current, raw_n, raw_k = counts(w, split.cutoff, pair, model.tau_grid, flip_xor)
        nu = obs.sum(axis=1)[:, None] - raw_n[:, 0].sum(axis=1)[:, None]
        ku = won.sum(axis=1)[:, None] - raw_k[:, 0].sum(axis=1)[:, None]
        ll_u = betaln(u_alpha + ku, u_beta + nu - ku) - betaln(u_alpha, u_beta)
        a = (s0 * mu[ctx, pi])[:, :, None, :]
        b = s0 - a
        n, k = n[:, None], k[:, None]
        cell_ll = betaln(a + k, b + n - k) - betaln(a, b)
        llp[:, pi] = log(1 - model.null_weight) + log(.5 * model.pair_prior[pi]) + cell_ll.sum(axis=-1) + ll_u[:, :, None]
        mean = (a + k) / (s0 + n)
        selected = np.take_along_axis(mean, np.maximum(current, 0)[:, None, None, None], axis=-1)[..., 0]
        pred[:, pi] = np.where(current[:, None, None] >= 0, selected, fallback[:, pi])
        if audit:
            audit_data.append((raw_n, raw_k, current, n[:, 0], k[:, 0], nu[:, 0], ku[:, 0]))
    frozen_win = (parent_post * pred).sum(axis=(1, 2, 3)) + original["ind"]["post0"] * original["core"]["pred0"]
    return dict(original["core"], llp=llp, pred=pred), frozen_win, audit_data


def forecasts(model, split, fit, flip_xor=A_XOR, original=None, *, audit=False):
    split = causal_prefix(split)
    if original is None:
        original = model._infer(split)
    assembled, frozen, audit_data = core(model, split, fit, flip_xor, original, audit=audit)
    independent = model._independent(assembled)
    shared, p_shared = model._shared(assembled, independent, split)
    wins = dict(zip(ARMS, (frozen, independent["p_win"], shared)))
    return wins, assembled, independent, p_shared, audit_data


def likelihood_check():
    # Exhaust all 2^8 possible chronological paths. Each path has history-dependent
    # cell allocation, so the sequential Beta–Bernoulli oracle checks the joint law.
    model = W1.FlipLagPairModel(max_lag=2, min_segment=1, tau_step=1, max_tick=8)
    model.params = {"kappa": 3.5}
    model.win_prior = np.full(W1.N_CONTEXTS, .42)
    model.hi = np.full(W1.N_CONTEXTS, .82)
    model.lo = np.full(W1.N_CONTEXTS, .11)
    model.pair_prior = np.ones(1)
    model.tau_q = np.ones(len(model.tau_grid)) / len(model.tau_grid)
    paths = np.asarray(list(product((0, 1), repeat=8)), dtype=np.int8)
    n = len(paths)
    split = W1.Split("tiny", list(map(str, range(n))), np.arange(n).astype(str),
                     list(map(str, range(n))), np.zeros(n, int), np.zeros(n, int),
                     np.full(n, 8), np.column_stack((paths, np.full(n, -1))))
    delta = np.array([[[-.3, .2, .1, .4], [.1, -.2, .3, -.1]]])
    rates = Rates(delta[0], np.zeros_like(delta), None, None)
    result = {}
    for flip_xor in (A_XOR, B_XOR):
        _, assembled, _, _, _ = forecasts(model, split, rates, flip_xor)
        probabilities = np.exp(assembled["llp"][:, 0] - log((1 - model.null_weight) / 2))
        pmax, raw_max, posterior_mean_max = 0., 0, 0.
        for direction, tau_index in product(DIRECTION, range(probabilities.shape[-1])):
            tau = np.r_[0, model.tau_grid][tau_index]
            prior = expit(logit(parent_rates(model)[split.context[0], direction]) + delta[0, direction])
            prior = np.r_[prior, model.win_prior[split.context[0]]]
            for i, path in enumerate(paths):
                successes = model.params["kappa"] * prior
                trials = np.full(5, model.params["kappa"])
                probability = 1.
                for tick, outcome in enumerate(path):
                    cell = 4 if tick < 2 else 2 * path[tick - 1] + path[tick - 2]
                    if cell < 4 and tick < tau:
                        cell ^= flip_xor
                    p_win = successes[cell] / trials[cell]
                    probability *= p_win if outcome else 1 - p_win
                    trials[cell] += 1
                    successes[cell] += outcome
                raw_max = max(raw_max, int(trials.sum() - 5 * model.params["kappa"]))
                pmax = max(pmax, abs(probability - probabilities[i, direction, tau_index]))
                at_cut = 2 * path[-1] + path[-2]
                posterior_mean_max = max(posterior_mean_max, abs(
                    assembled["pred"][i, 0, direction, tau_index] - successes[at_cut] / trials[at_cut]))
        normalization = float(np.abs(probabilities.sum(axis=0) - 1).max())
        assert max(pmax, normalization, posterior_mean_max) < 1e-12 and raw_max == 8
        result[str(flip_xor)] = {"paths": n, "hypotheses": int(probabilities.shape[1] * probabilities.shape[2]),
                                "max_joint_likelihood_error": pmax, "max_normalization_error": normalization,
                                "max_posterior_mean_error": posterior_mean_max}
    return result


def grouped(train):
    with W1.FOLDS.open(newline="") as stream:
        by_episode = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(stream) if r["split"] == "train"}
    W1.require_ids(by_episode, set(train.episode_ids))
    fold = np.array([by_episode[e] for e in train.episode_ids])
    grouped_rows, fold_trace = [], []
    for held_fold in sorted(set(fold)):
        fit_idx, held_idx = np.flatnonzero(fold != held_fold), np.flatnonzero(fold == held_fold)
        fitting = W1.subset_split(train, fit_idx)
        held = W1.subset_split(train, held_idx)
        fit_ids, held_ids = sorted(set(fitting.episode_ids)), sorted(set(held.episode_ids))
        assert set(fit_ids).isdisjoint(held_ids)
        print(f"Grouped fold {held_fold}: {len(fit_ids)} fit episodes, {len(held_ids)} held", flush=True)
        model = W1.refit_m3(fitting, tune=True)
        original = model._infer(fitting)
        a_rates = fit_rates(model, fitting, A_XOR, original)
        b_rates = fit_rates(model, fitting, B_XOR, original)
        fold_trace.append({"fold": int(held_fold), "fit_episodes": fit_ids, "held_episodes": held_ids,
                           "fit_sha256": W1.id_hash(fit_ids), "held_sha256": W1.id_hash(held_ids),
                           "fit_query_count": len(fit_idx), "held_query_count": len(held_idx),
                           "kappa_fit_only": model.params["kappa"],
                           "pooled_a": a_rates.pooled.tolist(), "pooled_b": b_rates.pooled.tolist()})
        del original
        for shift in W1.SHIFTS:
            frame = W1.prefix(held, shift)
            original = model._infer(frame)
            a_wins, _, _, _, _ = forecasts(model, frame, a_rates, original=original)
            b_wins, _, _, _, _ = forecasts(model, frame, b_rates, B_XOR, original)
            p_pair = original["ind"]["post"].sum(axis=(2, 3))
            pair_idx = p_pair.argmax(axis=1)
            slice_47 = pair_idx == model.pairs.index(W1.TARGET_PAIR)
            for i in np.flatnonzero(frame.labels != W1.MISSING):
                grouped_rows.append({"query_id": frame.query_ids[i], "episode_id": frame.episode_ids[i],
                                     "fold": int(held_fold), "shift": shift, "signal": int(frame.signal[i]),
                                     "win": int(frame.labels[i] > 0), "top_pair_4_7": bool(slice_47[i]),
                                     "high_pair_mass": bool(p_pair[i].max() >= HIGH_PAIR_MASS),
                                     "M3": float(original["p_win"][i]),
                                     **{name: float(values[i]) for name, values in a_wins.items()},
                                     "E3Full_XOR_B": float(b_wins["E3Full"][i])})
    W1.write_rows(OUT / "train_grouped_predictions.csv", tuple(grouped_rows[0]), grouped_rows)
    W1.write_json(OUT / "episode_fold_trace.json", fold_trace)
    y = np.array([r["win"] for r in grouped_rows])
    episodes = np.array([r["episode_id"] for r in grouped_rows])
    endpoint = np.array([r["shift"] == 0 for r in grouped_rows])
    metrics = {}
    for name in ("M3", *ARMS, "E3Full_XOR_B"):
        p = np.array([r[name] for r in grouped_rows])
        masks = {"all": np.ones(len(y), bool), "actual_cutoff": endpoint, "earlier_cutoffs": ~endpoint,
                 **{key: np.array([r[key] for r in grouped_rows]) for key in ("top_pair_4_7", "high_pair_mass")}}
        metrics[name] = {key: W1.grouped_metrics(episodes, y, p, mask) for key, mask in masks.items()}
    W1.write_json(OUT / "grouped_metrics.json", metrics)
    return metrics


def causality(model, train, fit):
    sample = W1.subset_split(train, np.arange(16))
    earlier = W1.prefix(sample, 16)
    changed = np.where(np.arange(sample.X.shape[1])[None, :] >= earlier.cutoff[:, None], 4, sample.X)
    altered = replace(earlier, X=changed, labels=np.ones(sample.n, dtype=int))
    a, core_a, post_a, _, _ = forecasts(model, earlier, fit)
    b, core_b, post_b, _, _ = forecasts(model, altered, fit)
    for arm in ARMS:
        np.testing.assert_array_equal(a[arm], b[arm])
    for key in ("llp", "pred"):
        np.testing.assert_array_equal(core_a[key], core_b[key])
    np.testing.assert_array_equal(post_a["post"], post_b["post"])
    padded = replace(earlier, X=np.pad(earlier.X, ((0, 0), (0, 13)), constant_values=W1.MISSING))
    padded_wins, _, _, _, _ = forecasts(model, padded, fit)
    for arm in ARMS:
        np.testing.assert_array_equal(a[arm], padded_wins[arm])
    w = W1.w0.win_history(earlier)
    raw_n, raw_k, _ = W1.ordered_counts(w, earlier.cutoff, (4, 7), model.tau_grid)
    n, k, _, _, _ = counts(w, earlier.cutoff, (4, 7), model.tau_grid, A_XOR)
    np.testing.assert_array_equal(n.sum(axis=-1), np.broadcast_to(raw_n[:, :1].sum(axis=-1), n.shape[:2]))
    np.testing.assert_array_equal(k.sum(axis=-1), np.broadcast_to(raw_k[:, :1].sum(axis=-1), k.shape[:2]))
    return {"at_cutoff_and_future_outcomes_and_cutoff_labels_invariant": True,
            "extra_missing_padding_invariant": True, "xor_permutation_preserves_counts": True,
            "chronological_likelihood_enumeration": likelihood_check()}


def emit(split_name, model, fit):
    split = W1.input_only(DATA, split_name)
    original = model._infer(split)
    audit = split_name == "validation"
    wins, assembled, posterior, p_shared, details = forecasts(model, split, fit, original=original, audit=audit)
    hybrid = W1.COMBINED.with_suffix(".csv") if audit else W1.COMBINED.with_name(W1.COMBINED.name + "_test.csv")
    donor = W1.conditional_mix(W1.array_at(hybrid, split))
    forecast_hashes = {}
    for name, w in wins.items():
        path = OUT / f"{split_name}_{name}.csv"
        probabilities = W1.combine(w, donor)
        W1.write_submission(path, dict(zip(split.query_ids, probabilities)))
        actual = W1.array_at(path, split)
        np.testing.assert_allclose(W1.conditional_mix(actual), donor, atol=1e-12, rtol=0)
        np.testing.assert_allclose(probabilities, actual, atol=1e-12, rtol=0)
        forecast_hashes[path.name] = W1.sha256(path)
    if audit:
        write_details(split, model, fit, original, posterior, details, assembled, p_shared, wins)
    return forecast_hashes


def write_details(split, model, fit, original, posterior, details, assembled, p_shared, wins):
    pair_posterior = posterior["post"].sum(axis=(2, 3))
    parent_pair_posterior = original["ind"]["post"].sum(axis=(2, 3))
    top = parent_pair_posterior.argmax(axis=1)
    active_tau = np.r_[0, model.tau_grid]
    rate_mu = expit(logit(parent_rates(model))[:, None] + fit.pooled + fit.pair_delta)
    rows = []
    for i, q in enumerate(split.query_ids):
        pi = top[i]
        raw_n, raw_k, current, n, k, n_u, k_u = details[pi]
        d, tau = np.unravel_index(posterior["post"][i, pi].argmax(), posterior["post"].shape[-2:])
        pre_raw_n = raw_n[i, 0] - raw_n[i, tau]
        pre_raw_k = raw_k[i, 0] - raw_k[i, tau]
        s0 = model.params["kappa"]
        cell_rates = (s0 * rate_mu[split.context[i], pi, d] + k[i, tau]) / (s0 + n[i, tau])
        mass = np.r_[pair_posterior[i], posterior["post0"][i]]
        parent_mass = np.r_[parent_pair_posterior[i], original["ind"]["post0"][i]]
        entropy = lambda a: float(-(a * np.log(np.clip(a, 1e-300, 1))).sum())
        rows.append({"query_id": q, "episode_id": split.episode_ids[i],
                     "parent_top_pair": "-".join(map(str, model.pairs[pi])),
                     "parent_top_pair_mass": parent_pair_posterior[i, pi],
                     "E3_top_pair": "-".join(map(str, model.pairs[pair_posterior[i].argmax()])),
                     "E3_top_pair_mass": pair_posterior[i].max(),
                     "parent_pair_entropy": entropy(parent_mass), "E3_pair_entropy": entropy(mass),
                     "E3_flip_mass": posterior["post"][i, :, :, 1:].sum(), "E3_shared_posterior": p_shared[i],
                     "audit_pair_direction": d, "audit_pair_flip_tick": active_tau[tau], "raw_current_cell": current[i],
                     "unavailable_n": n_u[i], "unavailable_k": k_u[i],
                     **{f"raw_pre_{c}_{m}": v for m, values in (("n", pre_raw_n), ("k", pre_raw_k))
                        for c, v in zip(W1.CELL_NAMES, values)},
                     **{f"raw_post_{c}_{m}": v for m, values in (("n", raw_n[i, tau]), ("k", raw_k[i, tau]))
                        for c, v in zip(W1.CELL_NAMES, values)},
                     **{f"canonical_{c}_{m}": v for m, values in (("n", n[i, tau]), ("k", k[i, tau]), ("rate", cell_rates))
                        for c, v in zip(W1.CELL_NAMES, values)},
                     **{f"{name}_p_win": values[i] for name, values in wins.items()}})
    W1.write_rows(OUT / "validation_cell_diagnostics.csv", tuple(rows[0]), rows)
    np.savez_compressed(OUT / "validation_hypothesis_grid.npz", query_id=np.asarray(split.query_ids),
                        pair=np.array(model.pairs), direction=DIRECTION, tau=active_tau,
                        valid_tau=np.isfinite(assembled["log_tau_prior"]), posterior=posterior["post"].astype(np.float32),
                        null=posterior["post0"], posterior_win=posterior["p_win"], p_shared=p_shared)


def synthetic():
    result = {}
    for ordered in (False, True):
        fitting = W2.generate("flip", episodes=24, seed=38102, lead=20, four_cells=ordered).split
        model = W1.refit_m3(fitting, tune=False)
        rates = fit_rates(model, fitting, A_XOR, model._infer(fitting))
        for scenario, lead in (("stable", 16), ("flip", 16)):
            fixture = W2.generate(scenario, episodes=12, seed=38104, lead=lead, four_cells=ordered)
            for older in (24, None):
                split = fixture.split if older is None else W2.mask_history(fixture.split, older)
                original = model._infer(split)
                wins, _, posterior, _, _ = forecasts(model, split, rates, original=original)
                all_wins = {"M3": original["p_win"], **wins}
                record = {}
                for arm, p in all_wins.items():
                    post = original["ind"]["post"] if arm in ("M3", "E3Frozen") else posterior["post"]
                    pair = np.asarray(model.pairs)[post.sum(axis=(2, 3)).argmax(axis=1)]
                    direction = post.sum(axis=(1, 3)).argmax(axis=1)
                    tick = np.r_[0, model.tau_grid][post.sum(axis=(1, 2)).argmax(axis=1)]
                    truth = split.targets
                    pred = W1.combine(p, W1.conditional_mix(truth))
                    item = {"p0_mse": float(np.square(pred[:, 0] - truth[:, 0]).mean()),
                            "five_class_sse": float(np.square(pred - truth).sum(axis=1).mean())}
                    if arm != "E3Shared":  # The shared forecast has a separate mixture, without a MAP grid export.
                        item.update(pair_recovery=float(np.all(pair == fixture.pairs, axis=1).mean()),
                                    direction_recovery=float((direction == fixture.polarity).mean()),
                                    map_flip_share=float((tick > 0).mean()))
                        if scenario == "flip":
                            item["map_flip_within_4"] = float(((tick > 0) & (np.abs(tick - fixture.change) <= 4)).mean())
                    record[arm] = item
                result[f"{'ordered' if ordered else 'parity'}_{scenario}_older_{older}"] = record
    return {"fit_seed": 38102, "held_seed": 38104, "rows_per_case": 96,
            "fixture_scope": "Repository four_cells=True adds raw-bit offsets in both segments; chronological XOR law is checked separately by exhaustive enumeration.",
            "cases": result}


def main():
    trace("four_cells")
    started = perf_counter()
    control = json.loads((OUT / "control_replay.json").read_text())
    assert max(control[s][a]["archive_delta"] for s in ("validation", "test") for a in ("W1a", "W1b", "W2")) < 1e-12
    W1.write_json(OUT / "prespecified_settings.json", SPEC)
    train = W1.labeled_input(DATA)
    checks = {"likelihood_enumeration_pre_grouped": likelihood_check()}
    metrics = grouped(train)
    model = W1.refit_m3(train, tune=True)
    fit = fit_rates(model, train, A_XOR, model._infer(train))
    checks.update(causality(model, train, fit))
    W1.write_json(OUT / "invariance_checks.json", checks)
    rate_path = OUT / "train_rates.npz"
    np.savez_compressed(rate_path, pairs=np.array(model.pairs), direction=DIRECTION, tau=np.r_[0, model.tau_grid],
                        pool_logit_offset=fit.pooled, pair_logit_offset=fit.pair_delta,
                        n_by_context_pair_direction_cell=fit.fit_n, k_by_context_pair_direction_cell=fit.fit_k,
                        parent_mu=parent_rates(model),
                        mu=expit(logit(parent_rates(model))[:, None] + fit.pooled + fit.pair_delta),
                        unavailable_mu=model.win_prior, S0=model.params["kappa"], pair_prior=model.pair_prior,
                        tau_q=model.tau_q, p_flip=model.p_flip, null_weight=model.null_weight,
                        min_segment=model.min_segment, flip_xor=A_XOR)
    grouped_path = OUT / "train_grouped_predictions.csv"
    W1.write_json(OUT / "frozen_fit.json", {"scope": "new win model fit only on main train histories and labels",
                                           "source_sha256": W1.sha256(Path(__file__)),
                                           "grouped_forecast_sha256": W1.sha256(grouped_path),
                                           "train_rates_sha256": W1.sha256(rate_path),
                                           "production_settings": SPEC, "kappa": model.params["kappa"],
                                           "m3_train_kappa_selection": model.selection})
    print("Froze rates; grouped full-arm metrics:", metrics["E3Full"], flush=True)
    files = {}
    for name in ("validation", "test"):
        files.update(emit(name, model, fit))
    W1.write_json(OUT / "forecast_freeze.json", {"fit_sha256": W1.sha256(OUT / "frozen_fit.json"),
                                                "forecast_sha256": files, "runtime_seconds": perf_counter() - started})
    print("Wrote frozen validation and test forecasts", flush=True)
    W1.write_json(OUT / "synthetic_scores.json", synthetic())


if __name__ == "__main__":
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
