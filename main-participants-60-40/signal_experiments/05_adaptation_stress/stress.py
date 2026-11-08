"""Evaluate lag-prior transfer and sensitivity to fewer observed recent ticks."""

import argparse
from copy import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
REPAIR = KIT / "m5_m6_repair"
MAIN = REPAIR / "results/repaired-m5-m6"
sys.path[:0] = [str(KIT), str(REPAIR), str(HERE.parent)]

from models.common.data import MISSING, load_split
from models.m3_lag_pair_flip.model import FlipLagPairModel
from models.m4_lag_pair_flip_shared.model import SharedFlipLagPairModel
from run_repaired_comparison import clean, input_only
from score import compare, sha256
from slot_forecasting.io import read_submission, write_json, write_rows, write_submission

OLDER_TICKS = 24
RECENT_WINDOW = 40
MASK_RECENT = (16, 8)


def hash_key(value):
    return hashlib.sha256(value.encode()).digest()


def select_episodes(split, limit, seed):
    if limit is None:
        return split
    episodes = sorted(set(split.episode_ids), key=lambda key: hash_key(f"{seed}:{key}"))
    if not 1 <= limit <= len(episodes):
        raise ValueError(f"Fit episode limit must be between 1 and {len(episodes)}.")
    keep = np.isin(split.episode_ids, episodes[:limit])
    return replace(
        split,
        query_ids=np.asarray(split.query_ids)[keep].tolist(),
        episode_ids=split.episode_ids[keep],
        machine_ids=np.asarray(split.machine_ids)[keep].tolist(),
        machine_type=split.machine_type[keep],
        signal=split.signal[keep],
        cutoff=split.cutoff[keep],
        X=split.X[keep],
        labels=None if split.labels is None else split.labels[keep],
        conditions=None, targets=None, reference=None,
    )


def sparse_prefix(split, recent):
    """Keep observed original ticks and leave the prediction cutoff untouched."""
    masked = np.full_like(split.X, MISSING)
    for row, (key, cutoff) in enumerate(zip(split.query_ids, split.cutoff)):
        older = np.flatnonzero(split.X[row, : max(0, cutoff - RECENT_WINDOW)] != MISSING)
        older = sorted(older, key=lambda t: hash_key(f"adaptation-mask-v1:{key}:{t}"))[:OLDER_TICKS]
        start = max(0, cutoff - recent)
        masked[row, older] = split.X[row, older]
        masked[row, start:cutoff] = split.X[row, start:cutoff]
    return replace(split, X=masked)


def with_pair_prior(fitted, prior):
    replacement = copy(fitted)
    replacement.pair_prior = np.asarray(prior).copy()
    return replacement


def pair_prevalence(fitted, split):
    """Summarize prefix-inferred pairs under a uniform 28-pair prior."""
    uniform = with_pair_prior(fitted, np.full(len(fitted.pairs), 1 / len(fitted.pairs)))
    result = uniform._infer(split)
    post = result["ind"]["post"].sum(axis=(2, 3))
    map_index = post.argmax(axis=1)
    confidence = post.max(axis=1)
    credible = confidence >= 0.90
    rows = []
    for i, (lag_a, lag_b) in enumerate(fitted.pairs):
        available = ((split.X[:, lag_b:] != MISSING)
                     & (split.X[:, lag_b-lag_a:-lag_a] != MISSING)
                     & (split.X[:, :-lag_b] != MISSING)).sum(axis=1)
        rows.append({
            "lag_a": lag_a, "lag_b": lag_b,
            "expected_pattern_count": float(post[:, i].sum()),
            "map_count": int((map_index == i).sum()),
            "confident_count": int(((map_index == i) & credible).sum()),
            "mean_available_prefix_rows": float(available.mean()),
        })
    return rows, float(np.mean(result["ind"]["post0"]))


def write_forecast(out, name, split, pred):
    path = out / f"{name}.csv"
    write_submission(path, dict(zip(split.query_ids, pred)))
    return path


def synthetic_pair_shift(template, pairs, *, seed):
    """Independent generated binary lag parity, with an exact positive payout law."""
    rng = np.random.default_rng(seed)
    X = np.full_like(template.X, MISSING)
    labels = np.empty(template.n, dtype=np.int8)
    chosen_pairs = rng.integers(len(pairs), size=template.n)
    size_law = np.array([[0.55, 0.31, 0.125, 0.015],
                         [0.39, 0.33, 0.24, 0.04],
                         [0.27, 0.29, 0.31, 0.13]])
    truth = np.empty((template.n, 5))
    for row, cutoff in enumerate(template.cutoff):
        a, b = pairs[chosen_pairs[row]]
        positive = size_law[template.machine_type[row]]
        low = 0.07 + 0.02 * template.signal[row]
        high = 0.87 + 0.02 * template.signal[row]
        win_on_match = rng.integers(2) == 1
        for tick in range(cutoff + 1):
            win_prob = 0.45
            if tick >= b:
                lags_match = (X[row, tick - a] > 0) == (X[row, tick - b] > 0)
                win_prob = high if lags_match == win_on_match else low
            payout = 1 + rng.choice(4, p=positive) if rng.random() < win_prob else 0
            if tick < cutoff:
                X[row, tick] = payout
            else:
                labels[row] = payout
                truth[row] = np.r_[1 - win_prob, win_prob * positive]
    return replace(template, X=X, labels=labels, conditions=None, targets=None, reference=None), truth, chosen_pairs


def exact_synthetic_scores(validation, truth, forecasts, *, draws=10000):
    rng = np.random.default_rng(802)
    episodes, group, counts = np.unique(validation.episode_ids, return_inverse=True, return_counts=True)
    resamples = rng.integers(len(episodes), size=(draws, len(episodes)))
    episode_loss, output = {}, {}
    for name, pred in forecasts.items():
        row_loss = np.square(pred - truth).sum(axis=1)
        episode_loss[name] = np.bincount(group, weights=row_loss) / counts
        output[name] = {
            "distribution_sse": float(episode_loss[name].mean()),
            "p0_mse": float(np.square(pred[:, 0] - truth[:, 0]).mean()),
        }
    delta = episode_loss["target_refit_prior"] - episode_loss["carry_main_prior"]
    output["paired_refit_minus_carry"] = {
        "distribution_sse_delta": float(delta.mean()),
        "paired_ci95": np.quantile(delta[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
    }
    return output


def run_synthetic(out, main_m3, train_template, validation_template):
    print("Fitting a synthetic rare-pair shift", flush=True)
    index = np.argsort(main_m3.pair_prior, kind="stable")[:4]
    new_pairs = [main_m3.pairs[i] for i in index]
    templates = (select_episodes(train_template, min(24, len(set(train_template.episode_ids))), "synthetic-train-v1"),
                 select_episodes(validation_template, min(24, len(set(validation_template.episode_ids))), "synthetic-val-v1"))
    generated_train, _, train_pairs = synthetic_pair_shift(templates[0], new_pairs, seed=73013)
    generated_val, truth, val_pairs = synthetic_pair_shift(templates[1], new_pairs, seed=73014)
    fitted = FlipLagPairModel().fit(generated_train)
    carry = with_pair_prior(fitted, main_m3.pair_prior)
    eval_splits = {"normal": generated_val}
    for recent in MASK_RECENT:
        eval_splits[f"sparse{recent}"] = sparse_prefix(generated_val, recent)
    metrics = {}
    for name, split in eval_splits.items():
        forecasts = {"target_refit_prior": fitted.predict(split), "carry_main_prior": carry.predict(split)}
        metrics[name] = exact_synthetic_scores(split, truth, forecasts)
    write_json(out / "synthetic_rare_pair_shift.json", {
        "scope": "Generated stable histories and exact target law, a deliberate shift to the four least-weighted main pairs.",
        "fit_episodes": len(set(generated_train.episode_ids)),
        "validation_episodes": len(set(generated_val.episode_ids)),
        "pairs": [
            dict(lag_a=a, lag_b=b, train_machines=int((train_pairs == j).sum()),
                 validation_machines=int((val_pairs == j).sum()), main_prior=float(main_m3.pair_prior[i]),
                 refit_prior=float(fitted.pair_prior[i]))
            for j, ((a, b), i) in enumerate(zip(new_pairs, index))
        ],
        "fitted_kappa": fitted.params["kappa"],
        "metrics": metrics,
    })


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.data / "metadata.json").read_text())
    points = args.max_points if args.max_points is not None else metadata["max_points"]
    main_models = joblib.load(args.main_checkpoint)
    main_m3, main_m4 = main_models["m3"], main_models["m4"]
    train = load_split(args.data, "train")
    validation = input_only(args.data, "validation")
    fit_train = select_episodes(train, args.fit_episodes, "prior-transfer-v1")
    if set(train.episode_ids) & set(validation.episode_ids):
        raise ValueError("Target train and validation episode IDs overlap.")
    if fit_train.labels is None:
        raise ValueError("Target train cutoff labels are required for model selection.")

    print(f"Refitting M3 on {len(set(fit_train.episode_ids))} target train episodes", flush=True)
    refit = FlipLagPairModel().fit(fit_train)
    if refit.pairs != main_m3.pairs:
        raise ValueError("Frozen and refit lag grids differ.")
    carry = with_pair_prior(refit, main_m3.pair_prior)
    if metadata["round"] == "main":
        stress_models = {"m3": main_m3, "m4": main_m4}
        stress_scope = "Frozen main M3 and M4 checkpoint replay."
    else:
        print("Refitting shared-flip M4 on the same target train episodes", flush=True)
        stress_models = {"m3": refit, "m4": SharedFlipLagPairModel().fit(fit_train)}
        stress_scope = "M3 and M4 refitted on the same target-round train episodes."
    print("Freezing validation predictions on normal and masked inputs", flush=True)
    forecast_files = {"carry_main_prior": write_forecast(args.out, "carry_main_prior", validation, carry.predict(validation)),
                      "target_refit_prior": write_forecast(args.out, "target_refit_prior", validation, refit.predict(validation))}
    frozen = {}
    mask_summary = {}
    eval_splits = {"normal": validation}
    for recent in MASK_RECENT:
        eval_splits[f"sparse{recent}"] = sparse_prefix(validation, recent)
    for name, split in eval_splits.items():
        frozen[name] = {
            label: write_forecast(args.out, f"{label}_{name}", split, model.predict(split))
            for label, model in stress_models.items()
        }
        counts = (split.X != MISSING).sum(axis=1)
        last_eight = [int((row[max(0, c-8):c] != MISSING).sum()) for row, c in zip(split.X, split.cutoff)]
        mask_summary[name] = {"min": int(counts.min()), "median": float(np.median(counts)),
                              "max": int(counts.max()), "min_observed_last_8": min(last_eight)}
    if metadata["round"] == "main":
        replay_errors = {}
        for name in ("m3", "m4"):
            published_path = args.main_checkpoint.parent / name / "validation_predictions.csv"
            if not published_path.exists():
                continue
            old = read_submission(published_path, validation.query_ids)
            new = read_submission(frozen["normal"][name], validation.query_ids)
            replay_errors[name] = float(np.max(np.abs(np.array(list(old.values())) - np.array([new[k] for k in old]))))
            if replay_errors[name] > 1e-8:
                raise ValueError(f"Frozen main {name} replay does not match the published main validation forecasts.")
    else:
        replay_errors = {}
    write_json(args.out / "forecast_freeze.json", {
        "round": metadata["round"], "points_max": points,
        "fit_episodes": sorted(set(fit_train.episode_ids)),
        "target_fitted_parameters": clean(refit.describe()),
        "m3_m4_stress_scope": stress_scope,
        "m3_m4_stress_parameters": {name: clean(model.describe()) for name, model in stress_models.items()},
        "fixed_checkpoint": {"path": str(args.main_checkpoint), "sha256": sha256(args.main_checkpoint)},
        "main_normal_replay_max_difference": replay_errors,
        "mask_observed_ticks_per_query": mask_summary,
        "forecasts": {path.name: sha256(path) for path in sorted(args.out.glob("*.csv"))},
        "fit_input_sha256": {
            f"{split}/{filename}": sha256(args.data / split / filename)
            for split, names in (("train", ("queries.csv", "history.csv", "labels.csv")),
                                 ("validation", ("queries.csv", "history.csv")))
            for filename in names
        },
    })
    if args.synthetic:
        run_synthetic(args.out, main_m3, train, validation)

    print("Scoring paired forecasts against fixed validation targets", flush=True)
    validation_dir = args.data / "validation"
    write_json(args.out / "prior_transfer.json", compare(validation_dir, forecast_files["carry_main_prior"],
                                                       {"target_refit_prior": forecast_files["target_refit_prior"]}, points))
    for name, files in frozen.items():
        write_json(args.out / f"m3_m4_{name}.json", compare(validation_dir, files["m3"], {"m4": files["m4"]}, points))
    for model in ("m3", "m4"):
        write_json(args.out / f"{model}_mask_penalty.json", compare(
            validation_dir, frozen["normal"][model],
            {f"{model}_sparse{n}": frozen[f"sparse{n}"][model] for n in MASK_RECENT}, points,
        ))
    rows = {}
    for label, split in (("target_train", fit_train), ("target_validation", validation)):
        rows[label], null = pair_prevalence(refit, split)
        write_json(args.out / f"{label}_pair_summary.json", {"queries": split.n, "mean_null_posterior": null})
    output = []
    for i, pair in enumerate(refit.pairs):
        output.append({
            "lag_a": pair[0], "lag_b": pair[1], "main_prior": main_m3.pair_prior[i], "target_prior": refit.pair_prior[i],
            **{f"{label}_{key}": value for label, source in rows.items()
               for key, value in source[i].items() if key not in {"lag_a", "lag_b"}},
        })
    write_rows(args.out / "pair_prevalence.csv", tuple(output[0]), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=KIT / "public", help="Target round's public directory")
    parser.add_argument("--max-points", type=float, help="Defaults to the target kit's metadata")
    parser.add_argument("--fit-episodes", type=int, help="Reduce target train by a deterministic episode hash")
    parser.add_argument("--main-checkpoint", type=Path, default=MAIN / "models.joblib")
    parser.add_argument("--synthetic", action="store_true", help="Also run a generated rare-pair shift")
    parser.add_argument("--out", type=Path, required=True)
    with threadpool_limits(limits=2):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
