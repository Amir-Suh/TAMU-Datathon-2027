"""Train three paired seeds of matched masked GRU and causal TCN residuals."""
import argparse
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
EXPERIMENTS = KIT / "signal_experiments"
PUBLIC = KIT / "public"
FOLDS = EXPERIMENTS / "folds.csv"
BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair")]

from models.common.crossfit import subset_split
from models.common.data import MISSING, PAYOUT_INDEX
from run_repaired_comparison import input_only
from signal_experiments.score import compare, sha256
from slot_forecasting.io import read_rows, read_submission, require_ids, write_json, write_rows, write_submission
from slot_forecasting.schema import PAYOUTS
from infer import distribution, residual
from structure import Frames, SIZE_DECAY, SIZE_STRENGTH, SWITCH, endpoints, frame_inputs, size_prior

SEEDS = (101, 202, 303)
EPOCHS = 8
BATCH = 48
ARMS = {(architecture, evidence): f"{architecture}_{'pair_evidence' if evidence else 'raw'}"
        for architecture in ("gru", "tcn") for evidence in (False, True)}
NAMES = tuple(ARMS.values())


def load_training():
    train = input_only(PUBLIC, "train")
    labels = {row["query_id"]: PAYOUT_INDEX[int(row["outcome"])] for row in
              read_rows(PUBLIC / "train/labels.csv", ("query_id", "outcome"))}
    require_ids(labels, train.query_ids)
    return replace(train, labels=np.array([labels[key] for key in train.query_ids], dtype=np.int8))


def priors(train):
    # The frozen upstream population estimate sees only its fold's histories.
    history = replace(train, labels=np.full(train.n, MISSING, dtype=np.int8))
    w1 = importlib.import_module("signal_experiments.research_next.W1.run")
    return w1.refit_m3(history, tune=False), size_prior(history)


def make_training_frames(train):
    rows = [row for row in read_rows(FOLDS, ("split", "episode_id", "fold")) if row["split"] == "train"]
    episode_folds = {row["episode_id"]: int(row["fold"]) for row in rows}
    if len(episode_folds) != len(rows):
        raise ValueError("Duplicate train episode in fold manifest.")
    require_ids(episode_folds, train.episode_ids)
    fold_ids = np.array([episode_folds[ep] for ep in train.episode_ids])
    values = {}
    for fold in sorted(set(fold_ids)):
        held = np.flatnonzero(fold_ids == fold)
        model, size = priors(subset_split(train, np.flatnonzero(fold_ids != fold)))
        # The full shape is kept to align all pre-outcome rows across folds.
        frames = frame_inputs(subset_split(train, held), model, size, include_cutoff_labels=True)
        for name, arr in vars(frames).items():
            if name not in values:
                values[name] = np.empty((train.n, *arr.shape[1:]), dtype=arr.dtype)
            values[name][held] = arr
        print(f"N1 train feature fold {fold} complete", flush=True)
    return Frames(**values)


def check_causality(validation, model, size):
    sample = subset_split(validation, np.arange(16))
    neutral = replace(sample, signal=np.zeros(sample.n, dtype=int))
    before = frame_inputs(neutral, model, size)
    cutoff = sample.cutoff - 12
    future = np.arange(sample.X.shape[1])[None, :] >= cutoff[:, None]
    changed = np.where(future, 4, sample.X)
    # A historical endpoint has no real-time cutoff signal.
    shortened = replace(neutral, cutoff=cutoff, X=changed,
                        labels=np.full(sample.n, 4), targets=np.zeros((sample.n, 5)))
    after = frame_inputs(shortened, model, size)
    through_endpoint = np.arange(sample.X.shape[1])[None, :] <= cutoff[:, None]
    for name in ("token", "context", "evidence", "win_logit", "size_logit"):
        if not np.array_equal(getattr(before, name)[through_endpoint], getattr(after, name)[through_endpoint]):
            raise ValueError(f"Causal prefix changed in {name}.")
    return before, after, through_endpoint


def train_all(frames, validation, valid_frames, causal_check, upstream, size, out):
    import torch
    from network import export, fit, inputs

    baseline = distribution(*endpoints(valid_frames, validation.cutoff))
    forecasts = {"structural": baseline}
    checkpoints = {f"structural__{name}": getattr(upstream, name)
                   for name in ("pair_prior", "null_weight", "hi", "lo", "win_prior")}
    checkpoints["structural__size"] = size
    fits = {}
    for (architecture, evidence), name in ARMS.items():
        runs = []
        fits[name] = {}
        for seed in SEEDS:
            start = perf_counter()
            model, trace = fit(frames, architecture, evidence, seed, EPOCHS, BATCH)
            weights = export(model)
            prefix = f"{name}_{seed}__"
            checkpoints.update({prefix + key: array for key, array in weights.items()})
            before, after, through = causal_check
            prediction_before = residual(before, weights, architecture, evidence)
            prediction_after = residual(after, weights, architecture, evidence)
            causal_error = float(np.max(np.abs(prediction_before[through] - prediction_after[through])))
            if causal_error > 1e-7:
                raise ValueError(f"{name}/{seed}: network has a future dependence of {causal_error}.")
            with torch.no_grad():
                expected = model(*inputs(before, evidence)).numpy()
            parity = float(np.max(np.abs(prediction_before - expected)))
            if parity > 2e-6:
                raise ValueError(f"{name}/{seed}: Torch and NumPy disagree by {parity} in logits.")
            start_inference = perf_counter()
            res = residual(valid_frames, weights, architecture, evidence)
            pred = distribution(*endpoints(valid_frames, validation.cutoff),
                                res[np.arange(validation.n), validation.cutoff])
            runs.append(pred)
            forecasts[f"{name}_seed{seed}"] = pred
            print(f"{name}/{seed}: {perf_counter() - start:.2f}s train + checks, "
                  f"{perf_counter() - start_inference:.3f}s inference, NLL {trace[-1]:.6f}", flush=True)
            fits[name][str(seed)] = {"parameters": sum(p.numel() for p in model.parameters()),
                                     "train_nll_trace": trace, "future_perturbation_max": causal_error,
                                     "torch_numpy_logit_max_abs_difference": parity}
        forecasts[name] = np.mean(runs, axis=0)

    np.savez_compressed(out / "weights.npz", **checkpoints)
    return forecasts, fits


def replay(validation, frames, path):
    forecasts = {"structural": distribution(*endpoints(frames, validation.cutoff))}
    with np.load(path, allow_pickle=False) as archive:
        for (architecture, evidence), name in ARMS.items():
            seed_predictions = []
            for seed in SEEDS:
                prefix = f"{name}_{seed}__"
                weights = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                adjustment = residual(frames, weights, architecture, evidence)
                p = distribution(*endpoints(frames, validation.cutoff),
                                 adjustment[np.arange(validation.n), validation.cutoff])
                forecasts[f"{name}_seed{seed}"] = p
                seed_predictions.append(p)
            forecasts[name] = np.mean(seed_predictions, axis=0)
    return forecasts


def scores(validation, valid_frames, forecasts, fits, paths):
    ceiling = importlib.import_module("signal_experiments.ceiling_analysis.analyze")
    points = json.loads((PUBLIC / "metadata.json").read_text())["max_points"]
    paired = {"vs_hybrid": compare(PUBLIC / "validation", BASE, paths, points),
              "vs_structural": compare(PUBLIC / "validation", paths["structural"],
                                       {n: paths[n] for n in NAMES}, points)}
    for architecture in ("gru", "tcn"):
        raw, feature = ARMS[architecture, False], ARMS[architecture, True]
        paired[f"{architecture}_evidence_vs_raw"] = compare(
            PUBLIC / "validation", paths[raw], {feature: paths[feature]}, points)
    for evidence, label in ((False, "raw"), (True, "pair_evidence")):
        gru, tcn = ARMS["gru", evidence], ARMS["tcn", evidence]
        paired[f"tcn_vs_gru_{label}"] = compare(
            PUBLIC / "validation", paths[gru], {tcn: paths[tcn]}, points)
    for report in paired.values():
        report["data"] = "public/validation"
        for model in report["models"].values():
            model["path"] = Path(model["path"]).name

    split = ceiling.load_split(PUBLIC, "validation")
    if split.query_ids != validation.query_ids:
        raise ValueError("Scoring slices disagree with forecast query order.")
    hybrid = read_submission(BASE, split.query_ids)
    hybrid_array = np.array([hybrid[key] for key in split.query_ids])
    pair, _, _, _ = ceiling.pair_diagnostics(split)
    ref_loss = np.square(split.reference - split.targets).sum(axis=1)
    weights, _ = ceiling.weights(split.episode_ids, split.conditions, ref_loss)
    support = (validation.X >= 0).sum(axis=1)
    confidence = valid_frames.evidence[np.arange(split.n), split.cutoff, 0]
    groups = {
        "history": {"64_observed": support == 64, "more_than_64": support > 64},
        "pair_confidence": {"<0.7": confidence < .7, "0.7_to_0.9": (confidence >= .7) & (confidence < .9),
                            "at_least_0.9": confidence >= .9},
        "pair": {"inferred_4_7": (pair == [4, 7]).all(axis=1)},
    }
    metrics = []
    intervals_by_slice = {}
    true_win = 1 - split.targets[:, 0]
    true_mix = split.targets[:, 1:] / true_win[:, None]
    raw_loss = np.square(hybrid_array - split.targets).sum(axis=1)
    for group, slices in groups.items():
        for label, mask in slices.items():
            if not mask.any():
                continue
            diff = {name: (np.square(forecasts[name] - split.targets).sum(axis=1) - raw_loss) * mask
                    for name in ("structural", *NAMES)}
            intervals_by_slice[f"{group}/{label}"] = ceiling.paired_intervals(
                split, ref_loss, diff, np.zeros(split.n))
            for name in ("hybrid", "structural", *NAMES):
                p = hybrid_array if name == "hybrid" else forecasts[name]
                mix = p[:, 1:] / (1 - p[:, :1])
                size_error = np.square(mix - true_mix).sum(axis=1) * true_win**2
                sq = np.square(p - split.targets)
                metrics.append(dict(group=group, slice=label, model=name, queries=int(mask.sum()),
                                    episodes=len(set(split.episode_ids[mask])), p0_mse=float(sq[mask, 0].mean()),
                                    win_weighted_payout_sse=float(size_error[mask].mean()),
                                    delta_R_contribution_vs_hybrid=float(weights[mask] @
                                                                         (sq.sum(axis=1) - raw_loss)[mask])))
    return {"paired": paired, "slice_delta_R_vs_hybrid": intervals_by_slice, "fits": fits}, metrics


def manifest():
    import torch
    w1 = importlib.import_module("signal_experiments.research_next.W1.run")
    ceiling = importlib.import_module("signal_experiments.ceiling_analysis.analyze")
    git = {}
    for name, args in {"branch": ["branch", "--show-current"],
                       "commit": ["rev-parse", "--verify", "HEAD"]}.items():
        git[name] = subprocess.run(["git", *args], cwd=KIT, text=True, capture_output=True).stdout.strip() or None
    source = [HERE / name for name in ("run.py", "infer.py", "structure.py", "network.py")]
    source += [Path(w1.__file__), Path(ceiling.__file__), ceiling.FROZEN / "models.joblib", EXPERIMENTS / "score.py"]
    data = [PUBLIC / split / name for split, names in {
        "train": ("queries.csv", "history.csv", "labels.csv"),
        "validation": ("queries.csv", "history.csv", "targets.csv", "reference.csv", "conditions.csv"),
    }.items() for name in names]
    meta = json.loads((PUBLIC / "metadata.json").read_text())
    return {
        "experiment": "N1", "round": meta["round"], "season_id": meta["season_id"], "git": git,
        "scope": "Main-validation development with neural training exclusively on train episodes.",
        "payout_order": list(PAYOUTS), "seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH,
        "pair_switch_probability": SWITCH, "size_decay_per_tick": SIZE_DECAY, "size_prior_strength": SIZE_STRENGTH,
        "arms": list(NAMES), "cpu_threads": 2,
        "bootstrap": {"draws": 10000, "seed": 20261107, "unit": "condition-stratified episode"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__},
        "input_sha256": {str(path.relative_to(KIT)): sha256(path) for path in
                         [*source, *data, BASE, FOLDS, PUBLIC / "metadata.json"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--replay", type=Path, help="Export forecast CSVs from this NPZ without neural fitting.")
    args = parser.parse_args()
    start = perf_counter()
    if not args.replay:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        train = load_training()
    if args.smoke:
        from network import fit
        train = subset_split(train, np.flatnonzero(np.isin(train.episode_ids, sorted(set(train.episode_ids))[:12])))
        upstream, size = priors(train)
        frames = frame_inputs(train, upstream, size, include_cutoff_labels=True)
        for architecture in ("gru", "tcn"):
            t0 = perf_counter()
            model, trace = fit(frames, architecture, True, SEEDS[0], 1, BATCH)
            print(architecture, "seconds", perf_counter() - t0, "NLL", trace[-1])
        return

    validation = input_only(PUBLIC, "validation")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as archive:
            upstream = SimpleNamespace(**{
                name: archive[f"structural__{name}"]
                for name in ("pair_prior", "null_weight", "hi", "lo", "win_prior")
            })
            size = archive["structural__size"]
        valid_frames = frame_inputs(validation, upstream, size)
        forecasts = replay(validation, valid_frames, args.replay)
    else:
        upstream, size = priors(train)
        valid_frames = frame_inputs(validation, upstream, size)
        causal_check = check_causality(validation, upstream, size)
        frames = make_training_frames(train)
        forecasts, fits = train_all(frames, validation, valid_frames, causal_check, upstream, size, args.out)
        restored = replay(validation, valid_frames, args.out / "weights.npz")
        if any(not np.array_equal(restored[name], p) for name, p in forecasts.items()):
            raise ValueError("Exported weights failed bitwise inference replay.")

    paths = {}
    for name, p in forecasts.items():
        paths[name] = args.out / f"{name}.csv"
        write_submission(paths[name], dict(zip(validation.query_ids, p)))
    if not args.replay:
        results, diagnostics = scores(validation, valid_frames, forecasts, fits, paths)
        write_json(args.out / "scores.json", results)
        write_rows(args.out / "slices.csv", tuple(diagnostics[0]), diagnostics)
        write_json(args.out / "plan.json", manifest())
    print(f"N1 complete in {perf_counter() - start:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
