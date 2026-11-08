"""Fit or replay the E6 main-development proxy without accessing held answers."""
import argparse
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]
EXP = KIT / "signal_experiments"
PUBLIC = KIT / "public"
FOLDS = EXP / "folds.csv"
N1 = EXP / "research_next/N1"
FREDY = EXP / "01_fredy_replay"
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair"), str(N1)]

from models.common.crossfit import subset_split
from models.common.data import PAYOUT_INDEX
from run_repaired_comparison import input_only
from slot_forecasting.io import read_rows, read_submission, require_ids, write_json, write_submission
from structure import Frames, endpoints, frame_inputs, size_prior
from infer import distribution, residual
from signal_experiments.research_next.synthetic import mask_history

SEEDS = (101, 202, 303)
ARMS = ("factored", "signal_offset", "signal_offset_mask")
STRUCTURE_KEYS = ("pair_prior", "null_weight", "hi", "lo", "win_prior")
HISTORICAL_MASK_TICKS = (128, 192)
MASK_RECENT = 40
MASK_OLDER = 24
BATCH = 48
EPOCHS = 8
CUTOFF_WEIGHT = 8
WIN_PENALTY = .01


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_training():
    train = input_only(PUBLIC, "train")
    labels = {r["query_id"]: PAYOUT_INDEX[int(r["outcome"])] for r in
              read_rows(PUBLIC / "train/labels.csv", ("query_id", "outcome"))}
    require_ids(labels, train.query_ids)
    return replace(train, labels=np.array([labels[key] for key in train.query_ids], dtype=np.int8))


def priors(train):
    history = replace(train, labels=np.full(train.n, -1, dtype=np.int8))
    w1 = importlib.import_module("signal_experiments.research_next.W1.run")
    return w1.refit_m3(history, tune=False), size_prior(history)


def eligible_prefixes(split):
    times = np.arange(split.X.shape[1])[None, :]
    older = times < split.cutoff[:, None] - MASK_RECENT
    recent = ~older & (times < split.cutoff[:, None])
    seen = split.X >= 0
    return ((older & seen).sum(axis=1) > MASK_OLDER) & ((recent & seen).sum(axis=1) == MASK_RECENT)


def train_masks(train):
    selections, manifest = {}, {}
    for tick in (*HISTORICAL_MASK_TICKS, None):
        cutoff = train.cutoff if tick is None else np.full(train.n, tick)
        candidate = replace(train, cutoff=cutoff)
        eligible = eligible_prefixes(candidate)
        if tick is not None and tick < train.X.shape[1]:
            eligible &= (cutoff < train.cutoff) & (train.X[:, tick] >= 0)
        elif tick is not None:
            eligible[:] = False
        ep = sorted(set(train.episode_ids), key=lambda key: sha256(f"E6-mask-v1:{tick}:{key}".encode()).digest())
        grouped = [key for key in ep if eligible[train.episode_ids == key].all()]
        selected = grouped[:len(grouped) // 4]
        label = "actual" if tick is None else str(tick)
        manifest[label] = dict(eligible=grouped, masked=selected,
                               selected_fraction=len(selected) / len(grouped) if grouped else None)
        selections[tick] = np.flatnonzero(np.isin(train.episode_ids, selected))
    return selections, manifest


def cat_frames(blocks):
    return Frames(**{name: np.concatenate([getattr(block, name) for block in blocks])
                     for name in vars(blocks[0])})


def training_frames(train, episode_folds):
    mask_selections, mask_manifest = train_masks(train)
    folds = np.array([episode_folds[key] for key in train.episode_ids])
    main_blocks, endpoint_blocks, mask_blocks = [], [], []
    fold_manifest = {}
    for fold in sorted(set(folds)):
        fit = subset_split(train, np.flatnonzero(folds != fold))
        held_indices = np.flatnonzero(folds == fold)
        held = subset_split(train, held_indices)
        model, size = priors(fit)
        base = frame_inputs(held, model, size, include_cutoff_labels=True)
        base_cutoff_weight = np.ones_like(base.win_logit)
        base_cutoff_weight[np.arange(held.n), held.cutoff] = CUTOFF_WEIGHT
        fold_manifest[str(fold)] = {"fit_episodes": sorted(set(fit.episode_ids)),
                                    "held_episodes": sorted(set(held.episode_ids))}
        for tick, selected in mask_selections.items():
            local = np.flatnonzero(np.isin(held_indices, selected))
            if not len(local):
                continue
            prefix = subset_split(held, local)
            if tick is not None:
                prefix = replace(prefix, cutoff=np.full(prefix.n, tick),
                                 labels=prefix.X[:, tick].copy(), signal=np.zeros(prefix.n, dtype=int))
            endpoint_labels = np.full_like(prefix.X, -1)
            rows = np.arange(prefix.n)
            endpoint_labels[rows, prefix.cutoff] = prefix.labels
            before = replace(frame_inputs(prefix, model, size), labels=endpoint_labels)
            masked = mask_history(prefix, MASK_OLDER, recent=MASK_RECENT)
            assert np.all((masked.X >= 0).sum(axis=1) == MASK_OLDER + MASK_RECENT)
            after = replace(frame_inputs(masked, model, size), labels=endpoint_labels)
            weight = np.ones_like(before.win_logit)
            if tick is None:
                weight[rows, prefix.cutoff] = CUTOFF_WEIGHT
            endpoint_blocks.append((before, weight))
            mask_blocks.append((after, weight))
            base.labels[local, prefix.cutoff] = -1
        main_blocks.append((base, base_cutoff_weight))
        print(f"features for fold {fold} ready, {held.n} held machines", flush=True)

    def merge(blocks):
        return cat_frames([frames for frames, _ in blocks]), np.concatenate([weights for _, weights in blocks])

    plain = merge(main_blocks + endpoint_blocks)
    augmented = merge(main_blocks + mask_blocks)
    assert np.array_equal(plain[0].labels, augmented[0].labels)
    assert np.array_equal(plain[1], augmented[1])
    return plain, augmented, fold_manifest, mask_manifest


def inference_frames(split, model, size):
    frames = frame_inputs(split, model, size)
    return frames


def design(frames, use_structure):
    times = np.broadcast_to(np.arange(frames.token.shape[1], dtype=np.float32) / 256, frames.token.shape)
    signals = frames.context[..., 2].astype(np.int64) if use_structure else np.zeros_like(frames.token, dtype=np.int64)
    type_signal = frames.kind[:, None] * 3 + signals + 1
    context = np.stack((frames.context[..., 0], times), axis=-1)
    evidence = frames.evidence if use_structure else np.zeros_like(frames.evidence)
    return frames.token.astype(np.int64), type_signal, context, evidence


def numpy_logits(frames, weights, use_structure):
    _, type_signal, context, evidence = design(frames, use_structure)
    # N1's portable recurrence consumes the joint embedding as context.
    learned_context = np.concatenate((weights["type_signal.weight"][type_signal], context), axis=-1)
    portable = {**weights, "machine.weight": np.empty((3, 0), dtype=np.float32)}
    return residual(replace(frames, context=learned_context, evidence=evidence), portable, "gru", True)


def endpoint_forecast(split, frames, weights=None, use_structure=True):
    if weights is None:
        return distribution(*endpoints(frames, split.cutoff))
    out = []
    for batch in np.array_split(np.arange(split.n), (split.n + 63) // 64):
        local = frames.subset(batch)
        offsets, _ = endpoints(local, split.cutoff[batch])
        logits = numpy_logits(local, weights, use_structure)[np.arange(len(batch)), split.cutoff[batch]]
        win = logits[:, 0] + (offsets if use_structure else 0)
        out.append(distribution(win, logits[:, 1:]))
    return np.concatenate(out)


def offset_counterfactuals(split, model, size):
    return {str(sign): inference_frames(replace(split, signal=np.full(split.n, sign)), model, size)
            for sign in (-1, 1)}


def causality_fixture(train, model, size):
    sample = subset_split(train, np.arange(min(train.n, 16)))
    times = np.arange(sample.X.shape[1])[None, :]
    pseudo = sample.cutoff - 12
    before = inference_frames(sample, model, size)
    future = times >= pseudo[:, None]
    changed = replace(sample, X=np.where(future, (sample.X + 2) % 5, sample.X),
                      signal=np.where(sample.signal == 1, -1, 1), labels=(sample.labels + 1) % 5)
    after = inference_frames(changed, model, size)
    through = times <= pseudo[:, None]
    historical_signal_neutral = np.all(before.context[..., 2][times != sample.cutoff[:, None]] == 0)
    actual_signal_matches = np.array_equal(before.context[np.arange(sample.n), sample.cutoff, 2], sample.signal)
    signal_before_actual = inference_frames(replace(sample, signal=changed.signal), model, size)
    actual_only = times < sample.cutoff[:, None]
    # The alternate signal changes the endpoint without ever changing historical inputs.
    signal_changes_actual = np.all(signal_before_actual.context[np.arange(sample.n), sample.cutoff, 2] != sample.signal)
    fixed_mask_before = mask_history(replace(sample, cutoff=pseudo, signal=np.zeros(sample.n, dtype=int)),
                                     MASK_OLDER, recent=MASK_RECENT)
    fixed_mask_after = mask_history(replace(changed, cutoff=pseudo, signal=np.zeros(sample.n, dtype=int)),
                                    MASK_OLDER, recent=MASK_RECENT)
    assert np.array_equal(fixed_mask_before.X, fixed_mask_after.X)
    actual_changed = replace(sample, X=np.where(times >= sample.cutoff[:, None], 4, sample.X),
                             labels=(sample.labels + 1) % 5)
    actual_after = inference_frames(actual_changed, model, size)
    actual_through = times <= sample.cutoff[:, None]
    masked_before = inference_frames(fixed_mask_before, model, size)
    masked_after = inference_frames(fixed_mask_after, model, size)
    cases = [(before, after, through), (before, signal_before_actual, actual_only),
             (before, actual_after, actual_through), (masked_before, masked_after, through)]
    for left, right, keep in cases:
        for name in ("token", "context", "evidence", "win_logit", "size_logit"):
            assert np.array_equal(getattr(left, name)[keep], getattr(right, name)[keep]), name
    assert historical_signal_neutral and actual_signal_matches and signal_changes_actual
    return cases


def fit(frames, sample_weight, arm, seed, epochs, checks):
    import torch
    from torch import nn
    from torch.nn import functional as F
    use_structure = arm != "factored"

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.payout = nn.Embedding(6, 6)
            self.type_signal = nn.Embedding(9, 8)
            self.encoder = nn.GRU(6 + 8 + 2 + 4, 128, batch_first=True)
            self.head = nn.Linear(128, 5)
            if use_structure:
                with torch.no_grad():
                    self.head.weight[0].zero_()
                    self.head.bias[0].zero_()

        def forward(self, token, type_signal, context, evidence):
            x = torch.cat((self.payout(token), self.type_signal(type_signal), context, evidence), dim=-1)
            return self.head(self.encoder(x)[0])

    torch.manual_seed(seed)
    model = Network()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    inputs = tuple(torch.from_numpy(x) for x in design(frames, use_structure))
    target = torch.from_numpy(frames.labels.astype(np.int64))
    weights = torch.from_numpy(sample_weight)
    offset = torch.from_numpy(frames.win_logit) if use_structure else torch.zeros_like(weights)
    losses = []
    for epoch in range(epochs):
        total, count = 0.0, 0.0
        for rows in torch.randperm(len(target), generator=generator).split(BATCH):
            outcome = target[rows]
            known, won = outcome >= 0, outcome > 0
            last = int(torch.nonzero(known.any(dim=0))[-1]) + 1
            output = model(*(x[rows, :last] for x in inputs))
            outcome, known, won = outcome[:, :last], known[:, :last], won[:, :last]
            w = weights[rows, :last]
            win = offset[rows, :last] + output[..., 0]
            win_loss = F.binary_cross_entropy_with_logits(win[known], won[known].float(), reduction="none")
            size_loss = F.cross_entropy(output[..., 1:][won], outcome[won] - 1, reduction="none")
            n = w[known].sum()
            objective = ((win_loss * w[known]).sum() + (size_loss * w[won]).sum()) / n
            if use_structure:
                objective += WIN_PENALTY * (w[known] * output[..., 0][known].square()).sum() / n
            optimizer.zero_grad()
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += objective.item() * n.item()
            count += n.item()
        losses.append(total / count)
    exported = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    causal_errors = []
    parity_errors = []
    for left, right, keep in checks:
        first = numpy_logits(left, exported, use_structure)
        second = numpy_logits(right, exported, use_structure)
        causal_errors.append(float(np.max(np.abs(first[keep] - second[keep]))))
        with torch.no_grad():
            expected = model(*(torch.from_numpy(x) for x in design(left, use_structure))).numpy()
        parity_errors.append(float(np.max(np.abs(first - expected))))
    assert max(causal_errors) <= 1e-7, causal_errors
    assert max(parity_errors) <= 3e-6, parity_errors
    return exported, {"weighted_training_objective": losses, "future_logit_errors": causal_errors,
                      "torch_numpy_logit_max_abs": max(parity_errors),
                      "parameters": sum(p.numel() for p in model.parameters())}


def conditional_swap(win, size):
    positive = size[:, 1:]
    return np.column_stack((win[:, 0], (1 - win[:, :1]) * positive / positive.sum(axis=1, keepdims=True)))


def archived_forecasts(split):
    is_test = split.name == "test"
    directory = FREDY / ("test_forecasts" if is_test else "forecasts")
    stem = "m3_match_split_win_pair_l31_factor_gru_size" + ("_test" if is_test else "")
    paths = {"hybrid": EXP / "combined" / f"{stem}.csv", "pair_l31": directory / "pair_l31.csv"}
    arrays = {}
    for name, path in paths.items():
        rows = read_submission(path, split.query_ids)
        arrays[name] = np.array([rows[key] for key in split.query_ids])
    return arrays


def write_forecasts(split, forecasts, out, replay_existing=False):
    for name, p in forecasts.items():
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-12)
        filename = f"{split.name}_{name}.csv"
        path = out / filename
        write_submission(path, dict(zip(split.query_ids, p)))
    return {f"{split.name}_{name}.csv": digest(out / f"{split.name}_{name}.csv") for name in forecasts}


def predict(split, model, size, checkpoints, arms, seeds, out):
    frames = inference_frames(split, model, size)
    plain = {"structural": endpoint_forecast(split, frames)}
    force = {"plain": offset_counterfactuals(split, model, size)} if split.name == "validation" else {}
    masked = None
    if split.name == "validation":
        masked_split = mask_history(split, MASK_OLDER, recent=MASK_RECENT)
        masked_frames = inference_frames(masked_split, model, size)
        masked = {"structural_mask24_40": endpoint_forecast(masked_split, masked_frames)}
    archived = archived_forecasts(split)
    # The payout replacement follows the incumbent's equal joint mixture before conditioning.
    # Its win-head replacement holds the original incumbent payout proportions fixed.
    def save_hybrid(name):
        plain[f"{name}_hybrid_size"] = conditional_swap(archived["hybrid"], (archived["pair_l31"] + plain[name]) / 2)
        plain[f"{name}_hybrid_win"] = conditional_swap(plain[name], archived["hybrid"])

    counterfactual = {}
    if force:
        counterfactual["structural"] = np.stack([endpoint_forecast(split, f)[:, 0] for f in force["plain"].values()])
    for arm in arms:
        use_structure = arm != "factored"
        predicted, predicted_mask, predicted_force = [], [], []
        for seed in seeds:
            prefix = f"{arm}_{seed}__"
            weights = {k[len(prefix):]: v for k, v in checkpoints.items() if k.startswith(prefix)}
            p = endpoint_forecast(split, frames, weights, use_structure)
            name = f"{arm}_seed{seed}"
            plain[name] = p
            predicted.append(p)
            if masked is not None:
                masked_p = endpoint_forecast(masked_split, masked_frames, weights, use_structure)
                masked[f"{name}_mask24_40"] = masked_p
                predicted_mask.append(masked_p)
                predicted_force.append(np.stack([endpoint_forecast(split, f, weights, use_structure)[:, 0]
                                                 for f in force["plain"].values()]))
        plain[arm] = np.mean(predicted, axis=0)
        save_hybrid(arm)
        if masked is not None:
            masked[f"{arm}_mask24_40"] = np.mean(predicted_mask, axis=0)
            counterfactual[arm] = np.mean(predicted_force, axis=0)
            counterfactual[f"{arm}_hybrid_win"] = counterfactual[arm]
    if counterfactual:
        np.savez(out / "signal_counterfactuals.npz", query_ids=np.asarray(split.query_ids), **counterfactual)
    return write_forecasts(split, plain | (masked or {}), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.replay:
        assert "torch" not in sys.modules
        # Refuse to import Torch even if a future replay implementation tries it.
        class BlockTorch:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError("Frozen replay cannot import Torch")
        sys.meta_path.insert(0, BlockTorch())
        with np.load(args.replay, allow_pickle=False) as archive:
            checkpoints = dict(archive)
        model = SimpleNamespace(**{name: checkpoints[f"structural__{name}"] for name in STRUCTURE_KEYS})
        size = checkpoints["structural__size"]
        seeds, arms = SEEDS, ARMS
        fits = None
    else:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        train = load_training()
        rows = [r for r in read_rows(FOLDS, ("split", "episode_id", "fold")) if r["split"] == "train"]
        episode_folds = {r["episode_id"]: int(r["fold"]) for r in rows}
        require_ids(episode_folds, train.episode_ids)
        assert len(episode_folds) == len(rows)
        if args.smoke:
            episode_subset = sorted(set(train.episode_ids))[:12]
            train = subset_split(train, np.flatnonzero(np.isin(train.episode_ids, episode_subset)))
        model, size = priors(train)
        checks = causality_fixture(train, model, size)
        plain, augmented, fold_manifest, mask_manifest = training_frames(train, episode_folds)
        checkpoints = {f"structural__{name}": getattr(model, name) for name in STRUCTURE_KEYS}
        checkpoints["structural__size"] = size
        seeds, arms = (SEEDS[:1], ARMS) if args.smoke else (SEEDS, ARMS)
        fits = {}
        fit_start = perf_counter()
        write_json(args.out / "folds_and_masks.json", {"folds": fold_manifest, "masks": mask_manifest})
        for arm in arms:
            source = augmented if arm.endswith("_mask") else plain
            for seed in seeds:
                t0 = perf_counter()
                weights, result = fit(*source, arm, seed, 1 if args.smoke else EPOCHS, checks)
                key = f"{arm}_{seed}"
                result["seconds_train_and_checks"] = perf_counter() - t0
                checkpoints.update({f"{key}__{name}": value for name, value in weights.items()})
                fits[key] = result
                print(f"{key} complete in {result['seconds_train_and_checks']:.2f}s", flush=True)
        np.savez_compressed(args.out / "weights.npz", **checkpoints)
        write_json(args.out / "fit_checks.json", {
            "runs": fits, "episode_count": len(set(train.episode_ids)),
            "training_seconds": perf_counter() - fit_start,
            "perturbations_in_order": ["future_outcomes_and_signal_at_historical_endpoint",
                                       "future_signal_all_historical_ticks",
                                       "actual_cutoff_and_future_outcomes",
                                       "masked_future_outcomes_and_signal"],
            "same_targets_and_weights_for_all_arms": True,
        })
    forecasts = {}
    if not args.smoke:
        for name in (("validation",) if args.validation_only else ("validation", "test")):
            t0 = perf_counter()
            forecasts.update(predict(input_only(PUBLIC, name), model, size, checkpoints, arms, seeds, args.out))
            print(f"{name} inference, all arms, {perf_counter() - t0:.2f}s", flush=True)
        forecasts["signal_counterfactuals.npz"] = digest(args.out / "signal_counterfactuals.npz")
        write_json(args.out / "forecast_freeze.json", forecasts)
    if fits is not None:
        round_manifest = json.loads((HERE.parents[1] / "manifest.json").read_text())
        sources = [Path(__file__), FOLDS, N1 / "infer.py", N1 / "structure.py",
                   EXP / "research_next/W1/run.py", EXP / "research_next/synthetic.py",
                   KIT / "m5_m6_repair/run_repaired_comparison.py"]
        write_json(args.out / "manifest.json", {
            "owner": "neural", "card": "E6", "round": round_manifest["round"],
            "season_id": round_manifest["season_id"],
            "scope": "main development with the N1 proxy lag-structure offset",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__},
            "commands": sys.argv, "seeds": seeds, "arms": arms, "threads": 1,
            "epochs": 1 if args.smoke else EPOCHS, "batch_size": BATCH,
            "actual_cutoff_supervision_weight": CUTOFF_WEIGHT, "win_logit_penalty": WIN_PENALTY,
            "upstream_fit": "N1 priors refit from main train histories only, four grouped cross-fit splits",
            "network_fit": "main train pre-cutoff observed history outcomes and actual train cutoff labels only",
            "external_controls": "Archived factored GRU and hybrid used observed histories across all main splits. "
                                 "Only hybrid swaps inherit this history exposure.",
            "sources_sha256": {str(path.relative_to(KIT)): digest(path) for path in sources},
            "coordinator_manifest_sha256": digest(HERE.parents[1] / "manifest.json"),
            "weights_sha256": digest(args.out / "weights.npz"),
            "wall_seconds": perf_counter() - start,
        })
    print(f"E6 finished in {perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
