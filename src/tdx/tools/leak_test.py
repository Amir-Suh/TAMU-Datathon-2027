"""Audit dev files, cutoff predictions, and historical forward-pass causality."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys

import numpy as np


@contextmanager
def audit_dev_files(root):
    """Raise on evaluation-answer or non-input test-file opens while active.

    Python cannot remove audit hooks, so the installed hook becomes inert on
    leaving this context, including when fitting raises an exception. Resolved
    paths catch both the data symlink and its actual public-directory target.
    """
    root = Path(root).resolve()
    state = {"active": True, "blocked": []}
    forbidden = {root / "validation" / f"{name}.csv"
                 for name in ("labels", "targets", "conditions", "reference")}

    def hook(event, args):
        if event != "open" or not state["active"]:
            return
        filename = args[0]
        if isinstance(filename, int):
            # fdopen is not necessarily accompanied by a filename open event.
            filename = f"/proc/self/fd/{filename}"
        if not isinstance(filename, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(filename)).resolve()
        try:
            relative = path.relative_to(root / "test")
        except ValueError:
            relative = None
        if path in forbidden or (relative is not None and str(relative) not in ("queries.csv", "history.csv")):
            state["blocked"].append(str(path))
            raise PermissionError(f"Dev leakage audit blocked opening {path}")

    sys.addaudithook(hook)
    try:
        yield state
    finally:
        state["active"] = False


def truncate_for_probe(split, *, episodes=5, gap=4):
    """Return legal earlier-cutoff histories for the first selected episodes."""
    if episodes < 1 or gap < 1:
        raise ValueError("episodes and gap must be positive")
    count = min(episodes, len(split.ids))
    if not count:
        raise ValueError("Causality check needs at least one episode")
    cutoff = split.cutoff.copy()
    cutoff[:count] -= gap
    if np.any(cutoff[:count] < 0):
        raise ValueError("Cannot reduce a cutoff below zero")
    y = split.y.copy()
    signal = split.signal.copy()
    for e in range(count):
        y[e, :, int(cutoff[e]):] = -1
        signal[e] = 0  # The original signal is unavailable at the earlier cutoff.
    return replace(split, y=y, cutoff=cutoff, signal=signal,
                   n_obs=(y >= 0).sum(axis=2).astype(split.n_obs.dtype))


def assert_step_causality(model, y, mtype, *, k):
    """Require exact prefix equality under future values and observedness edits.

    Inputs are already masked at their query cutoffs. Keep their full storage
    length when calling step_outputs: slicing first would hide whole-row leaks.
    Output step t consumes input tick t, so only steps strictly below k are tested.
    """
    y = np.asarray(y)
    if y.ndim != 3 or not 0 < k < y.shape[-1]:
        raise ValueError("Step probe needs [E,M,T] histories and 0 < k < T")
    expected = np.asarray(model.step_outputs(y.copy(), mtype, steps=k)).copy()
    if expected.ndim != 4 or expected.shape[:3] != (*y.shape[:2], k):
        raise AssertionError("Step output shape must be [E,M,k,outputs]")
    if expected.dtype != np.float64 or not np.all(np.isfinite(expected)):
        raise AssertionError("Step outputs must be finite float64")

    values = y.copy()
    tail = values[..., k:]
    observed = tail >= 0
    tail[observed] = (tail[observed] + 1) % 5
    remove, add, move = y.copy(), y.copy(), y.copy()
    removed = added = moved = 0
    for e, m in np.ndindex(y.shape[:2]):
        obs = np.flatnonzero(y[e, m, k:] >= 0) + k
        missing = np.flatnonzero(y[e, m, k:] < 0) + k
        if obs.size:
            remove[e, m, obs[-1]] = -1
            removed += 1
        if missing.size:
            add[e, m, missing[-1]] = 2
            added += 1
        if obs.size and missing.size:
            move[e, m, missing[-1]] = y[e, m, obs[-1]]
            move[e, m, obs[-1]] = -1
            moved += 1
    if not (removed and added and moved):
        raise ValueError("Step probe needs observed and missing future ticks for all perturbations")
    differences = {}
    for kind, z in (("values", values), ("remove", remove), ("add", add), ("move", move)):
        actual = np.asarray(model.step_outputs(z, mtype, steps=k))
        if actual.shape != expected.shape or actual.dtype != np.float64 or not np.all(np.isfinite(actual)):
            raise AssertionError(f"Step causality {kind}: invalid outputs")
        difference = float(np.max(np.abs(actual - expected)))
        if not np.array_equal(actual, expected):
            raise AssertionError(f"Step causality {kind}: future history changed steps < {k}; "
                                 f"max abs difference {difference:.17g}")
        differences[kind] = difference
    return differences


def assert_prediction_causality(model, split, *, episodes=5, seed=20261107):
    """Compare legal histories with randomized storage at/after their cutoff.

    The model must be fitted using the legal truncated round first, so its
    transductive training cannot consume the withheld future either.
    """
    count = min(episodes, len(split.ids))
    dirty_y = split.y.copy()
    rng = np.random.default_rng(seed)
    for e in range(count):
        tail = dirty_y[e, :, int(split.cutoff[e]):]
        tail[:] = rng.integers(0, 5, size=tail.shape, dtype=np.int8)
    dirty = replace(split, y=dirty_y)
    expected = np.asarray(model.predict(split)).copy()
    actual = np.asarray(model.predict(dirty))
    required = (*split.y.shape[:2], 5)
    if expected.shape != required or actual.shape != required:
        raise AssertionError(f"Prediction shape must be {required}")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise AssertionError("Causality probe produced nonfinite probabilities")
    difference = float(np.max(np.abs(expected - actual)))
    if not np.array_equal(expected, actual):
        raise AssertionError(f"Future history changed predictions; max abs difference {difference:.17g}")
    return difference


def check_arm(arm, variant=None, *, data=None, round_name="main", episodes=5, gap=4, overrides=None):
    from tdx.data import load_round
    from tdx.fit import dev_spec
    from tdx.run import fit_round

    module = importlib.import_module(f"tdx.arms.{arm}")
    variants = module.VARIANTS
    variant = variant or next(iter(variants))
    if variant not in variants:
        raise ValueError(f"Unknown {arm} variant {variant!r}; choose from {', '.join(variants)}")
    params = dict(variants[variant])
    params.update(overrides or {})
    spec = dev_spec(bool(params.get("transductive", False)))
    data = data or os.environ.get("TDX_DATA_ROOT", f"data/{round_name}")
    with audit_dev_files(data) as audit:
        rnd = load_round(data, name=round_name, answers=False)
        splits = dict(rnd.splits)
        splits["validation"] = truncate_for_probe(splits["validation"], episodes=episodes, gap=gap)
        probe = replace(rnd, splits=splits)
        sanitized = fit_round(probe, spec)
        for name in ("validation", "test"):
            split = sanitized.splits[name]
            for field in ("label", "target", "condition", "reference"):
                if getattr(split, field, None) is not None:
                    raise AssertionError(f"dev fit exposes {name}.{field}")
        model = module.build(**params)
        model.fit(sanitized, spec)
        difference = assert_prediction_causality(model, splits["validation"], episodes=episodes)
        step_difference = None
        if hasattr(model, "step_outputs"):
            # Keep actual query layouts here: truncating a 64-observation row
            # first would hide the count == 64 leak this audit must detect.
            split = rnd.splits["validation"]
            count = min(3, len(split.ids))
            y = split.y[:count].copy()
            for e, cutoff in enumerate(split.cutoff[:count]):
                y[e, :, int(cutoff):] = -1
            mtype = split.mtype[:count].astype(np.int64)
            ks = {min(32, int(split.cutoff[:count].min()) - 8),
                  int(split.cutoff[:count].min()) - 8}
            step_difference = max(delta for k in sorted(ks)
                                  for delta in assert_step_causality(model, y, mtype, k=k).values())
    # An arm may catch an audit exception, but the attempted leak still fails.
    if audit["blocked"]:
        raise AssertionError(f"Forbidden file access attempted: {audit['blocked']}")
    return {"arm": arm, "variant": variant, "episodes": min(episodes, len(splits["validation"].ids)),
            "gap": gap, "max_abs_difference": difference, "step_causality": step_difference,
            "forbidden_opens": len(audit["blocked"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--round", dest="round_name", choices=("main", "adaptation"), default="main")
    parser.add_argument("--data")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--set", action="append", default=[], metavar="key=json")
    args = parser.parse_args(argv)
    try:
        from tdx.run import parse_set
        result = check_arm(args.arm, args.variant, data=args.data, round_name=args.round_name,
                           episodes=args.episodes, gap=args.gap, overrides=parse_set(args.set))
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"leak_test: FAIL: {exc}\n")
    step = "not available" if result['step_causality'] is None else f"{result['step_causality']:.1f}"
    print(f"PASS {result['arm']}/{result['variant']}: {result['episodes']} episodes, cutoff reduced "
          f"by {result['gap']}, max abs difference {result['max_abs_difference']:.1f}; "
          f"forbidden dev opens {result['forbidden_opens']}; step_causality: {step}")


if __name__ == "__main__":
    main()
