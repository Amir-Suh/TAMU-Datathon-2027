"""Compare dev runs with paired, condition-stratified episode uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

CONDITIONS = ("stable", "short_history", "drift", "shared_change")


def paired_points_interval(errors, baseline_errors, reference_errors, conditions,
                           max_points, *, repeats=2000, seed=20261107):
    """Percentile interval for candidate minus baseline contest points.

    Each draw samples whole episodes within each condition. Candidate, baseline
    and frozen-reference errors use the same sampled indices; the reference
    denominator is recomputed in every draw, as in the contest statistic.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in
              (errors, baseline_errors, reference_errors)]
    groups = np.asarray(conditions)
    if (groups.ndim != 1 or not len(groups)
            or any(v.shape != groups.shape for v in arrays)
            or any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in arrays)):
        raise ValueError("Expected matching finite, nonnegative episode-error vectors")
    if set(groups) != set(CONDITIONS):
        raise ValueError("Bootstrap requires all four announced conditions")
    if repeats < 1 or not np.isfinite(max_points) or not 0 < max_points <= 100:
        raise ValueError("Invalid bootstrap repeats or maximum points")
    rng = np.random.default_rng(seed)
    ratios = np.zeros((2, repeats), dtype=np.float64)
    for condition in CONDITIONS:
        ix = np.flatnonzero(groups == condition)
        sample = rng.choice(ix, size=(repeats, len(ix)), replace=True)
        denominator = np.maximum(1e-9, arrays[2][sample].mean(axis=1))
        for j in range(2):
            ratios[j] += arrays[j][sample].mean(axis=1) / denominator / len(CONDITIONS)
    points = max_points / (1.0 + ratios)
    return tuple(float(v) for v in np.quantile(points[0] - points[1], [0.025, 0.975]))


@dataclass
class Run:
    directory: Path
    arm: str
    variant: str
    result: dict
    meta: dict

    @property
    def points(self):
        return float(self.result.get("points", self.result.get("technical_points")))


def discover_runs(root):
    """Read only complete dev outputs, never interpreting final runs as dev."""
    runs = []
    for filename in sorted(Path(root).rglob("score.json")):
        directory = filename.parent
        try:
            result = json.loads(filename.read_text())
            meta = json.loads((directory / "meta.json").read_text())
            if meta.get("mode", "dev") != "dev":
                continue
            run = Run(directory, str(meta.get("arm", directory.parent.name)),
                      str(meta.get("variant", directory.name)), result, meta)
            if not np.isfinite(run.points) or not (directory / "val_pred.npy").is_file():
                raise ValueError("missing finite score or validation predictions")
            runs.append(run)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Skipping {directory}: {exc}", file=sys.stderr)
    return sorted(runs, key=lambda r: (-r.points, r.arm, r.variant, str(r.directory)))


def _ratio(result, condition):
    value = result["by_condition"][condition]
    return float(value["loss_ratio"] if isinstance(value, dict) else value)


def validate_run_data(run, rnd, *, query_sha256, metadata_sha256):
    """Refuse predictions whose recorded validation dataset differs."""
    if run.meta.get("round") != rnd.name:
        raise ValueError(f"{run.directory}: round differs from loaded data")
    roots = [run.meta[key] for key in ("data", "data_root") if run.meta.get(key)]
    if not roots:
        raise ValueError(f"{run.directory}: metadata has no data root; cannot verify predictions")
    if any(Path(root).resolve() != rnd.root.resolve() for root in roots):
        raise ValueError(f"{run.directory}: recorded data root differs from loaded data")
    fingerprints = [run.meta.get("validation_queries_sha256")]
    manifest = run.meta.get("query_manifest_sha256")
    fingerprints.append(manifest.get("validation") if isinstance(manifest, dict) else manifest)
    fingerprints.append(run.meta.get("input_sha256", {}).get("validation/queries.csv"))
    if any(value != query_sha256 for value in fingerprints if value is not None):
        raise ValueError(f"{run.directory}: validation query manifest fingerprint differs")
    if run.meta.get("metadata_sha256", metadata_sha256) != metadata_sha256:
        raise ValueError(f"{run.directory}: round metadata fingerprint differs")
    query_ids = run.meta.get("validation_query_ids")
    if query_ids is not None and not np.array_equal(np.asarray(query_ids).reshape(-1), rnd.splits["validation"].query_ids.reshape(-1)):
        raise ValueError(f"{run.directory}: validation query IDs/order differ")


def compare(root, *, baseline="pooled_recent", data=None, round_name=None):
    from tdx.data import load_round
    from tdx.score import episode_values

    runs = discover_runs(root)
    if not runs:
        raise ValueError(f"No complete dev runs under {root}")
    matches = [r for r in runs if baseline in (r.variant, f"{r.arm}/{r.variant}", str(r.directory))]
    if len(matches) != 1:
        raise ValueError(f"Baseline {baseline!r} matched {len(matches)} runs; use arm/variant or a path")
    incumbent = matches[0]
    round_name = round_name or incumbent.meta.get("round", "main")
    data = data or incumbent.meta.get("data") or incumbent.meta.get("data_root")
    data = data or os.environ.get("TDX_DATA_ROOT", f"data/{round_name}")
    rnd = load_round(data, name=round_name)
    split = rnd.splits["validation"]
    ids = list(split.ids)
    query_sha256 = hashlib.sha256((rnd.root / "validation" / "queries.csv").read_bytes()).hexdigest()
    metadata_sha256 = hashlib.sha256((rnd.root / "metadata.json").read_bytes()).hexdigest()

    def values(run):
        validate_run_data(run, rnd, query_sha256=query_sha256, metadata_sha256=metadata_sha256)
        raw = episode_values(rnd, np.load(run.directory / "val_pred.npy", allow_pickle=False))
        if set(raw) != set(ids):
            raise ValueError(f"{run.directory}: validation episode IDs differ")
        return ([raw[e]["error"] for e in ids],
                [raw[e]["reference_error"] for e in ids],
                [raw[e]["condition"] for e in ids])

    baseline_error, _, _ = values(incumbent)
    rows = []
    for run in runs:
        errors, reference, conditions = values(run)
        low, high = paired_points_interval(errors, baseline_error, reference, conditions, rnd.max_points)
        seconds = run.meta.get("seconds", run.result.get("seconds", 0.0))
        rows.append([run.arm, run.variant, f"{run.points:.3f}",
                     *[f"{_ratio(run.result, c):.3f}" for c in CONDITIONS],
                     f"{float(seconds):.2f}", f"{run.points - incumbent.points:+.3f}",
                     f"[{low:+.3f}, {high:+.3f}]"])
    headers = ["arm", "variant", "points", "stable", "short", "drift", "shared", "seconds", "Δpoints", "95% interval"]
    widths = [max(len(row[j]) for row in [headers, *rows]) for j in range(len(headers))]
    print(f"Baseline: {incumbent.arm}/{incumbent.variant}; {round_name}; 2000 paired episode resamples, seed 20261107")
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    print("PROPOSED reading rule: intervals containing zero are ties; prefer simpler or faster models.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="runs")
    parser.add_argument("--baseline", default="pooled_recent")
    parser.add_argument("--round", dest="round_name", choices=("main", "adaptation"))
    parser.add_argument("--data")
    args = parser.parse_args(argv)
    try:
        compare(args.root, baseline=args.baseline, data=args.data, round_name=args.round_name)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"compare: {exc}\n")


if __name__ == "__main__":
    main()
