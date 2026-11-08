"""Fit models on train, score them on validation, and write a results folder.

Usage (from the repository root):
    python -m models.run                       # baselines + m1 + m2
    python -m models.run --models m2 --out results/my-run

Every model is fitted on the train split only; hyperparameters are chosen on
train (see models/common/selection.py). Validation answers are used only to
score, never to fit or select. Test predictions are written in upload format
and format-checked, but test answers are private so they are not scored.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from .common.data import CONDITIONS, TYPES, load_split
from .common.evaluate import (DEFAULT_DATA, REPO, check_test_file, contest_points, extra_metrics, to_dict,
                              write_predictions)
from .m1_hierarchical_frequency import HierarchicalFrequencyModel
from .m2_lag_pair import LagPairModel
from .m3_lag_pair_flip import FlipLagPairModel
from .m4_lag_pair_flip_shared import SharedFlipLagPairModel
from .m5_boosted_corrector import BoostedCorrector
from .m6_ensemble import Ensemble

MODELS = {"m1": HierarchicalFrequencyModel, "m2": LagPairModel, "m3": FlipLagPairModel,
          "m4": SharedFlipLagPairModel, "m5": BoostedCorrector, "m6": Ensemble}
BASELINES = ("pooled", "pooled_recent")


def build(key: str, train, fitted: dict):
    """Fit one model, reusing already-fitted models that later ones are built on."""
    if key == "m5":
        return BoostedCorrector(base=fitted.get("m4")).fit(train)
    if key == "m6":
        return Ensemble(members={k: fitted[k] for k in ("m2", "m3", "m4", "m5") if k in fitted}).fit(train)
    return MODELS[key]().fit(train)


def run_kit_baseline(method: str, data_dir: Path):
    """The kit's FrequencyForecaster, fitted on train exactly like the starter."""
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels
    tq, th = load_inputs(data_dir / "train")
    model = FrequencyForecaster(method=method).fit(tq, th, read_labels(data_dir / "train" / "labels.csv", tq))
    return {s: model.predict(*load_inputs(data_dir / s)) for s in ("validation", "test")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or REPO / "results" / f"{dt.date.today().isoformat()}_{'_'.join(args.models)}"
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out so earlier runs are kept.")

    train, val, test = (load_split(args.data, s) for s in ("train", "validation", "test"))
    entries = []

    def record(name, kind, val_pred: dict, test_pred: dict, seconds: float, params=None, val_probs=None):
        d = out / name
        write_predictions(d / "validation_predictions.csv", val_pred)
        write_predictions(d / "submission.csv", test_pred)
        n_test = check_test_file(d / "submission.csv", args.data)
        score = contest_points(d / "validation_predictions.csv", args.data)
        if val_probs is None:
            val_probs = np.array([val_pred[q] for q in val.query_ids])
        if params is not None:
            (d / "params.json").write_text(json.dumps(params, indent=2, default=str) + "\n", encoding="utf-8")
        entry = {"name": name, "kind": kind, "points": score["technical_points"],
                 "mean_loss_ratio": score["mean_loss_ratio"],
                 "loss_ratio": {c: score["by_condition"][c]["loss_ratio"] for c in CONDITIONS},
                 "seconds": round(seconds, 1), "test_rows_checked": n_test, **extra_metrics(val, val_probs)}
        entries.append(entry)
        print(f"{name:28s} {entry['points']:6.2f} / 60   ({seconds:.1f}s)")

    for method in BASELINES:
        t0 = time.time()
        preds = run_kit_baseline(method, args.data)
        record(f"baseline_{method}", "kit baseline", preds["validation"], preds["test"], time.time() - t0)

    fitted = {}
    for key in sorted(args.models):
        t0 = time.time()
        model = fitted[key] = build(key, train, fitted)
        val_probs, test_probs = model.predict(val), model.predict(test)
        record(model.name, "model", to_dict(val, val_probs), to_dict(test, test_probs), time.time() - t0,
               params=model.describe(), val_probs=val_probs)
        if hasattr(model, "explain"):
            rows = model.explain(val)
            for r, cond in zip(rows, val.conditions):
                r["condition"] = cond
            with open(out / model.name / "validation_rules.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    (out / "scores.json").write_text(json.dumps({"run": out.name, "entries": entries}, indent=2) + "\n",
                                     encoding="utf-8")
    write_summary(out, entries)
    plot(out, entries)
    print(f"\nwrote {out}")


def write_summary(out: Path, entries: list) -> None:
    ref = next(e for e in entries if e["name"] == "baseline_pooled")
    lines = [f"# Run `{out.name}`", "",
             "Validation scores. Every model is fitted on **train only**; hyperparameters are chosen on train. "
             "Points use the kit's own scorer (`contest_score`, max 60). Loss ratio = model error / frozen pooled "
             "reference error for that condition (lower is better, 1.00 = reference).", "",
             "| Model | Points /60 | Δ vs reference | Stable | Short history | Drift | Shared change | Mean ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for e in sorted(entries, key=lambda e: -e["points"]):
        lr = e["loss_ratio"]
        lines.append(f"| `{e['name']}` | **{e['points']:.2f}** | {e['points'] - ref['points']:+.2f} | "
                     + " | ".join(f"{lr[c]:.3f}" for c in CONDITIONS) + f" | {e['mean_loss_ratio']:.3f} |")
    lines += ["", "## Extra diagnostics (validation)", "",
              "| Model | Mean Σ sq. error | " + " | ".join(f"Error: {t}" for t in TYPES)
              + " | Log-loss on observed outcomes | Runtime (s) |",
              "|---|---|" + "---|" * len(TYPES) + "---|---|"]
    for e in sorted(entries, key=lambda e: -e["points"]):
        lines.append(f"| `{e['name']}` | {e['mean_error']:.4f} | "
                     + " | ".join(f"{e['error_by_type'][t]:.4f}" for t in TYPES)
                     + f" | {e['outcome_logloss']:.4f} | {e['seconds']} |")
    lines += ["", "## Files", "",
              "* `<model>/validation_predictions.csv`: validation forecasts (score with "
              "`python check_submission.py <file> --split validation` from the kit folder).",
              "* `<model>/submission.csv`: test forecasts in upload format; format-checked against the test "
              "queries. Test answers are private, so these are not scored.",
              "* `<model>/params.json`: fitted priors, chosen hyperparameters and the train-only selection table.",
              "* `m2_lag_pair/validation_rules.csv`: the lag pair and direction model 2 believes each validation "
              "machine follows.",
              "* `loss_ratio_by_condition.png`: the table above as a chart.",
              "* `scores.json`: everything above in machine-readable form."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(out: Path, entries: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-8 in fixed order (validated for adjacent pairs, as in grouped bars)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    labels = {"stable": "Stable", "short_history": "Short history", "drift": "Drift", "shared_change": "Shared change"}
    fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    n = len(entries)
    width = 0.8 / n
    for i, e in enumerate(entries):
        xs = np.arange(len(CONDITIONS)) + (i - (n - 1) / 2) * width
        vals = [e["loss_ratio"][c] for c in CONDITIONS]
        ax.bar(xs, vals, width=width - 0.03, color=colors[i % len(colors)],
               label=f"{e['name']}  ({e['points']:.1f} pts)")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7 if n > 4 else 8, color=ink)
    ax.axhline(1.0, color=ink2, lw=1, ls=(0, (4, 3)))
    ax.text(len(CONDITIONS) - 0.5, 1.01, "reference = 1.00", ha="right", va="bottom", fontsize=9, color=ink2)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([labels[c] for c in CONDITIONS], color=ink2)
    ax.set_ylabel("Loss ratio vs. pooled reference (lower is better)", color=ink2)
    ax.set_ylim(0, 1.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.yaxis.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=ink2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(n, 4) if n > 4 else 2,
              fontsize=9)
    ax.set_title("Validation loss ratio by condition", loc="left", fontsize=13, fontweight="bold", color=ink)
    fig.savefig(out / "loss_ratio_by_condition.png", dpi=160, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
