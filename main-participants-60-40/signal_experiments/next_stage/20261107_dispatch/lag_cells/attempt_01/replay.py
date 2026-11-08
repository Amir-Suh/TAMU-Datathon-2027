"""Replay published E3 controls without any validation outcomes."""
import importlib.util
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
DATA = ROOT / "public"
W2 = ROOT / "signal_experiments/research_next/W2"
FORBIDDEN = {"labels.csv", "targets.csv", "reference.csv"}


def trace(phase):
    stream = (OUT / f"{phase}_file_reads.jsonl").open("w")
    seen = set()

    def hook(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(args[0]).absolute()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return
        if phase != "score" and relative.parts[:2] in {("public", "validation"), ("public", "test")} and path.name in FORBIDDEN:
            raise PermissionError(f"Forbidden in {phase}: {relative}")
        if relative not in seen:
            seen.add(relative)
            print(json.dumps({"phase": phase, "path": str(relative), "mode": args[1]}), file=stream, flush=True)

    sys.addaudithook(hook)


def load_w2():
    spec = importlib.util.spec_from_file_location("w2_replay", W2 / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    trace("replay")
    started = time.perf_counter()
    w2 = load_w2()
    w1 = w2.w1
    np = w2.np
    manifest = json.loads((OUT.parents[1] / "manifest.json").read_text())
    for filename in ("EXPERIMENT_PLAN.md", "AGENT_DISPATCH_PLAN.md", "folds.csv"):
        path = w1.EXPERIMENTS / filename
        key = str(path.relative_to(ROOT))
        assert manifest["input_sha256"][key] == w1.sha256(path), key
    previous = json.loads((w2.W1 / "plan.json").read_text())
    strength = previous["selected_strength"]
    model = w1.w0.read_checkpoint(w1.FROZEN)
    fit = w1.train_contrasts(model, w1.labeled_input(DATA))
    variants = {arm: (arm, strength) for arm in ("W1a", "W1b")}
    result = {}
    for name in ("validation", "test"):
        split = w1.input_only(DATA, name)
        hybrid = w1.COMBINED.with_suffix(".csv") if name == "validation" else w1.COMBINED.with_name(w1.COMBINED.name + "_test.csv")
        outputs = {arm: OUT / f"{name}_{arm}.csv" for arm in variants}
        result[name], details = w1.compose_forecasts(model, split, fit, variants, hybrid, outputs)
        outputs["W2"] = OUT / f"{name}_W2.csv"
        w2result, _ = w2.write_forecast(model, split, fit, strength, hybrid,
                                         outputs["W1a"], outputs["W2"])
        result[name]["W2"] = w2result
        for arm, output in outputs.items():
            archive = (w2.W1 / f"{arm}_{name}_predictions.csv" if arm != "W2"
                       else W2 / f"{name}_predictions.csv")
            key = str(archive.relative_to(ROOT))
            assert manifest["input_sha256"][key] == w1.sha256(archive), key
            delta = np.abs(w1.array_at(archive, split) - w1.array_at(output, split)).max()
            assert delta < 1e-12, (arm, delta)
            result[name][arm]["archive_delta"] = float(delta)
        if name == "validation":
            result["causal_W1"] = w1.causal_checks(model, split, fit, variants)
            result["causal_W2"] = w2.causal_checks(model, split, fit, strength)
            inference = model._infer(split)
            pair_mass = inference["ind"]["post"].sum(axis=(2, 3)).max(axis=1)
            rows = [{"query_id": q, "episode_id": split.episode_ids[i], "top_pair": "-".join(map(str, model.pairs[details["top_pair"][i]])),
                     "top_pair_mass": float(pair_mass[i]), "top_cell": int(details["top_context"][i])}
                    for i, q in enumerate(split.query_ids)]
            w1.write_rows(OUT / "frozen_validation_slices.csv", tuple(rows[0]), rows)
    result["elapsed_seconds"] = time.perf_counter() - started
    w1.write_json(OUT / "control_replay.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
