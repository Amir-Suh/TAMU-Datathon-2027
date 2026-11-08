"""Summarize E4 provenance after score and format checks complete."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze_path = HERE / "forecast_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    traces = {}
    source_files = set()
    for name in ("fit_predict_audit.jsonl", "score_audit.jsonl"):
        records = [json.loads(line) for line in (HERE / name).read_text().splitlines()]
        assert records[-1] == {"event": "end", "exit": 0}
        assert all(row.get("allowed", True) for row in records)
        accessed = {r["path"] for r in records if r["event"] == "open" and r["flags"] & 3 == 0}
        protected = sorted(p for p in accessed if Path(p).name in
                           ("targets.csv", "labels.csv", "reference.csv", "conditions.csv")
                           and set(Path(p).parts) & {"validation", "test", "lockbox"})
        if name.startswith("fit"):
            assert not protected
        else:
            assert set(protected) == {f"public/validation/{file}" for file in
                                       ("conditions.csv", "reference.csv", "targets.csv")}
        source_files.update(p for p in accessed if p.endswith(".py") and not p.startswith("/") and (KIT / p).is_file())
        traces[name] = {"sha256": sha(HERE / name), "start": records[0], "end": records[-1],
                        "protected_reads": protected, "distinct_read_paths": len(accessed),
                        "child_processes": [r["command"] for r in records if r["event"] == "child_process"]}
    inputs = {
        **freeze["input_sha256"],
        **{p: sha(KIT / p) for p in sorted(source_files)},
        **{p: sha(KIT / p) for p in (
            "public/metadata.json", "public/validation/targets.csv", "public/validation/reference.csv",
            "public/validation/conditions.csv", "signal_experiments/research_next/S1/validation_global.csv",
            "signal_experiments/research_next/S1/coefficients.json",
            "signal_experiments/research_next/S2/validation_drop_flip.csv",
            "signal_experiments/research_next/S2/validation_predictions.csv",
            "signal_experiments/next_stage/20261107_dispatch/lag_cells/attempt_01/frozen_validation_slices.csv",
            "m5_m6_repair/results/repaired-m5-m6/m3/validation_predictions.csv",
            "m5_m6_repair/results/repaired-m5-m6/m4/validation_predictions.csv")},
    }
    audited_outputs = {p.name: sha(p) for p in sorted(HERE.iterdir())
                       if p.is_file() and p.name != "manifest.json"}
    report = {
        "round": freeze["round"], "season_id": freeze["season_id"], "owner": "sparse_filter", "card": "E4",
        "attempt": HERE.name, "seed": 20261107, "numerical_threads": 1, "python": sys.version,
        "platform": platform.platform(), "packages": freeze["packages"],
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "commands": "commands.sh", "fold_manifest": "episode_fold_manifest.json",
        "forecast_freeze": {"path": freeze_path.name, "sha256": sha(freeze_path)},
        "mask_manifests": ["validation_mask_manifest.csv", "test_mask_manifest.csv"],
        "fit_scope": freeze["fit_scope"], "checks": freeze["checks"], "audit": traces,
        "inference_seconds": {split: value["timing"] for split, value in freeze["result"].items()},
        "fit_predict_elapsed_seconds": freeze["elapsed_seconds"], "arms": freeze["result"],
        "input_sha256": inputs, "output_sha256": audited_outputs,
    }
    (HERE / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
