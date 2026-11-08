"""Export fitted state and build the two main code-only submission ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import joblib

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
SIGNAL = KIT / "signal_experiments"
FREDY = "bbff7379e57984eb623588273ed1c958bb8b4f47"
REPLAY = SIGNAL / "01_fredy_replay/checkpoints"
HYBRID_CSV = SIGNAL / "combined/m3_match_split_win_pair_l31_factor_gru_size_test.csv"
M3_CSV = FROZEN / "m3/submission.csv"
sys.path[:0] = [str(REPAIR), str(KIT)]

from slot_forecasting.submission_contract import package_code


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def write(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def export_m3(path):
    checkpoint = FROZEN / "models.joblib"
    assert sha(checkpoint) == json.loads((FROZEN / "freeze.json").read_text())["checkpoint_sha256"]
    model = joblib.load(checkpoint)["m3"]
    assert type(model).__name__ == "FlipLagPairModel" and not model.share_episode
    export = {name: getattr(model, name) for name in
              ("min_segment", "null_weight", "p_flip", "pairs", "params")}
    for name in ("tau_grid", "win_prior", "hi", "lo", "pair_prior", "tau_q"):
        export[name] = getattr(model, name).tolist()
    export["size"] = {
        "prior": model.size.prior.tolist(),
        "strength": model.size.params["strength"],
        # JSON null is the exact no-decay setting.
        "half_life": None if model.size.params["half_life"] == float("inf")
        else model.size.params["half_life"],
    }
    write(path, json.dumps(export, indent=2, allow_nan=False) + "\n")


def stage_common(stage, hybrid):
    for name in ("__init__.py", "schema.py", "io.py"):
        relative = Path("slot_forecasting") / name
        copy(KIT / relative, stage / relative)
    for name in ("__init__.py", "common/__init__.py", "common/data.py",
                 "common/components.py", "common/selection.py",
                 "m3_lag_pair_flip/__init__.py", "m3_lag_pair_flip/model.py"):
        relative = Path("models") / name
        copy(REPAIR / relative, stage / relative)
    loader = (REPAIR / "run_repaired_comparison.py").read_text().split("def input_only(", 1)[1]
    loader = loader.split("\n\ndef binary_metrics", 1)[0]
    write(stage / "input_only.py",
          "import csv\nimport numpy as np\n"
          "from models.common.data import MISSING, PAYOUT_INDEX, TYPES, Split\n\n"
          "def input_only(" + loader + "\n")
    copy(HERE / "predict.py", stage / "predict.py")
    export_m3(stage / "weights/m3.json")
    entry = "hybrid" if hybrid else "m3"
    command = ["python", "-B", "predict.py", "--model", entry, "--data", "/data",
               "--out", "/output/submission.csv"]
    write(stage / "reproduce.json", json.dumps({"image": "python", "command": command}, indent=2) + "\n")
    dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
    if hybrid:
        dependencies += ["lightgbm==4.7.0", "narwhals==2.28.0"]
    write(stage / "requirements.txt", "\n".join(dependencies) + "\n")
    data = json.loads((KIT / "public/metadata.json").read_text())
    forecast = HYBRID_CSV if hybrid else M3_CSV
    models = (
        "The win head is frozen M3 with a train-history 00/11 match-cell contrast. "
        "The positive payout mix conditions an equal mixture of the 31-leaf LightGBM forecast "
        "and the mean of factored GRU seeds 1, 2, and 3. The archived tree and GRUs were "
        "fitted transductively on released pre-cutoff histories. The GRUs also used released "
        "train cutoff outcomes. Tree early stopping used held train episodes. "
        "The branch source is " + FREDY + "."
        if hybrid else "The frozen M3 win and conditional payout heads were fitted using "
        "released main training histories and train cutoff outcomes."
    )
    write(stage / "README.md", f"""# House Edge main {entry}

Season `{data['season_id']}`. Frozen main forecast SHA-256 `{sha(forecast)}`.

Python 3.13. Install the exact versions in `requirements.txt` before running offline.
Run from `/submission` with the released round mounted at `/data` and writable output
at `/output`.

```sh
{' '.join(command)}
```

The command reads test query IDs and pre-cutoff histories and writes
`/output/submission.csv`. All learned state is under `weights/`.
{models}
""")
    if hybrid:
        stage_hybrid(stage)


def stage_hybrid(stage):
    original = (SIGNAL / "04_four_lag_cells/run.py").read_text()
    constants = original[original.index("GLOBAL_PRIOR ="):original.index("def read_checkpoint(")]
    functions = original[original.index("def win_history("):original.index("def validate_sparse_and_causal(")]
    start = functions.index("def train_contrasts(")
    stop = functions.index("def child_forecasts(")
    write(stage / "match_split.py",
          "from dataclasses import dataclass\nimport numpy as np\n"
          "from models.common.components import combine\n"
          "from models.common.data import MISSING, Split\n"
          "from models.m3_lag_pair_flip.model import FlipLagPairModel, _lagged\n\n"
          + constants + functions[:start] + functions[stop:])
    copy(SIGNAL / "04_four_lag_cells/train_fit.json", stage / "weights/match_contrast.json")
    sources = (
        "__init__.py", "arm.py", "data.py", "feat.py",
        "arms/__init__.py", "arms/gbdt/__init__.py", "arms/gbdt/arm.py",
        "arms/gbdt/features.py", "arms/seq/__init__.py", "arms/seq/arm.py",
    )
    for name in sources:
        content = subprocess.check_output(
            ["git", "show", f"{FREDY}:src/tdx/{name}"], cwd=KIT, text=True)
        if name == "data.py":
            # Keep the original feature inputs and omit answer-loading and caches.
            start = content.index("def _read_split(")
            stop = content.index("def load_round(")
            declaration = content[:content.index("def _answers(")]
            declaration = declaration.replace("from .paths import ROOT, data_root\n", "")
            reader = content[start:stop].replace(
                "return _answers(split, root) if name == 'train' else split", "return split")
            content = declaration + reader
        write(stage / "tdx" / name, content)
    for name, files in {
        "pair_l31": ("features.json", "model.txt.gz"),
        **{f"gru128_factor_seed_{seed}": ("meta.json", "model.npz") for seed in range(1, 4)},
    }.items():
        for filename in files:
            copy(REPLAY / name / filename, stage / "weights" / name / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    for arm in ("hybrid", "m3"):
        with tempfile.TemporaryDirectory(prefix=f"house-edge-{arm}-build.") as temporary:
            source = Path(temporary)
            stage_common(source, hybrid=arm == "hybrid")
            print(package_code(source, args.out / f"{arm}_code.zip"))


if __name__ == "__main__":
    main()
