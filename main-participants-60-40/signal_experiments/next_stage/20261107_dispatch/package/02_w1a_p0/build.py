"""Package the selected W1a win head and P0 conditional tail."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_frozen_controls"))
from build import KIT, SIGNAL, package_code, sha, stage_common, write

W1 = SIGNAL / "research_next/W1"
P0 = SIGNAL / "research_next/P0"
CONTROL = HERE.parents[1] / "coordinator/assembly/test_W1a_P0.csv"


def stage_candidate(source):
    stage_common(source, hybrid=True)
    frozen = (W1 / "run.py").read_text()
    ordered = frozen[frozen.index("def ordered_counts("):frozen.index("def train_contrasts(")]
    predict = frozen[frozen.index("def centered_children("):frozen.index("def prefix(")]
    write(source / "ordered_cells.py",
          "import numpy as np\n"
          "import match_split as w0\n"
          "from models.common.data import MISSING\n"
          "from models.m3_lag_pair_flip.model import _lagged\n\n"
          "CELLS = ((0, 3), (1, 2))\n\n\n" + ordered + predict)
    plan = json.loads((W1 / "plan.json").read_text())
    contrasts = json.loads((W1 / "train_contrasts.json").read_text())
    p0 = json.loads((P0 / "plan.json").read_text())
    state = {"w1a_strength": plan["selected_strength"], "w1a_pooled": contrasts["pooled"],
             "p0_type_order": ["steady", "balanced", "jackpot"],
             "p0_tail_counts": p0["counts_by_type"]}
    write(source / "weights/w1a_p0.json", json.dumps(state, indent=2) + "\n")
    # In this candidate match_split supplies history semantics and pseudo-counts.
    (source / "weights/match_contrast.json").unlink()
    original = (source / "predict.py").read_text()
    start = original.index("def predict_hybrid(")
    stop = original.index("    split = _read_split", start)
    replacement = """def predict_hybrid(model, flat_split, root, split_name):
    from types import SimpleNamespace

    import lightgbm as lgb
    from models.common.components import combine
    from ordered_cells import forecast
    from tdx.arms.gbdt.arm import GBDTArm
    from tdx.arms.seq.arm import SeqArm
    from tdx.data import _read_split

    state = json.loads((WEIGHTS / "w1a_p0.json").read_text())
    fit = SimpleNamespace(pooled=np.asarray(state["w1a_pooled"]))
    wins, _ = forecast(model, flat_split, fit, {"W1a": ("W1a", state["w1a_strength"])})

"""
    original = original[:start] + replacement + original[stop:]
    original = original.replace(
        "    return np.concatenate([p0, (1 - p0) * size / size.sum(axis=1, keepdims=True)], axis=1)",
        """    probabilities = combine(wins["W1a"], size / size.sum(axis=1, keepdims=True))
    counts = np.asarray([state["p0_tail_counts"][key] for key in state["p0_type_order"]])
    ratios = counts / counts.sum(axis=1, keepdims=True)
    probabilities[:, 2:] = probabilities[:, 2:].sum(axis=1, keepdims=True) * ratios[flat_split.machine_type]
    return probabilities""")
    write(source / "predict.py", original.replace('"hybrid"', '"w1a_p0"'))
    command = ["python", "-B", "predict.py", "--model", "w1a_p0", "--data", "/data",
               "--out", "/output/submission.csv"]
    write(source / "reproduce.json", json.dumps({"image": "python", "command": command}, indent=2) + "\n")
    readme = (source / "README.md").read_text()
    readme = readme[:readme.index("The win head is")] + (
        "The frozen M3 posterior and the W1a pooled 00/11 and 01/10 contrasts "
        "supply the win head. W1a contrasts and smoothing were estimated using "
        "released main training history and train-only grouped pseudo-cutoffs. "
        "An equal mixture of the 31-leaf LightGBM forecast and the mean of "
        "factored GRU seeds 1, 2, and 3 supplies the conditional 1x probability and "
        "combined tail mass. The P0 type-specific 2x, 5x, and 20x shares come "
        "from released pre-cutoff training payout counts. The archived tree and "
        "GRUs were fitted transductively on released pre-cutoff histories. GRUs also "
        "used released train cutoff outcomes. Tree early stopping used held train episodes.\n"
    )
    from build import HYBRID_CSV
    readme = readme.replace("main hybrid", "main W1a + P0").replace(
        sha(HYBRID_CSV), sha(CONTROL)).replace("--model hybrid", "--model w1a_p0")
    write(source / "README.md", readme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "w1a_p0_code.zip")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="house-edge-w1a-p0-build.") as temporary:
        source = Path(temporary)
        stage_candidate(source)
        print(package_code(source, args.out))
