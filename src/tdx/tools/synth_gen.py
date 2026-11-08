"""Synthetic data generator matching the slot machine simulator rules.

UNDERSTANDING ONLY: Synthetic data must NEVER be used to train anything that ships.
Generates kit-compatible data splits (train, validation, test) for evaluating models.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure kit is on sys.path
KIT_PATH = Path(__file__).resolve().parents[3] / "kit"
if str(KIT_PATH) not in sys.path:
    sys.path.insert(0, str(KIT_PATH))

# -----------------------------------------------------------------------------
# Configuration constants (all named parameters)
# -----------------------------------------------------------------------------

CONDITIONS: Tuple[str, ...] = ("stable", "short_history", "drift", "shared_change")
SPLIT_COUNTS: Dict[str, int] = {"train": 24, "validation": 24, "test": 48}

MACHINES_PER_EPISODE: int = 8
CUTOFF_MIN: int = 192
CUTOFF_MAX: int = 256
TOTAL_TICKS: int = 256

SHORT_HISTORY_RECENT_TICKS: int = 40
SHORT_HISTORY_OLDER_TICKS: int = 24
SHORT_HISTORY_TOTAL_TICKS: int = 64

MACHINE_TYPES: Tuple[str, ...] = ("steady", "balanced", "jackpot")
SIGNALS: Tuple[int, ...] = (-1, 0, 1)

# Marginal frequencies calibrated from real queries.csv
TYPE_FREQUENCIES: Tuple[float, ...] = (0.34, 0.34, 0.32)
SIGNAL_FREQUENCIES: Tuple[float, ...] = (0.33, 0.34, 0.33)

# Win rule parameters
LAG_PAIRS: Tuple[Tuple[int, int], ...] = ((1, 3), (2, 5), (4, 7))
DIR_MATCH_WINS: int = 0
DIR_DIFFER_WINS: int = 1

# Explicit P(win) table per (machine_type, signal)
# hi in 0.78-0.96, lo in 0.01-0.21
HI_TABLE: Dict[Tuple[str, int], float] = {
    ("steady", -1): 0.92,
    ("steady", 0): 0.94,
    ("steady", 1): 0.96,
    ("balanced", -1): 0.82,
    ("balanced", 0): 0.85,
    ("balanced", 1): 0.88,
    ("jackpot", -1): 0.78,
    ("jackpot", 0): 0.81,
    ("jackpot", 1): 0.84,
}

LO_TABLE: Dict[Tuple[str, int], float] = {
    ("steady", -1): 0.12,
    ("steady", 0): 0.16,
    ("steady", 1): 0.21,
    ("balanced", -1): 0.04,
    ("balanced", 0): 0.07,
    ("balanced", 1): 0.10,
    ("jackpot", -1): 0.01,
    ("jackpot", 0): 0.03,
    ("jackpot", 1): 0.05,
}

# Flip parameters
FLIP_MU: float = 145.0
FLIP_SIGMA: float = 35.0
FLIP_CLIP_MIN: int = 100
FLIP_CLIP_MAX: int = 230

SHARED_CHANGE_JITTER_SIGMA: float = 1.2
SHARED_CHANGE_MAX_SPREAD: float = 3.0
DRIFT_JITTER_SIGMA: float = 14.0

# Payout size parameters
PAYOUT_SIZES: Tuple[int, ...] = (1, 2, 5, 20)
ALL_PAYOUTS: Tuple[int, ...] = (0, 1, 2, 5, 20)

# Type mean vectors for (1x, 2x, 5x, 20x)
TYPE_PAYOUT_MEANS: Dict[str, Tuple[float, float, float, float]] = {
    "steady": (0.61, 0.30, 0.08, 0.01),
    "balanced": (0.37, 0.44, 0.16, 0.03),
    "jackpot": (0.24, 0.28, 0.35, 0.13),
}
# Concentration parameter giving std ~ 0.12 on 1x share
DIRICHLET_CONCENTRATION: float = 15.0

REFERENCE_METHOD: str = "pooled"
REFERENCE_PRIOR_STRENGTH: float = 20.0


# -----------------------------------------------------------------------------
# Generator Logic
# -----------------------------------------------------------------------------

class SynthSimulator:
    """Simulates slot machines under the teammate's lag-pair flip rule."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Tracking diagnostics
        self.winning_cell_ticks_by_ctx: Dict[Tuple[str, int], int] = {k: 0 for k in HI_TABLE}
        self.winning_cell_wins_by_ctx: Dict[Tuple[str, int], int] = {k: 0 for k in HI_TABLE}
        self.shared_change_spreads: List[int] = []

    def sample_flip_ticks(self, condition: str) -> Optional[List[int]]:
        """Return flip ticks for 8 machines in an episode."""
        if condition in ("stable", "short_history"):
            return None

        if condition == "shared_change":
            while True:
                base = float(np.clip(self.rng.normal(FLIP_MU, FLIP_SIGMA), FLIP_CLIP_MIN, FLIP_CLIP_MAX))
                jitters = self.rng.normal(0.0, SHARED_CHANGE_JITTER_SIGMA, size=MACHINES_PER_EPISODE)
                flips = np.clip(np.round(base + jitters).astype(int), FLIP_CLIP_MIN, FLIP_CLIP_MAX)
                spread = int(flips.max() - flips.min())
                if spread < SHARED_CHANGE_MAX_SPREAD:
                    self.shared_change_spreads.append(spread)
                    return flips.tolist()

        if condition == "drift":
            base = float(np.clip(self.rng.normal(FLIP_MU, FLIP_SIGMA), FLIP_CLIP_MIN, FLIP_CLIP_MAX))
            jitters = self.rng.normal(0.0, DRIFT_JITTER_SIGMA, size=MACHINES_PER_EPISODE)
            flips = np.clip(np.round(base + jitters).astype(int), FLIP_CLIP_MIN, FLIP_CLIP_MAX)
            return flips.tolist()

        raise ValueError(f"Unknown condition: {condition}")

    def simulate_machine(
        self,
        mtype: str,
        signal: int,
        cutoff: int,
        flip_tick: Optional[int],
    ) -> Dict[str, Any]:
        """Simulate one slot machine from tick 0 up to cutoff."""
        pair_idx = self.rng.integers(0, len(LAG_PAIRS))
        lag_a, lag_b = LAG_PAIRS[pair_idx]
        initial_dir = int(self.rng.integers(0, 2))  # 0: match wins, 1: differ wins

        ctx = (mtype, signal)
        hi = HI_TABLE[ctx]
        lo = LO_TABLE[ctx]

        mean_vector = np.array(TYPE_PAYOUT_MEANS[mtype], dtype=np.float64)
        alpha = mean_vector * DIRICHLET_CONCENTRATION
        size_probs = self.rng.dirichlet(alpha)  # (1x, 2x, 5x, 20x)

        payouts = np.zeros(cutoff + 1, dtype=np.int8)
        wins = np.zeros(cutoff + 1, dtype=bool)

        for t in range(cutoff + 1):
            is_win_a = wins[t - lag_a] if (t - lag_a >= 0) else False
            is_win_b = wins[t - lag_b] if (t - lag_b >= 0) else False
            match = (is_win_a == is_win_b)

            is_flipped = (flip_tick is not None and t >= flip_tick)
            current_dir = (1 - initial_dir) if is_flipped else initial_dir
            match_is_win = (current_dir == DIR_MATCH_WINS)
            in_winning_cell = (match == match_is_win)

            p_win = hi if in_winning_cell else lo

            if t < cutoff:
                won = bool(self.rng.random() < p_win)
                wins[t] = won
                if won:
                    payouts[t] = int(self.rng.choice(PAYOUT_SIZES, p=size_probs))
                else:
                    payouts[t] = 0

                if in_winning_cell:
                    self.winning_cell_ticks_by_ctx[ctx] += 1
                    if won:
                        self.winning_cell_wins_by_ctx[ctx] += 1
            else:
                # True next-spin probabilities at cutoff
                p_0 = 1.0 - p_win
                probs = [
                    float(p_0),
                    float(p_win * size_probs[0]),
                    float(p_win * size_probs[1]),
                    float(p_win * size_probs[2]),
                    float(p_win * size_probs[3]),
                ]
                # Normalize just in case of float precision
                s = sum(probs)
                probs = [p / s for p in probs]
                outcome = int(self.rng.choice(ALL_PAYOUTS, p=probs))

        return {
            "payouts": payouts[:cutoff],
            "probs": probs,
            "outcome": outcome,
            "lag_pair": (lag_a, lag_b),
            "initial_dir": initial_dir,
            "flip_tick": flip_tick,
            "size_probs": size_probs,
        }

    def generate_split(
        self,
        split_name: str,
        n_per_condition: int,
    ) -> Dict[str, Any]:
        """Generate all episodes and machines for a single split."""
        queries: List[Dict[str, Any]] = []
        histories: List[Dict[str, Any]] = []
        labels: List[Dict[str, Any]] = []
        conditions: List[Dict[str, Any]] = []
        targets: List[Dict[str, Any]] = []
        sample_submissions: List[Dict[str, Any]] = []

        ep_idx = 0
        for cond in CONDITIONS:
            for _ in range(n_per_condition):
                ep_id = f"synth-{split_name}-e{ep_idx:05d}"
                cutoff = int(self.rng.integers(CUTOFF_MIN, CUTOFF_MAX + 1))
                flip_ticks = self.sample_flip_ticks(cond)

                for m in range(MACHINES_PER_EPISODE):
                    mach_id = f"{ep_id}-m{m:03d}"
                    query_id = f"{mach_id}-next"

                    mtype = str(self.rng.choice(MACHINE_TYPES, p=TYPE_FREQUENCIES))
                    signal = int(self.rng.choice(SIGNALS, p=SIGNAL_FREQUENCIES))
                    flip_tick = flip_ticks[m] if flip_ticks is not None else None

                    sim = self.simulate_machine(mtype, signal, cutoff, flip_tick)

                    queries.append({
                        "query_id": query_id,
                        "episode_id": ep_id,
                        "machine_id": mach_id,
                        "machine_type": mtype,
                        "signal": signal,
                        "cutoff": cutoff,
                    })
                    conditions.append({
                        "query_id": query_id,
                        "condition": cond,
                    })
                    labels.append({
                        "query_id": query_id,
                        "outcome": sim["outcome"],
                    })
                    targets.append({
                        "query_id": query_id,
                        "episode_id": ep_id,
                        "p_0": sim["probs"][0],
                        "p_1": sim["probs"][1],
                        "p_2": sim["probs"][2],
                        "p_5": sim["probs"][3],
                        "p_20": sim["probs"][4],
                        "outcome": sim["outcome"],
                    })
                    sample_submissions.append({
                        "query_id": query_id,
                        "p_0": 0.2, "p_1": 0.2, "p_2": 0.2, "p_5": 0.2, "p_20": 0.2,
                    })

                    # History writing
                    payouts = sim["payouts"]
                    if cond == "short_history":
                        recent_ticks = list(range(cutoff - SHORT_HISTORY_RECENT_TICKS, cutoff))
                        older_pool = list(range(0, cutoff - SHORT_HISTORY_RECENT_TICKS))
                        sampled_older = self.rng.choice(
                            older_pool, size=SHORT_HISTORY_OLDER_TICKS, replace=False
                        ).tolist()
                        observed_ticks = sorted(sampled_older + recent_ticks)
                    else:
                        observed_ticks = list(range(cutoff))

                    for t in observed_ticks:
                        histories.append({
                            "episode_id": ep_id,
                            "machine_id": mach_id,
                            "tick": t,
                            "payout": int(payouts[t]),
                        })

                ep_idx += 1

        return {
            "queries": queries,
            "histories": histories,
            "labels": labels,
            "conditions": conditions,
            "targets": targets,
            "sample_submissions": sample_submissions,
        }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_synthetic_dataset(output_dir: Path, seed: int = 1) -> SynthSimulator:
    """Generate train, validation, test splits and metadata.json under output_dir."""
    output_dir = Path(output_dir)
    sim = SynthSimulator(seed=seed)

    splits_data = {}
    for split_name, count in SPLIT_COUNTS.items():
        splits_data[split_name] = sim.generate_split(split_name, count)

    # Write files for each split
    for split_name, data in splits_data.items():
        s_dir = output_dir / split_name
        s_dir.mkdir(parents=True, exist_ok=True)

        # queries.csv
        _write_csv(
            s_dir / "queries.csv",
            data["queries"],
            ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"],
        )

        # history.csv
        _write_csv(
            s_dir / "history.csv",
            data["histories"],
            ["episode_id", "machine_id", "tick", "payout"],
        )

        # sample_submission.csv
        _write_csv(
            s_dir / "sample_submission.csv",
            data["sample_submissions"],
            ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"],
        )

        # train & validation have labels and conditions
        if split_name in ("train", "validation"):
            _write_csv(
                s_dir / "labels.csv",
                data["labels"],
                ["query_id", "outcome"],
            )
            _write_csv(
                s_dir / "conditions.csv",
                data["conditions"],
                ["query_id", "condition"],
            )

        # validation only has targets.csv
        if split_name == "validation":
            _write_csv(
                s_dir / "targets.csv",
                data["targets"],
                ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"],
            )

    # Compute reference.csv on synthetic validation using FrequencyForecaster
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import read_history, read_labels, read_queries, write_submission

    train_queries = read_queries(output_dir / "train" / "queries.csv")
    train_history = read_history(output_dir / "train" / "history.csv")
    train_labels = read_labels(output_dir / "train" / "labels.csv", train_queries)

    val_queries = read_queries(output_dir / "validation" / "queries.csv")
    val_history = read_history(output_dir / "validation" / "history.csv")

    forecaster = FrequencyForecaster(
        method=REFERENCE_METHOD, prior_strength=REFERENCE_PRIOR_STRENGTH
    )
    forecaster.fit(train_queries, train_history, train_labels)
    val_preds = forecaster.predict(val_queries, val_history)
    write_submission(output_dir / "validation" / "reference.csv", val_preds)

    # Copy / write metadata.json
    metadata = {
        "conditions": list(CONDITIONS),
        "counts_per_condition": {
            "test": SPLIT_COUNTS["test"],
            "train": SPLIT_COUNTS["train"],
            "validation": SPLIT_COUNTS["validation"],
        },
        "history": "Machine-specific interactions, lags up to 8, missing observations and changing regimes.",
        "machines_per_episode": MACHINES_PER_EPISODE,
        "max_points": 60,
        "mode": "event",
        "reference": "Pooled, strength=20, fitted separately on this round's training observations.",
        "round": "main",
        "season_id": "5fffdc4d0c700bbf61c3f911",
        "suite_version": "experimentation-suite-v3",
        "target": "Next-spin probabilities for 0x, 1x, 2x, 5x, 20x.",
        "test_policy": "Test targets, conditions and reference predictions stay private.",
        "warning": "Public demo seeds are reconstructable; event seeds are private.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--out", type=Path, default=Path("/tmp/agy_synth/data_seed1"), help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", help="Multiple seeds to generate")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/agy_synth"), help="Base output directory")
    args = parser.parse_args(argv)

    if args.seeds:
        for s in args.seeds:
            target = args.out_dir / f"data_seed{s}"
            print(f"Generating synthetic dataset for seed {s} in {target}...")
            generate_synthetic_dataset(target, seed=s)
            print(f"Done seed {s}.")
    else:
        print(f"Generating synthetic dataset for seed {args.seed} in {args.out}...")
        generate_synthetic_dataset(args.out, seed=args.seed)
        print(f"Done seed {args.seed}.")


if __name__ == "__main__":
    main()
