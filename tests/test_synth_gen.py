"""Unit tests for synthetic data generator."""
import csv
from pathlib import Path
import pytest
import numpy as np

from tdx.tools.synth_gen import (
    SynthSimulator,
    generate_synthetic_dataset,
    HI_TABLE,
    CONDITIONS,
    SHORT_HISTORY_TOTAL_TICKS,
)

REAL_PUBLIC_DIR = Path("/home/ganfwe/Downloads/main-participants-60-40/public")
SYNTH_TEST_DIR = Path("/tmp/agy_synth/test_verify")


@pytest.fixture(scope="module")
def synth_data():
    """Generate synthetic data once for the test suite."""
    SYNTH_TEST_DIR.mkdir(parents=True, exist_ok=True)
    sim = generate_synthetic_dataset(SYNTH_TEST_DIR, seed=42)
    return sim, SYNTH_TEST_DIR


def test_file_layout_and_column_names(synth_data):
    """(a) Test generator reproduces kit file layout and column names."""
    _, synth_dir = synth_data

    # Check files match real kit layout
    expected_files = [
        "metadata.json",
        "train/queries.csv",
        "train/history.csv",
        "train/labels.csv",
        "train/conditions.csv",
        "train/sample_submission.csv",
        "validation/queries.csv",
        "validation/history.csv",
        "validation/labels.csv",
        "validation/conditions.csv",
        "validation/targets.csv",
        "validation/reference.csv",
        "validation/sample_submission.csv",
        "test/queries.csv",
        "test/history.csv",
        "test/sample_submission.csv",
    ]

    for rel_path in expected_files:
        synth_file = synth_dir / rel_path
        real_file = REAL_PUBLIC_DIR / rel_path
        assert synth_file.exists(), f"Missing synthetic file: {rel_path}"
        assert real_file.exists(), f"Missing real reference file: {rel_path}"

        if rel_path.endswith(".csv"):
            with open(synth_file, newline="", encoding="utf-8") as sf:
                synth_cols = next(csv.reader(sf))
            with open(real_file, newline="", encoding="utf-8") as rf:
                real_cols = next(csv.reader(rf))
            assert synth_cols == real_cols, f"Column mismatch in {rel_path}: {synth_cols} vs {real_cols}"


def test_winning_cell_win_fraction(synth_data):
    """(b) Test fraction of ticks that are wins in winning cell is within 0.05 of hi."""
    sim, _ = synth_data

    for ctx, hi in HI_TABLE.items():
        n_ticks = sim.winning_cell_ticks_by_ctx[ctx]
        n_wins = sim.winning_cell_wins_by_ctx[ctx]
        assert n_ticks > 500, f"Insufficient ticks for context {ctx}: {n_ticks}"
        win_frac = n_wins / n_ticks
        diff = abs(win_frac - hi)
        assert diff < 0.05, f"Context {ctx}: win fraction {win_frac:.4f} differs from hi {hi:.4f} by {diff:.4f} (>= 0.05)"


def test_shared_change_spread(synth_data):
    """(c) Test shared_change flip ticks within an episode spread < 3 ticks."""
    sim, _ = synth_data
    assert len(sim.shared_change_spreads) > 0, "No shared_change spreads recorded"
    for i, spread in enumerate(sim.shared_change_spreads):
        assert spread < 3, f"Episode {i} shared_change spread {spread} >= 3"


def test_short_history_row_count(synth_data):
    """(d) Test short_history episodes have exactly 64 history rows per machine."""
    _, synth_dir = synth_data

    for split in ("train", "validation", "test"):
        # Map query_id -> condition for train/validation, or detect by count for test
        queries_path = synth_dir / split / "queries.csv"
        history_path = synth_dir / split / "history.csv"

        with open(queries_path, newline="", encoding="utf-8") as qf:
            queries = list(csv.DictReader(qf))

        with open(history_path, newline="", encoding="utf-8") as hf:
            histories = list(csv.DictReader(hf))

        # Count rows per (episode_id, machine_id)
        from collections import Counter
        counts = Counter((r["episode_id"], r["machine_id"]) for r in histories)

        if split in ("train", "validation"):
            cond_path = synth_dir / split / "conditions.csv"
            with open(cond_path, newline="", encoding="utf-8") as cf:
                cond_map = {r["query_id"]: r["condition"] for r in csv.DictReader(cf)}

            for q in queries:
                cond = cond_map[q["query_id"]]
                key = (q["episode_id"], q["machine_id"])
                if cond == "short_history":
                    assert counts[key] == SHORT_HISTORY_TOTAL_TICKS, (
                        f"Machine {key} in {split} short_history has {counts[key]} rows, expected {SHORT_HISTORY_TOTAL_TICKS}"
                    )
                else:
                    assert counts[key] == int(q["cutoff"]), (
                        f"Machine {key} in {split} {cond} has {counts[key]} rows, expected cutoff {q['cutoff']}"
                    )
        else:
            # In test, short_history episodes have exactly 64 rows per machine
            # and there should be 48 * 8 = 384 machines with 64 rows
            short_machines = [k for k, count in counts.items() if count == 64]
            assert len(short_machines) == 48 * 8, f"Expected {48*8} short machines in test, got {len(short_machines)}"
