"""Re-score the research suite's saved main-validation forecasts against one control."""
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
KIT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
from score import compare

BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
ARMS = {
    "B0": {
        "m3": KIT / "m5_m6_repair/results/repaired-m5-m6/m3/validation_predictions.csv",
        "frozen_replay": HERE / "B0/frozen_hybrid_replay.csv",
    },
    "P0": ("raw_counts", "half_count"),
    "W1": ("W1a_validation_predictions", "W1b_validation_predictions"),
    "Z1": ("intercept", "validation_predictions", "secondary_soft_intercept", "secondary_soft_intercept_state"),
    "S1": ("validation_equal", "validation_global", "validation_predictions", "validation_entropy"),
    "W2": ("validation_predictions",),
    "Z2": ("static_joint", "static_payout", "sticky_16_joint", "sticky_16_payout",
           "sticky_64_joint", "sticky_64_payout", "sticky_128_joint", "sticky_128_payout"),
    "S2": ("validation_drop_constant", "validation_drop_flip", "validation_exact_constant",
           "validation_predictions", "validation_expected_constant", "validation_expected_flip"),
    "C1": ("fixed", "polarity_32", "polarity_96", "pair_32", "pair_96", "strength_32", "strength_96"),
    "C2": ("m4_same_size", "independent", "aligned", "shifted", "permuted"),
    "N1": ("structural", "gru_raw", "gru_pair_evidence", "tcn_raw", "tcn_pair_evidence"),
    "W3": ("m3_same_mix", "main_three", "train_three", "all_28", "train_three_full_categories",
           "main_three_native", "train_three_native", "all_28_native", "train_three_full_categories_native"),
    "X1": ("chronos_tiny", "fixed_10_percent"),
}


def main():
    forecasts = {}
    for experiment, arms in ARMS.items():
        paths = arms if isinstance(arms, dict) else {name: HERE / experiment / f"{name}.csv" for name in arms}
        forecasts.update({f"{experiment}/{name}": path for name, path in paths.items()})
    result = compare(KIT / "public/validation", BASE, forecasts, 60)
    assert abs(result["models"]["control"]["points"] - 56.34102223545075) < 1e-10
    (HERE / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for name, r in sorted(result["models"].items(), key=lambda item: item[1]["R"]):
        rows.append({"candidate": name, "points": r["points"], "R": r["R"], "delta_R": r["delta_R"],
                     "delta_R_ci_low": r["delta_R_95"][0], "delta_R_ci_high": r["delta_R_95"][1],
                     **{f"{c}_R": v for c, v in r["condition_R"].items()}, "p0_mse": r["p0_mse"],
                     "weighted_positive_SSE": r["win_mass_weighted_conditional_payout_sse"],
                     "forecast": str(Path(r["path"]).relative_to(KIT)), "sha256": r["sha256"]})
    with (HERE / "scoreboard.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rescored {len(rows) - 1} saved arms plus the frozen hybrid across 96 episodes.")
    for row in rows[:12]:
        print(row["candidate"], f'{row["points"]:.6f}', f'{row["delta_R"]:+.8f}')


if __name__ == "__main__":
    main()
