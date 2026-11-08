# House Edge subagent dispatch plan

The [experiment plan](EXPERIMENT_PLAN.md) defines the six cards and acceptance criteria. This runbook is ready for a coordinator to issue once execution starts. The coordinator owns the run manifest, frozen controls, adaptation lockbox, final assembly, and release decision.

## Setup and capacity

1. Work from the participant-kit root. Use the current checkout and disjoint writable directories under `signal_experiments/next_stage/<run-id>/`. Give each worker a single owner directory and a distinct output directory for every attempt. Treat existing code, forecasts, priors, checkpoints, and inputs as read-only. Keep each worker's changes in that directory until the coordinator integrates a selected candidate.
2. Create a read-only shared manifest after it is populated. Record source snapshot or file hashes, round and season, dataset and query hashes, main folds or adaptation fold manifest, candidate parent paths and hashes, software versions, and CPU limits. In this workspace the participant kit is Git-ignored at the parent repository and the current branch has no commits. Use file hashes for reproducibility.
3. Keep at most three workers active plus the coordinator. Give each numerical job one thread when two jobs run together. Run a two-thread benchmark or training job alone and schedule other workers for code, review, or reporting. Set the relevant numerical library limits before imports, including OpenMP, OpenBLAS, MKL, and vecLib on macOS. Start dynamic filters and neural work with 12 episodes. Expand after invariants pass and the forecast or training budget is measured.
4. Replay the two main controls and check their format. Keep their CSVs immutable. Check whether a clean, matching ZIP exists for each fallback. A format-checked CSV is one half of a submission. The package worker prepares the incumbent first and records any unavailable checkpoints or toolchain dependencies as soon as they are identified.

## Dispatch order

| Wave | Slots and ownership | Dependencies and stop point |
| --- | --- | --- |
| Main A | `state_tail` owns E1/E2. `lag_cells` owns E3. `package` owns incumbent replay and ZIP work. | Start after the control score and hashes are verified. E2 and E3 can run without E1. When a main candidate misses its gate, preserve the result and keep the incumbent eligible. |
| Main B | `sparse_filter` owns E4. `adapt_pairs` owns E5 synthetic fixtures. `package` continues in its existing slot. | Start each new worker when a Main A research slot frees. E4 begins with S2 and S1 controls. Its E3-integrated arm waits for a frozen E3 emission export and a separate matched comparison. |
| Main lock | `package` owns the clean-room run. The coordinator owns the decision and immutable CSV/ZIP pair. | Stop changing the main candidate at 8:15 PM Central on Nov 7. Use the best eligible, verified pair. The main Form deadline and adaptation release are 9:00 PM Central. |
| Adaptation A | `adapt_pairs` resumes E5 with the released kit. `neural` prepares E6. `package` prepares the adaptation incumbent package. | Coordinator verifies the kit and freezes development and lockbox membership before any validation-odds reads. E6 can smoke-test with the current structured baseline, then run the matched experiment once offsets are locked. |
| Adaptation B | `neural` completes E6. `adapt_pairs` supplies frozen carry, adaptation-only, and selected refit controls. `package` benchmarks eligible artifacts. | Coordinator can reuse `state_tail` or `sparse_filter` after a slot frees to rerun a gated component on adaptation development. Freeze at most two challengers and open the lockbox once. |
| Adaptation lock | `package` reproduces the chosen candidate. The coordinator preserves exact files and handles the organizer handoff. | Begin by 9:00 AM Central on Nov 8. The hard deadline is 11:00 AM Central. |

If slots or time are tight, run the incumbent package and the cheapest matched arms first. Carry synthetic E5 readiness work and unfinished main research into the adaptation window when the main release cutoff has passed. Select the current wave from the actual kit and time before issuing prompts.

## Prompt preamble for every worker

> Read the assigned experiment card, its prior results, and the coordinator's manifest. Work only inside your assigned directory. Keep the shared controls immutable. Use pre-cutoff observations, preserve missing ticks, and keep episodes together. The manifest defines whether archived transductive parents may be used. Fit on permitted training data and report reused-main, synthetic, strict-holdout, and transductive scores separately. The coordinator retains access to adaptation lockbox odds and runs the final scoring. Do the smallest matched comparison first, run the listed invariants, then expand only after the wave's gate. Return exact commands, file paths and hashes, scores, timing, and a recommendation tied to the card's gate. Flag an unavailable parent or reproducibility blocker immediately and keep working on independent checks.

Append one assignment below. Substitute the run ID, owner directory, and round data path before dispatching. Resume an existing worker for a dependent stage when its prior context is useful.

| Worker | Assignment to append |
| --- | --- |
| `state_tail` | **E1/E2.** Replay the exact hybrid blend and P0. Fit the static payout-state and win-coupling arms using grouped train pseudo-cutoffs. Separate changes to win, joint covariance and tail allocation. Sweep finite tail strengths against identity and hard projection. Export p0/p1 invariance checks, a calibrated state summary, and a normalized forecast for each locked arm. Read [Z1](research_next/Z1/README.md), [Z2](research_next/Z2/README.md), and [P0](research_next/P0/plan.json) first. |
| `lag_cells` | **E3.** Replay W1a/W1b and W2 with the same payout mix. Add a genuine four-cell evidence arm only after the forecast-only comparison, using the card's pre-flip mapping. Export the hypothesis grid, ordered rates and support, posterior win forecast, and a frozen (4,7) slice. Hand E4 a stable rates interface and source hash if the new arm advances. Start from [W1](research_next/W1/README.md) and [W2](research_next/W2/README.md). |
| `sparse_filter` | **E4.** Reuse S2's forward kernel and S1's global blend. Run matched drop/filter and constant/gated blend contrasts. Freeze the mask manifest for 24+16 and 24+8, and verify one short enumerated example and complete-history equivalence. Report frozen-head and end-to-end mask results separately. Attempt the E3 rate handoff only after both standalones pass. Start from [S2](research_next/S2/README.md) and [S1](research_next/S1/README.md). |
| `adapt_pairs` | **E5.** Before release, use the existing synthetic generator and stress runner for rare pairs, double flips, and late pair swaps. After release, verify the adaptation season, fit 28-pair discovery and carry/refit controls, then isolate dynamic tree features and multi-switch updates. Return pair ranks, fold-safe offsets and feature definitions to E6. Report mask-specific recovery, pruning and detection delays. Start from [C1](research_next/README.md#remaining-experiment-cards), the [generator](research_next/synthetic.py), and [adaptation stress](05_adaptation_stress/README.md). |
| `neural` | **E6.** Reuse the existing causal feature and NumPy inference implementations. Start with one seed on 12 episodes, check future-tick invariance and actual-cutoff signal routing, then train the paired three-seed arms under frozen offsets and masks. Return both standalone and hybrid forecasts, a portable weight export, and parity logs. Start from [N1](research_next/N1/README.md) and [factored GRU replay](01_fredy_replay/README.md). |
| `package` | **Reproduction.** Prepare the incumbent CSV/ZIP first. Export exact required weights and load them through a Torch-free frozen inference path. Check schema, package contents, target-round IDs, and offline Python 3.13 replay with the organizer mounts and resource limits. Score or train only as needed to reproduce the designated candidate. Return candidate and replay CSV hashes, normalized max difference, timing, and the ZIP path. See the [submission contract](../SUBMISSION_README.md) and [kit README](../README.md). |

## Artifacts and reporting

Use `<run-id>/<owner>/<attempt>/` below the run root. Use a fresh attempt directory for retries. Each completed attempt contains the following.

| Artifact | Required content |
| --- | --- |
| `manifest.json` | Round and season, owner, card and arm IDs, episode manifest and fit scope, source/input/control hashes, commands, environment, seeds, and timing. Record history exposure for each learned component. |
| `validation_<arm>.csv` | Predictions joined by exact query ID with columns `query_id,p_0,p_1,p_2,p_5,p_20`. For lockbox predictions write forecasts without reading targets. |
| `submission.csv` | The selected test forecast for a deployable attempt, written against the matching round's test queries. |
| `scores_vs_hybrid.json`, `scores_vs_m3.json`, `scores_vs_matched.json` | Main scorer outputs with the named control as baseline. For adaptation name files for the matched round controls. Keep mask and synthetic scores in separate files. |
| `slices.csv` | Slice definitions, query and episode counts, and paired differences or class contributions. Include the card's mechanism and target slice. |
| `REPORT.md` | Status as advance, hold, reject, or blocked, with reason, paths, commands, measured resources, invariance checks, and the next dependency. Package reports include clean replay logs. |

Every deployable arm needs its matched control, scored predictions, format checks, and a reproducible recipe. Treat a smoke result, oracle substitution, or main-soft-label sensitivity as development evidence. An agent that hits a failed assumption should hand back the smallest reproducible case and its strongest completed comparison.

## Coordinator checks

Run from the participant-kit root. The following verifies the main controls, then scores any selected main challenger twice. Replace `attempt` with a completed worker directory before running the challenger checks.

```sh
d=signal_experiments
m3=m5_m6_repair/results/repaired-m5-m6/m3/validation_predictions.csv
hybrid="$d/combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
tail="$d/research_next/P0/raw_counts.csv"

python3 -B "$d/score.py" --baseline "$m3" "hybrid=$hybrid" "tail=$tail"
python3 -B check_submission.py "$d/combined/m3_match_split_win_pair_l31_factor_gru_size_test.csv"
python3 -B check_submission.py m5_m6_repair/results/repaired-m5-m6/m3/submission.csv

attempt="$d/next_stage/<run-id>/<owner>/<attempt>"
val="$attempt/validation_selected.csv"
python3 -B check_submission.py "$val" --split validation
python3 -B "$d/score.py" --baseline "$m3" "candidate=$val" --out "$attempt/scores_vs_m3.json"
python3 -B "$d/score.py" --baseline "$hybrid" "candidate=$val" --out "$attempt/scores_vs_hybrid.json"
python3 -B check_submission.py "$attempt/submission.csv"
```

For adaptation, set the scorer's `--data` to the adaptation validation directory or a staged scoring subset, use `--points 40`, and provide a matching `--baseline` forecast. Set the checker's `--data` to the adaptation public directory when checking full validation or test forecasts. For the lockbox, construct an isolated scoring directory containing exactly the reserved episodes in `queries.csv`, `targets.csv`, `conditions.csv`, and `reference.csv`. Slice each candidate and control to those same query IDs. The scorer requires exact ID matches. Derive the reference denominators and bootstrap from the staged subset. Keep synthetic scores on their own SSE scale.

The coordinator independently re-scores the retained arms and checks changed files, hashes, leakage checks, and the validation/test format. Assemble E3/E4's selected win mass first, then E1's coupled vector using that win mass as its baseline, then E2's tail projection or shrinkage. Keep a control path with all new components disabled and the original hybrid formula intact. Re-score the exact assembled forecast and its single-component removals so an interaction cannot advance on individual results alone. Apply the same checks after any E5/E6 parent change.

Package with the kit builder after replay inputs are fixed. Keep the output ZIP outside the source directory.

```sh
python3 -B build_entry.py --source "$entry_source" --out "$entry_zip"
python3 -B check_submission.py "$replayed_csv" --data "$round_public"
```

Record a final decision with the candidate ID, held score, gate result, CSV and ZIP hashes, and the evidence of offline replay. The designated team submitter handles the Form and records its confirmation. Each lock window ends with a single chosen pair and a verified fallback when one is available.
