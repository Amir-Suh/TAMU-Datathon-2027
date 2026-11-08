# Bake-off plan: five architectures, one harness, pilots first

Written 2026-11-07 ~14:00 CT. Main closes 21:00 CT today (about 7 h). Round two runs 21:00 → Sunday 11:00.

## 0. Principles for this event
- Pilot, then read. Every arm runs a small grid in minutes; results land in `runs/<arm>/<variant>/`; `tools/compare.py` reads them after the wave. No gates between running and looking.
- All the deliberation is spent here, in design: data construction, the shared harness, what each arm must expose, and how arms get combined. Implementation then runs unattended.
- The one real analysis choice: with 24 validation episodes per condition, small gaps are noise. `compare.py` prints a paired episode bootstrap by default (**PROPOSED**; the analysing agent writes its own `runs/READING_RULE.md` if it prefers another rule).

## 1. What the data says about where points are (measured, see review)
1. True cutoff probabilities are sharply spread (445 of 768 validation queries have true `p_0` < 0.2 or > 0.8). A model that tracks per-machine state can beat pooled frequency by a wide margin; a model that smooths toward the mean cannot.
2. Lag effects differ per machine and sometimes flip sign; the rules are in the lags (up to 8) and neighbours, with regime changes.
3. Tick-level data is plentiful (139,944 + 141,392 + 282,664 history rows) but regime information is per episode (96 + 96 + 192 episodes).
4. `signal` exists only at the cutoff. It must enter in a cutoff-level stage.
5. Score weights the four conditions equally, relative to the pooled reference per condition. `short_history` machines have only 64 observed ticks, so most gain there will come from shrinkage and neighbours, not from deeper models.

## 2. Data construction (shared by all arms, built once in W0)
- Dense tensors per split: `y[E, 8, 256]` int8 payout class (0..4 for payouts 0,1,2,5,20; −1 = not observed), `mtype[E, 8]`, `signal[E, 8]`, `cutoff[E]`, `n_obs[E, 8]`; train/validation also `label[E, 8]` and `condition[E]`; validation also `target[E, 8, 5]` and `reference[E, 8, 5]`.
- Invariant (tested): everything an arm uses for a prediction at tick `t` comes from `y[e, :, :t]`; a perturbation test changes ticks ≥ t and must not change the output.
- Tick-level training rows ("self-supervised"): every observed `(e, m, t)` with `y ≥ 0` is a label whose features come from ticks `< t`, restricted to the same episode. Missing ticks stay unknown (not zero).
- Cutoff-level rows: the 768 train labels (hard) and 768 validation targets (soft, exact).
- Two fit modes for every arm:
  - `dev`: tick rows from train history; with `transductive=1` also from validation and test histories (their ticks are < cutoff, so legal and exactly what happens in the real run); cutoff rows from train labels only. Score on validation targets.
  - `final`: as `dev` plus validation labels/targets, then predict test. Used for the submission and for round two refits.
- Episodes are never split across folds. Grouped CV folds are fixed by hashing `episode_id` (same function as `kit/slot_forecasting/experiments.py::episode_folds`).

## 3. The five arms

### A. `glm` — hierarchical per-machine multinomial logit (primary bet)
- Model: softmax over 5 classes of `W · φ(history)`; φ = own payout class at lags 1..L (L up to 8; one-hot per class), win indicators of the other 7 machines at lags 1..2 (counts and per-type counts), recency-weighted rates at half-lives {4, 16, 64}, plus a bias. Missing lag = indicator 0 and a "missing" flag.
- Hierarchy: weights for an episode-machine = pooled global-by-type fit (trained on all tick rows) + a machine-specific deviation with L2 penalty λ (shrinks to the pooled fit). Fit by Newton/LBFGS in numpy/scipy; recency weights `2^(-(t_last - t)/τ)`; rows near the end count more.
- Why it should work: matches "machine-specific interactions, lags up to 8"; cheap enough to refit per machine at inference time (so it adapts to round two without retraining a big model); fully numpy.
- Pilot grid (≤ 12 variants, each < 5 min): L ∈ {2, 4, 8}; λ ∈ {1, 10, 100}; τ ∈ {32, 96, ∞}; neighbours on/off; transductive 0/1. Optional two-stage factorisation (win vs not, then size given win) as one variant.

### B. `gbdt` — global LightGBM on tick rows
- Features for row `(e, m, t)`: own lags 1..8 (class codes, −1 for missing), pair code of lags 1–2, EMA rates at {4, 12, 24, 64}, counts of wins over last {5, 20, 50} observed ticks, `n_obs`, machine type, other-machine aggregates at last 1/5/10 ticks (mean win rate, max payout, mean payout), short-vs-long divergence, tick position in the history.
- Training rows weighted toward the end of each machine's history (matches the cutoff query distribution); multiclass logloss; small trees, strong regularisation; fixed seed, `num_threads=1`, `deterministic=True`.
- Pilot grid (≤ 8 variants): recency weighting on/off, last-N-ticks only {64, 128, all}, leaves {7, 15}, transductive 0/1.
- Output exports to LightGBM text model; inference runs in the package via lightgbm pinned (or Treelite-free numpy tree evaluator if W6 finds lightgbm risky).

### C. `seq` — small causal sequence model (GPU train, numpy inference)
- Tick-synchronous grid: at each tick the 8 machines' payout tokens (5 classes + MISSING) with type embeddings; per-machine GRU state plus a pooled cross-machine message; head outputs 5 logits per machine for the next tick. Trained seq2seq on every observed next tick (masked), so one pass uses all 140k+ labels. Parameters ≤ 300k.
- Variants: GRU d=64 vs d=128; with/without cross-machine message; transductive pre-training then fine-tune on train labels; (optional) 2-layer causal transformer.
- Export weights to `.npz` and implement the forward pass in numpy float64; parity test against torch (< 1e-9) is part of the arm.
- Risk: 96 train episodes; overfit to regime specifics. Early stopping on a held-out set of whole episodes.

### D. `regime` — condition posterior + adaptive windows + changepoint/HMM
- D1: episode-level condition classifier from features (n_obs==64 flag, rolling mean-payout divergence at several windows, cross-machine synchrony, variance of 8-machine mean) trained on `conditions.csv` of train (+ validation in final); report held-out accuracy via grouped CV; output `cond_posterior[E, 4]` for the stacker.
- D2: specialist frequency/transition forecasters with condition-specific half-life and shrinkage (grid over half-life ∈ {8, 16, 32, 64, ∞}, strength ∈ {5, 20, 60}); routed by the posterior or hard flag.
- D3: changepoint-aware GLM-lite: two-window likelihood comparison per episode, using the post-change window only when evidence favours a change (shared_change).
- D4 (optional): HMM with K ∈ {3, 4}, lag-1 context emissions, forward algorithm in numpy.
- Pilot grid (≤ 10 variants).

### E. `stack` — blend plus cutoff-level calibrator (built after pilots)
- Inputs: each arm's validation predictions (dev mode) as `[E, 8, 5]` arrays, `cond_posterior`, `signal`, `mtype`.
- Stage 1: convex combination per inferred condition (weights on the simplex, minimise squared error against validation targets). Parameters: arms × 4 conditions (≈ 20) against 768 exact vectors, with grouped-by-episode CV to report an honest blended score.
- Stage 2: tiny multinomial logit on `[log p_blend, type one-hot, signal one-hot, type×signal]` fit on the cutoff-level rows (validation targets as soft labels in CV; train labels as an extra check). Keep only if it improves the grouped-CV score.
- Final weights fitted on all validation episodes; applied to `final`-mode arm outputs on test.

## 4. Evaluation harness (what every arm gets for free)
- `python -m tdx.run --arm A --variant name --mode dev --round main [--set k=v ...]` writes `runs/A/name/{val_pred.npy, val_pred.csv, test_pred.csv, score.json, meta.json}` and appends a line to `runs/LOG.md` (arm, variant, points, four ratios, seconds, mode).
- `score.json` comes from the kit's own `slot_forecasting.contest.contest_score` (same numbers `check_submission.py` prints) plus two-half tuning/audit scores.
- `python -m tdx.tools.compare runs/` prints one table sorted by validation points with the four condition ratios, runtime, and (PROPOSED) paired episode-bootstrap 95% interval of the points difference vs `pooled_recent` and vs the current best.
- `python -m tdx.tools.leak_test --arm A` perturbs history at ticks ≥ cutoff-k and asserts predictions at the cutoff do not change. It also checks that dev-mode fitting never reads validation labels/targets.

## 5. Decision flow after the pilot wave
1. Read `compare.py` output and per-condition ratios. Keep every arm whose points beat `pooled_recent` by more than the interval, drop the rest.
2. For the top two or three arms, run a refinement wave (W7+, specs written after the pilots) targeting the weakest condition of each.
3. Build the stack from the kept arms; accept it only if grouped-CV points beat the best single arm by more than the interval.
4. Freeze: choose arm set, write `final` mode outputs, build the entry (W6), run `repro_check.sh`.
5. Fallback chain if anything fails: stack → best single arm → `pooled_recent` (30.826). Upload a `pooled_recent` entry early so a valid main entry exists before any experiment finishes.

## 6. Timeline (Central, Saturday)
| Time | Step | Who |
|---|---|---|
| 14:00–14:30 | `00_setup_env.sh`, `01_link_data.sh`, `02_safe_entry.sh` → upload `pooled_recent` entry to the Main form | you (3 commands) |
| 14:30–15:15 | W0 harness (blocking) | Codex |
| 15:15–17:15 | W1 glm, W2 gbdt, W3 seq, W4 regime, W6 package skeleton in parallel | Codex |
| 17:15–18:00 | compare, write reading rule, choose arms; write refinement specs | Claude + you |
| 18:00–19:30 | refinement wave + W5 stack | Codex |
| 19:30–20:15 | W6 final package, `repro_check.sh`, `check_submission.py` | Codex + you |
| 20:15–20:30 | upload to Main form, save confirmation (45 min before close) | you |
| 21:00 | round two kit; `refit_round.sh` first run on unchanged code (README step 4) | you |
| 21:00–Sun 08:30 | adaptation experiments (overnight sweeps with stop conditions) | Codex |
| Sun 08:30–09:30 | final stack, package, upload (90 min before close) | you |

## 7. Round two (adaptation) design
- The kit changes "which temporal relationships recur". Arms that learn per-machine lag structure from the current round's own histories (GLM, regime, transductive GBDT) adapt automatically; globally trained weights (seq checkpoint, pooled GBDT) need a refit on the new train+validation+test histories.
- `refit_round.sh <new public dir>` runs: (1) unchanged main pipeline in `dev` mode on the new validation (points vs pooled — the README's step 4), (2) every arm's `dev` pilot grid with the main-chosen variants only, (3) stack refit on new validation targets, (4) `final` mode, package, check. All paths are arguments; `/main_data` available at rerun time is read-only extra training data and is treated as optional.
- Keep a fallback: if refit fails by Sunday 09:30, upload the best single arm or `pooled_recent` fitted on the new round.

## 8. What is deliberately NOT in the plan
Time-LLM, JEPA (dual encoders), full synthetic-simulator reverse engineering, distillation as a first step, external data, any use of private files. Each is parked for lack of evidence or time; revisit only if pilots show the structure that would justify them.
