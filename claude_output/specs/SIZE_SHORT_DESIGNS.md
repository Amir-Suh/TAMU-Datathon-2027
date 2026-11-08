# Five designs for payout size and short_history (from the Codex adversarial review, 2026-11-07)

Status: DESIGNS ONLY, nothing implemented. Each design is an arm in its own folder `src/tdx/arms/<name>/` with the standard interface (`fit(rnd, spec)`, `predict(split)`, numpy-only inference, `save/load`), scored with the leak-free tools: `python -m tdx.tools.tail_eval` (win/size parts, paired bootstrap, reliability; holds 30/60/90) and `python -m tdx.tools.short_backtest` (real short_history layout). Both tools are being built by the `eval_tools.md` agy job; do not start until they exist and their tests pass. Every new arm needs `python -m tdx.tools.leak_test` and tests; a submission candidate additionally needs `kit/check_submission.py` and `claude_output/sh/repro_check.sh`.

## Shared protocol (read first)
1. **Evidence rule.** Advance a design only if it improves, on the SAME held-out tail windows (holds 30, 60, 90) and with the same seeds as the incumbent, both (a) the size-part log-loss on wins by at least the threshold given below, and (b) the full Brier of the DEPLOYED blend (incumbent with the component swapped in), each with a paired episode bootstrap 95% interval excluding zero. Confirm across 3 matched seeds before touching the 12-member ensemble. Validation true probabilities are NOT used to choose or fit anything (they are for the final report only).
2. **Inner holdout for anything calibrated or learned on top of a model.** A calibrator must be fit on predictions the base model did not see. For an outer evaluation at hold H (tail = ticks [c-H, c-H+30)), fit the base model on history truncated at c-H-30, predict the inner window [c-H-30, c-H), fit the calibrator there, then apply the calibrator to the base model refit at truncation c-H. At deployment (no tail), the same recipe uses the last 30 observed ticks before the cutoff as the inner window. Never fit a calibrator on a checkpoint's own training ticks (the GRU memorises them).
3. **Baseline to beat (held-out tail, models refit with the tail hidden, 3 windows, ~60.8k outcomes).** Brier: m3 0.4066, GRU x3 0.4189, split-duty blend (win .7 m3 + .3 GRU, size from GRU) 0.4030. Size-part log-loss (hold 30): GRU 1.0099, m3 1.0234, GBDT 1.0306, pooled_recent 1.0379. Real short_history reference: GRU ratio ~0.17, m3 0.107 (validation, 24 episodes; noisy).
4. **Do not use `signal` at historical ticks.** It exists only at the cutoff (AGENTS.md); size designs must not feed it into per-tick features.
5. **Time boxes are hard.** Each pilot is one matched seed and one intervention; stop and report at the box.

---
## Design 1 — `sizecal`: size aggregation plus a small calibration layer (0.5-1.5 h; agy-suitable)
**Why:** cheap, targets systematic bias in the ensemble's size mix. (The Codex check that equal-weight vs win-weighted aggregation of members differs by 0.0016 in conditional probability is already measured: validation score 56.580 vs 56.581, so aggregation is closed; this design is the calibration part only.)
**Mechanism:** take the deployed size mix s (4 conditional probabilities) and apply  s' = softmax( log s / T + b ), with ONE global temperature T and 3 class biases b (the 4th fixed at 0), fit by minimising size-part log-loss on the inner-window wins with a ridge penalty pulling (1/T, b) to identity (lambda chosen by 3-fold CV by episode inside the inner window). Optional second stage: separate (T, b) by machine type only if the global fit improves held-out size log-loss by >= 0.003 nats.
**Files:** `src/tdx/arms/sizecal/{__init__,arm.py}`; the arm wraps an existing base (`seqens` or `seq`) and a fitted calibrator JSON (5 numbers). `predict` = base size mix -> calibrate -> recombine with the base P(win): p = [1-pw, pw*s'].
**Test:** identity calibrator reproduces base predictions to 1e-12; calibrated rows sum to 1; leak test.
**Keep/drop:** keep if size-part log-loss improves >= 0.003 nats on untouched tails (holds 30/60/90) and deployed Brier improves with an interval excluding zero; drop if only reliability-bin statistics improve or if it needs per-type/per-condition parameters to work.

## Design 2 — `sizehead`: nonlinear residual head on the GRU's size logits (1-3 h; Codex or agy)
**Why:** the factored GRU's size logits are linear in one hidden state; the probe shows the state encodes the right quantities (ticks since win R2 0.90, latest size 72%, running shares R2 0.72/0.41), but the readout may be too weak.
**Mechanism:** subclass `SeqArm`; freeze the trained backbone; add a residual MLP  128 -> 16 -> 4 (tanh) whose output is added to the 4 size logits (win logit untouched). Train ONLY the residual on win ticks of all self-supervised histories (train+validation+test, ticks < cutoff) with size cross-entropy plus L2; select epochs/weight decay on the inner holdout window (design 1 protocol), not on the training ticks. Warm-start the residual output layer at zero so the arm equals the baseline at step 0.
**Files:** `src/tdx/arms/sizehead/{__init__,arm.py}`; numpy inference = the existing float64 GRU recurrence plus two small matmuls; save the residual weights next to the backbone arrays.
**Test:** zero residual reproduces `seq` predictions exactly; numpy-vs-torch parity < 1e-9; leak test.
**Keep/drop:** advance for >= 0.005 nats size-part improvement with favorable paired Brier across 3 matched seeds, AND the gain survives recombination with m3's P(win). Drop if hidden-state decoding improves but next-size prediction does not, or if the gain disappears on tails not used for model selection. Only if the frozen-head pilot wins, try fine-tuning the last GRU layer jointly (second pilot).

## Design 3 — `sizeshrink`: hierarchical per-machine size correction with learned shrinkage (1-3 h; Codex)
**Why:** the true size mix varies by machine (sd ~0.12 on the 1x share) but raw counts barely help (0.0285 vs 0.0316 for the type prior); a learned-shrinkage correction on TOP of the GRU may capture what is left.
**Mechanism:** for machine m with n_m past wins and size counts c_m (1x,2x,5x,20x): correction logit  d_m = beta * (c_m/n_m - s0) * sqrt(n_m / (n_m + kappa))  where s0 is the GRU's average predicted size mix over that machine's past wins (computed from the inner-holdout checkpoint, NOT the full checkpoint, because the GRU memorised its own history), and final size logits = GRU logits + d_m. Two global parameters (beta, kappa), optionally three machine types x two parameters. Fit by minimising held-out size log-loss on the inner window (design 1 protocol).
**Files:** `src/tdx/arms/sizeshrink/{__init__,arm.py}`; needs per-machine win counts only (numpy).
**Test:** beta = 0 reproduces the base; shrinkage monotone in kappa; leak test; a unit test that the correction uses only ticks < cutoff.
**Keep/drop:** keep if size-part log-loss improves >= 0.003 nats on tails not used for fitting. Drop if beta fits near 0 or if the gain is within the paired interval (it duplicates what the GRU already tracks: the probe says the state holds running shares).

## Design 4 — `shortctx`: exact-layout inference for short histories (2-4 h; Codex, design carefully)
**Why:** the review confirmed that our tail backtest never presents the real layout (40 consecutive + 24 older) to short_history episodes; GRU ratio there is ~0.17 vs m3's 0.107. A model trained on exactly that layout may close the gap.
**Training-example generator:** for every long episode machine (and the real short ones) and many random pseudo-cutoffs c' (between tick 80 and the true cutoff): build the exact layout (the last 40 observed ticks before c' plus 24 uniformly sampled older observed ticks), label = the realized outcome at tick c' (class 0..4). This yields a very large supervised set of REAL next-tick outcomes; no validation/test labels are involved (outcomes at ticks < the true cutoff are histories).
**Model:** input is a set of 64 items, each = (payout class embedding, tick offset before c' as a bucketed/learned embedding, machine type); a small causal transformer or GRU over the 40 consecutive ticks plus a pooled summary of the 24 older items (mean of embeddings) concatenated to the final state; head = factored win/size (5 logits as in `seq`). Train on all layouts (so it also works as the generic head), with a layout-flag input if useful. Numpy inference.
**Files:** `src/tdx/arms/shortctx/{__init__,arm.py,sampler.py}`; the sampler is deterministic given a seed and unit-tested.
**Test:** the sampler produces exactly 64 observed items with the stated offsets; predictions ignore ticks >= cutoff; leak test; parity < 1e-9.
**Scoring:** `short_backtest` (real short episodes, last 8 observed ticks hidden) plus the standard tail on long episodes. Use it ONLY for short_history episodes (known from `n_obs == 64`) in the blend, as a specialist; do not replace the generic model.
**Keep/drop:** keep if, on the real short_history backtest, Brier improves over both GRU and m3 with an interval excluding zero (>= ~6k outcomes); drop if it merely matches m3 (m3 is already the specialist there).

## Design 5 — `sizeswitch`: switching model for persistent size residuals (2-4 h; lowest priority)
**Gate first (30 minutes, no arm):** compute, for each machine, the lag-1..5 autocorrelation of the size-class indicator sequence over its WINS after removing the machine's own mean and the type mean (permutation baseline: shuffle sizes within machine 2,000 times; use all splits' histories). If the autocorrelations are not significantly above the permutation null, STOP: there is no temporal size structure to switch on (consistent with the size-diagnostic result that recent windows were worse than all-history counts).
**If the gate passes — mechanism:** K=2 (maybe 3) hidden size-mix regimes per machine type, categorical emission over (1x,2x,5x,20x) given a win, sticky transition matrix, shared parameters per type learned by EM across all machines of that type; per machine, the forward filter over its win sequence gives the regime posterior; prediction = sum_k posterior_k * mix_k blended with the GRU mix by a single weight fitted on the inner window.
**Files:** `src/tdx/arms/sizeswitch/{__init__,arm.py}` (numpy EM, deterministic initialisation).
**Test:** K=1 equals the static per-type mix; EM log-likelihood non-decreasing; leak test.
**Keep/drop:** keep only if held-out size-part log-loss improves >= 0.003 nats AND the regimes are persistent and distinct (learned transition diagonal >= 0.9 and an L1 distance of at least 0.2 between the regimes' emission mixes); drop otherwise.

---
## Order and ownership
| # | Design | Depends on | Owner suggestion | Expected cost |
|---|---|---|---|---|
| 1 | sizecal | tail_eval | agy (mechanical, tight spec) | 0.5-1.5 h |
| 2 | sizehead | tail_eval, seq arm | Codex (arm + numpy parity) | 1-3 h |
| 3 | sizeshrink | tail_eval, inner-holdout helper from 1 | Codex | 1-3 h |
| 4 | shortctx | short_backtest | Codex (largest design) | 2-4 h |
| 5 | sizeswitch | gate diagnostic | agy runs the gate; Codex only if it passes | 0.5 h + 2-4 h |

Build the shared **inner-holdout helper** once (design 1) and reuse it in 2, 3 and 5. Run the gate for design 5 and the sampler tests for design 4 before any training. Report every pilot as: design | seed | size-part log-loss delta vs incumbent | Brier delta of the deployed blend | paired 95% interval | decision.

---
## Design 6 — `moegate`: learned soft mixture of experts inside one model (added 2026-11-07 evening)
**Why:** per-condition separate models are not workable (the condition is unknown at test time except short_history; condition-based gates and weights gave +0.01; only 24 labelled train episodes per condition). A SOFT gate learned inside one model sidesteps the labels: the network decides from the history itself which expert to trust, which is what the data-driven blend (m3 for P(win), GRU for size) already does by hand.
**Mechanism:** shared trunk = the working sequence encoder (GRU from `seq`, or the conv+ALiBi transformer from `tfm2`); K=3 expert heads, each a small MLP (d -> 64 -> 5 logits, factored win/size output as in `seq`); a gate g = softmax(W_g [pooled trunk state at the last tick, short flag, rule-confidence features]) with rule-confidence features = the pair hit rates from `tdx.arms.gbdt.features` (columns `pair*_acc_all`, `pair*_acc_rec`, 15 per machine, causal). Output = sum_k g_k p_k (a mixture of PROBABILITY vectors, not of logits). Train end to end with next-tick cross-entropy on every observed tick of all histories (train+validation+test, ticks < cutoff), plus: (a) a load-balancing penalty so each expert gets >= 15% of the average gate mass, (b) expert dropout (drop one expert's probability at random with prob 0.1 and renormalise), (c) entropy bonus on the gate early in training.
**Interface/files:** `src/tdx/arms/moegate/{__init__,arm.py}`, numpy float64 inference (reuse the `seq`/`tfm2` forward and add the gate), torch-vs-numpy parity < 1e-6, exact save/load, torch only imported inside `fit`; a torch-free loader module like `seqens`/`tfmens` for packaging.
**Test:** K=1 reproduces the base arm; gate rows sum to 1; leak test; parity.
**Scoring:** `tail_eval` (holds 30/60/90) against the deployed blend of the same base model seeds (the comparison is "one MoE seed vs one base seed", then 3 matched seeds), AND the real `short_backtest`. Report expert usage by condition (validation) and by short flag to see whether the experts specialise.
**Keep/drop:** keep if, over 3 matched seeds, the deployed blend with the MoE swapped in improves tail Brier with a paired interval excluding zero AND at least one expert's usage differs materially between short_history and the other conditions (otherwise it is just a bigger network). Drop if gate entropy collapses to one expert or experts are interchangeable.

## Design 4 (`shortctx`) — implementation notes added for the hand-off
- **Sampler first.** Implement `src/tdx/arms/shortctx/sampler.py` with `sample_layout(y_machine, cutoff_prime, rng)` -> (items[64], offsets[64]); tests: exactly 64 observed items, the last 40 are ticks cutoff_prime-40..cutoff_prime-1, the 24 older are distinct and < cutoff_prime-40, nothing at or after cutoff_prime; deterministic given the seed. Pseudo-cutoffs only where the machine has >= 64 observed ticks before them.
- **Training mix.** 50% exact-layout samples, 50% full-history samples (so the same weights serve long episodes), with a layout-flag embedding; the flag is derived from the observed count == 64, never from `n_obs` fields or condition labels.
- **Use in the blend.** Only for episodes with `n_obs == 64` for all machines (short_history); for them the blend weights are re-fitted on the real `short_backtest` windows, not on validation. Everything else is unchanged.
- **Acceptance numbers to beat (real short_history backtest, 96 episodes, 6,144 outcomes, Brier):** m3 0.3645, GBDT 0.3688, GRU 0.3687; size part GRU 1.0166 best, m3 1.0313.

## Which agent runs what
| Design | Agent | Why |
|---|---|---|
| 4 `shortctx` sampler + tests, then training | Codex (`gpt-6-astra`, workspace-write, clone) | largest, needs careful numpy parity |
| 6 `moegate` | Codex after 4's trunk plumbing exists, or agy if the spec is cut to the GRU trunk only | new architecture; needs parity and leak tests |
| 1 `sizecal`, 5 gate | agy | mechanical, one-command proof |
| 2 `sizehead`, 3 `sizeshrink` | Codex or agy after 1's inner-holdout helper exists | small arms |
