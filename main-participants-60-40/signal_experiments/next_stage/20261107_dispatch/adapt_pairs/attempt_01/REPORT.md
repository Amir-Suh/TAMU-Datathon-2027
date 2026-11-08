# E5 synthetic readiness

**Hold for adaptation development.** The 28-pair discovery control, exact C1 polarity filter and pruned swap-enabled filter are ready for matched tests on the released kit. This run uses generated histories and exact generated cutoff probabilities. Its errors are raw five-class SSE averaged over machines.

The final run fits 48 stable episodes from seed 587200 and evaluates 48 independent episodes per scenario from seeds 587400–587403. Each episode contains eight machines. The cutoff is 200 in every case. The four scenarios use the rare pairs (1,8), (2,7) and (3,6), with no change, polarity flips at 90 and 170, a pair swap at 184, or a pair swap at 192. The masked arms retain 24 deterministically selected observations before tick 160, then the final 16 or 8 observations. The mode-off law and payout tail follow the existing synthetic fixture, with local schedules and a history-only prediction entry point.

## Matched results

Lower raw SSE is better. Intervals in the score file resample whole episodes.

| Scenario and history | Control | Changed arm | Control SSE | Changed SSE | Paired difference and 95% interval |
|---|---|---|---:|---:|---|
| Stable, 24+16 | Static main | Static discovered prior | 0.14887 | 0.02095 | −0.12792 [−0.14962, −0.10781] |
| Stable, 24+8 | Static main | Static discovered prior | 0.21768 | 0.12828 | −0.08941 [−0.11174, −0.06730] |
| Stable, 24+16 | Static discovered prior | Static refit | 0.02095 | 0.02128 | +0.00033 [−0.00033, +0.00106] |
| Double flip, full | Segment-adaptive single flip | Segment-adaptive multi-flip | 0.50359 | 0.00311 | −0.50048 [−0.51762, −0.48289] |
| Double flip, 24+16 | Segment-adaptive single flip | Segment-adaptive multi-flip | 0.02192 | 0.02185 | −0.00008 [−0.00160, +0.00182] |
| Swap at 184, full | Multi-flip | Multi-flip with 0.25 swap probability | 0.23236 | 0.02701 | −0.20535 [−0.23217, −0.17945] |
| Swap at 192, full | Multi-flip | Multi-flip with 0.25 swap probability | 0.29263 | 0.11606 | −0.17657 [−0.19905, −0.15551] |
| Swap at 184, 24+16 | Multi-flip | Multi-flip with 0.25 swap probability | 0.03314 | 0.03254 | −0.00060 [−0.00184, +0.00023] |
| Swap at 192, 24+8 | Multi-flip | Multi-flip with 0.25 swap probability | 0.15026 | 0.15020 | −0.00007 [−0.00097, +0.00082] |

Two empirical Bayes updates start from uniform responsibilities over 28 pairs and mix in 15% uniform mass after each update. They use fixed main high and low emissions and stable adaptation-fit histories. (2,7), (1,8) and (3,6) moved from ranks 17, 18 and 14 to ranks 1, 2 and 3. Their fitted masses were 0.2935, 0.2872 and 0.2839. Stable 24+16 pair top-1 rose from 31.8% to 83.1%. Stable 24+8 rose from 0.0% to 43.2%. The 0.50, 0.80 and 1.00 warm-start cell and payout weights were close to the prior-only arm on sparse cases.

The matched single-adaptive and multi-flip arms share a 64-state deterministic beam, 1/128 elapsed-tick hazard, and beta strength 48. The single arm stops resetting once it has flipped. On full double flips, multi-flip recovered the correct joint MAP rule in 100% of machines after each flip, with median delay two ticks each. Single-adaptive recovered the first flip in two ticks and recovered the second on 0%. Exact C1 fixed-emission multi-flip reached 0.00271 SSE on the same full double-flip case. Its median per-case runtime was 0.086 s, compared with 0.493 s for multi-flip BOCPD. Native M3 and M4 scored 0.57137 and 0.58640 on that case.

With swaps enabled, full-history cutoff pair top-1 was 73.2% for tick 184 and 40.4% for tick 192. The correct joint MAP rule appeared after the change in 76.6% and 44.8% of machines, with median observed delays of seven and five ticks. The reset posterior alarm threshold of 0.15 fired after those swaps on 12.2% and 12.8% of machines. It fired at least once after tick 24 in 39.1% of full stable histories with swaps enabled and 51.8% with flip-only. Sparse rule recovery can reflect little evidence for the prior regime and should be interpreted alongside reset alarm coverage.

For elapsed-hazard multi-swap, mean discarded posterior mass per tick was 0.064% with full stable history, 0.730% for stable 24+16 and 0.757% for stable 24+8. Suppressing resets on missing history ticks reduced the two sparse figures to 0.110% and 0.105%. It had no full-history effect and changed sparse SSE in both directions. BOCPD uses unit likelihood and freezes sufficient statistics for a missing outcome, and skips segment count updates when either required lag is missing. The cutoff forecast advances the hazard once before averaging over growth and reset states.

The matched 31-leaf trees use scikit-learn HistGradientBoosting. They share an identical payout head, pseudo-cutoffs, masks, target rows, and training settings. The dynamic features use six pairs ranked on other whole episodes plus one causal best-of-28 summary. Each pair has current match, full and trailing 40 accuracy, their inferred rule predictions, and corresponding support. Accuracy updates require both lags and the observed outcome. Dynamic features lowered SSE in all four 24+16 fixtures by 0.10756–0.12676 and worsened both full swap cases by 0.15212–0.17664. The dynamic tree remained behind the structured comparisons on those sparse scenarios.

## Artifacts and verification

- [Final manifest](known_law/manifest.json), [frozen predictions and filter traces](known_law/predictions.npz), [synthetic scores](known_law/synthetic_scores.json), [slice CSV](known_law/slices.csv), and [pair rankings and grouped tree folds](known_law/pair_ranks.json).
- The final [fit-predict audit](known_law/fit_trace.jsonl) and [separate score audit](known_law/score_trace.jsonl) exited successfully. There were no public-data reads or denied file opens. The fit audit recorded local platform metadata helpers `uname` and `file`. The scoring process verified the prediction SHA-256 before generating exact cutoff targets and true regime labels.
- An independent path enumeration matched the unpruned filter on the short masked fixture within 2.8e-16 in all four switch/hazard variants. Row order, missing padding, mask nesting and tree target/future invariance also passed. [Enumeration results](verification.json)
- The final fit-predict run took 43.7 s on one numerical thread. Median per-case runtime for multi-swap was 0.536 s for 384 queries. The matched 12-episode preflight used a distinct evaluation seed from the final run. [Matched smoke manifest](smoke_matched/manifest.json)

From the participant-kit root, replay into a fresh directory under this attempt.

```sh
dispatch=signal_experiments/next_stage/20261107_dispatch
attempt="$dispatch/adapt_pairs/attempt_01"
replay="$attempt/replay"
mkdir "$replay"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -B "$dispatch/coordinator/audit_run.py" --trace "$replay/fit_trace.jsonl" --stage fit-predict "$attempt/run.py" fit-predict --out "$replay"
python3 -B "$dispatch/coordinator/audit_run.py" --trace "$replay/score_trace.jsonl" --stage score "$attempt/run.py" score --out "$replay"
python3 -B "$dispatch/coordinator/audit_run.py" --trace "$replay/verification_trace.jsonl" --stage fit-predict "$attempt/verify.py" "$replay/verification.json"
```

Once the coordinator verifies the adaptation kit and freezes its grouped split, carry forward strict train-only prior discovery and the exact C1 control alongside the proposed pruned swap filter. This fixture runner assumes a common cutoff and a fixed synthetic history schema. An adaptation runner needs per-row cutoff handling or equal-cutoff batches and independent configuration for strict versus transductive discovery. The tree uses neutral signal for historical pseudo-cutoffs and the recorded signal at the actual cutoff. Its fixed-endpoint implementation needs a per-row actual-cutoff check for heterogeneous batches.
