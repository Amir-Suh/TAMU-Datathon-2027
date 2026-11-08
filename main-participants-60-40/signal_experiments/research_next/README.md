# Forecasting research execution

The primary arms of all 13 cards ran on released main data, with separate known-law fixtures where appropriate. W1, Z1, and S1 started in parallel with three GPT‑6.1 Sol agents at medium reasoning. Subsequent work used the same frozen main hybrid control, **56.341022/60, R = .064943404** on 96 validation episodes and 768 queries. The control recomposed byte for byte. The consolidated scorecard independently reads and re-scores 57 saved forecasts using the official condition denominators and 10,000 paired episode bootstrap draws. [Original plan](../FORECASTING_EXPERIMENTS.md) · [Scoreboard CSV](scoreboard.csv) · [Full score JSON](scores.json) · [Frozen control manifest](B0/plan.json).

## First wave

| Arm | Points / 60 | ΔR vs frozen hybrid | Paired 95% interval |
|---|---:|---:|---:|
| W1a pooled ordered 01/10 contrast | 56.380468 | −.000745 | [−.001434, −.000124] |
| W1b extra pair-specific deviations | 56.379653 | −.000730 | [−.001481, −.000029] |
| Z1 state offset fitted on sampled cutoff outcomes | 56.025064 | +.006006 | [+.002199, +.009805] |
| Z1 state offset fitted on other episodes' released odds, post-hoc | 56.521828 | −.003407 | [−.005275, −.001635] |
| S1 learned global blend | 56.418604 | −.001464 | [−.003010, −.000072] |
| S1 support-gated blend | 56.419747 | −.001486 | [−.003220, +.000110] |

- **W1.** W1a selected local strength 20 in training-episode folds with refitted upstream M3. It improved grouped cutoff Brier overall and on the inferred (4,7) slice and lowered public main R in all four conditions. Its main gain is concentrated in stable episodes. W1b did not pass the gate for W1c, so the type-specific arm stayed gated. [W1 forecasts and checks](W1/README.md)
- **Z1.** The primary state coefficient changes sign across folds fitted to sampled outcomes. The separate odds fit excludes each evaluated episode's targets and gives negative state coefficients in all four folds. Its public main score improves all four conditions and p0 MSE falls to .003975 from .004468. This comparison is a post-hoc sensitivity on previously examined validation with fixed, transductive GRU forecasts. It is the strongest development signal for a fresh holdout. [Z1 fit scopes and forecasts](Z1/README.md)
- **S1.** The global blend captures the measured gain. Support gating adds 0.001142 points over it, with paired ΔR −.000022 [−.000566, +.000532]. The entropy alternative scores 56.405187. The held folds exclude their labels from weight fitting; both parents are the archived forecasts, including their earlier self-supervised exposure to evaluation histories. [S1 score and coefficients](S1/README.md)

## Remaining experiment cards

| ID | Main comparison | Result and decision |
|---|---|---|
| B0 | 56.341022 | Replayed the frozen hybrid exactly and checked the frozen component scores against the organizer formula. [Inputs and scores](B0/scores.json) |
| P0 | 56.385873 | Raw type-tail counts improve by 0.044851 points; half-count smoothing is effectively identical. The projection holds p0 and p1 fixed to floating-point precision and uses only train histories to fit the ratios. [Scores](P0/scores.json) · [Inputs and ratio estimates](P0/plan.json) |
| W2 | 56.381529 | Reweighting lag and flip evidence adds 0.001061 points over W1a, with paired ΔR [−.000046, +.000002]. Known-law results are mixed. Use W1a as the simpler challenger. W2's normalized likelihood correction is conditional on fixed exposure and win totals. [W2 findings](W2/README.md) |
| Z2 | 55.172876 joint; 55.415852 payout-only | The sticky 128-tick joint HMM recovered the direction of the state effect on generated data. The public main result favors the hybrid; the sticky-versus-static interval spans zero. The approximate model conditions on prefix MAP lag rules. The explicit-duration arm stayed gated. [Z2 results](Z2/README.md) |
| S2 | 55.927330 exact; 55.864158 matched drop | Exact finite-state filtering improves its matched main short-history R from .10709 to .10224. It improved the matched point estimate at every older-history mask budget and passed 60 exhaustive path checks. The frozen hybrid remains ahead overall and on short history. [Mask curve and exact filter](S2/README.md) |
| C1 | 55.123745 polarity reset at 96 ticks | This fixed-emission, constant-hazard filter exactly sums geometric run lengths in a finite state and tracks posterior mean age. Polarity resets outperform pair or strength resets on released main. In generated sparse pair swaps 16 ticks before cutoff, pair resets reduce p0 MSE from .1220 for M3 to .0078. Rare-pair prior refits helped pair recovery in the generated sparse setting. [Main scores](C1/scores.json) · [Known-law comparisons and alarm delays](C1/synthetic_scores.json) |
| C2 | 55.580848 aligned; 55.594415 time-shifted | Aligned sharing improved the matched independent filter at 55.123745. The 11-tick shifted control slightly exceeds aligned on main, while aligned beats shifted on generated synchronized changes. Existing M4 with the same payout head scored 56.274094. This shared filter carries machine marginals between ticks. [Main comparison](C2/scores.json) · [Generated comparisons](C2/synthetic_scores.json) |
| N1 | 55.121922 evidence GRU; 55.052316 evidence TCN | The GRU improves the common causal structural control at 54.769686. The matched GRU-versus-TCN interval spans zero, and both trail the frozen hybrid. The sparse augmentation arm stayed gated. All 17 forecasts replayed with Torch imports and training reads blocked. [Neural controls, seeds and export](N1/README.md) |
| W3 | 56.011361 all 28 pairs | The residual fit uses 4,608 episode-cross-fitted training cutoff and pseudo-cutoff rows. All-pair signed features slightly outperform the three-pair residuals, while 25-category expansions score 55.140569. The sparse learned set retained the same three pair identities on main, with fold-specific rank order. The tensor and context-tree variants stayed gated. [Main scores](W3/scores.json) · [Folds and fit](W3/plan.json) · [Rare-pair stress](W3/synthetic_scores.json) |
| X1 | 27.026298 stand-alone; 55.669688 with 10% weight | One frozen Chronos tiny CPU benchmark, sampled 128 times per query and binned to the five payout outcomes. It trailed the hybrid in every condition. The replay was byte-identical. [Local scores](X1/scores.json) · [Checkpoint hash and settings](X1/plan.json) · [Public checkpoint](https://huggingface.co/amazon/chronos-t5-tiny/tree/29d808298f1a62493e7b9a5e08529d0d930fa189) |

The decision to carry a candidate forward uses the experiment's matched control and the future adaptation lockbox gate. The first-wave gains from P0, W1a and the S1 global blend are below the proposed practical threshold of .002 R. The largest gain clears that numerical threshold on reused main data and comes from the post-hoc Z1 odds fit. Keep the frozen hybrid as the control for a fresh holdout. W1a and the Z1 odds-fit hypothesis are the proposed two candidates for that comparison.

## Evidence budget

The nested training curve refits ordinary M3 on 24, 48, and 96 whole episodes, with the same payout mix and validation cutoffs. Scores are **56.156543**, **56.212745**, and **56.259032**. Win MSE improves from .004958 to .004711 over those budgets. This is one nested training draw and one model family. The S2 mask curve varies only observed history length on the same cutoffs and is reported separately. [Training-size curve](curves/scores.json) · [History-length curve](S2/masked_public_scores.json).

## Reproduce

Run from the participant-kit root. Individual W1, Z1, S1, W2, Z2, S2 and N1 reports include exact fit and verification commands.

```sh
d=signal_experiments/research_next
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B "$d/score_all.py"
python3 -B "$d/synthetic.py"
python3 -B "$d/P0/run.py" --out /tmp/house-edge-p0-replay
python3 -B "$d/C1/run.py" --out /tmp/house-edge-c1-replay
python3 -B "$d/C2/run.py" --out /tmp/house-edge-c2-replay
python3 -B "$d/W3/run.py" --out /tmp/house-edge-w3-replay
```

For X1, provision `chronos-forecasting==1.5.3`, PyTorch 2.8.0, and Transformers 4.57.6. Fetch the pinned public snapshot to a local directory. Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B signal_experiments/research_next/X1/run.py --checkpoint /path/to/local/snapshot`. The scorer uses the released validation split; test scores are private. Format-checked candidate test CSVs accompany P0, W1 and W2.

The adaptation kit was unreleased during these runs. The generated rare-pair, shared-flip, and nested-mask cases remain separate from contest validation. Once the release is available, use the existing adaptation stress runner and the plan's whole-episode lockbox. The main development scores should retain their current scope. [Adaptation runner](../05_adaptation_stress/README.md) · [Adaptation lockbox protocol](../FORECASTING_EXPERIMENTS.md#data-and-evaluation-contract)
