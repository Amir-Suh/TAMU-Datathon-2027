# Remaining forecast error and research plan

The 56.3410 hybrid leaves substantial measurable error. Replacing only its win probability with the released true win probability gives **58.0446/60**. Replacing only its conditional payout mix gives **58.1384/60**. Each oracle substitution uses released answers to isolate one component's headroom while keeping the other component fixed.

Averaging each head's change in loss when corrected first and when corrected second assigns **48.7%** of the current contest loss to the win head and **51.3%** to the payout mix. That attribution includes each head's share of the interaction. It differs from comparing the marginal squared errors of the five classes.

## Common main validation comparison

All 96 episodes and 768 queries. Each condition contains 24 episodes and 192 queries. The four condition columns use the official loss ratio, where lower is better. The projections and slices below use the same episode and condition weighting. [Full model and slice diagnostics](diagnostics.json).

| Model | Points / 60 | Stable R | Short-history R | Drift R | Shared-change R |
|---|---:|---:|---:|---:|---:|
| M1 frequency | 31.092 | .9617 | .9793 | .9333 | .8448 |
| M2 lag pair | 52.598 | .0817 | .1192 | .2113 | .1508 |
| Frozen M3 | 55.268 | .0772 | .1069 | .0798 | .0785 |
| Shared-flip M4 | 55.283 | .0773 | .1058 | .0798 | .0783 |
| Repaired M5 | 54.842 | .0838 | .1167 | .0885 | .0872 |
| Repaired M6 | 55.282 | .0772 | .1067 | .0795 | .0780 |
| Pair GBDT | 54.290 | .1203 | .1261 | .0796 | .0947 |
| Seven plain GRUs | 55.419 | .0644 | .1150 | .0817 | .0695 |
| Three factored GRUs | 55.406 | .0633 | .1212 | .0813 | .0657 |
| Full three-way average | 56.064 | .0619 | **.0918** | .0634 | .0637 |
| Split-match M3 + GBDT payout mix | 55.833 | .0628 | .1007 | .0636 | .0714 |
| Split-match M3 + GBDT/factored-GRU payout mix | **56.341** | **.0511** | .0929 | **.0553** | **.0604** |

M3's flip model recovered most of M2's extra loss in drift and shared change. M4's sharing barely changes normal-cutoff scores. The GRU improves stable and shared-change scores over M3 but gives some of that back on sparse history. The current hybrid gains in every condition, with its smallest gain on short history. The full average's short-history edge over the hybrid is just .00116 in R, with a paired 95% interval of [−.01428, +.01050]. A support-based blend remains a hypothesis. [Prior controlled masking study](../05_adaptation_stress/README.md).

### Histories and lag evidence

- **Short history contributes 35.8% of the remaining R from 25% of queries.** Every validation query, including the sparse ones, has all last 40 ticks observed. The gap in observation coverage lies in the older history. The median number of observed historical outcomes with both lags available for M3's top pair is 36 in short histories and 220 in full histories. The short-history oracle win replacement reduces R from .0929 to .0417, versus .0518 when replacing only the payout mix.
- **The inferred (4,7) pair contributes 45.3% of remaining R from 32.3% of queries.** It has the highest mean squared error of the three inferred pairs in each condition. For example, stable SSE is .0130 for (4,7), versus .0085 and .0087 for (1,3) and (2,5). On short histories those values are .0227, .0084, and .0189. The pattern is also present within each machine type. Pair identities come from a history-only frozen M3 posterior, so these groups are estimates of rule identity.
- **Pair selection alone has limited reach on this split.** 717 of 768 rows have at least .99 posterior mass on the leading pair and account for 90.5% of remaining R. The 34 rows below .90 account for 6.4% of R. With fixed M3 parameters, only three rows have more than .5 posterior mass on a flip in the last 40 ticks. Main validation gives little leverage for comparing methods on genuinely recent flips.
- The highest-error tenth of queries accounts for **47.1%** of R. Short histories account for 31 of those 77 queries. Future experiments should verify both the full score and the diagnosed slices.

### Payouts and hidden state

| Payout | Share of the hybrid's remaining R |
|---|---:|
| 0× | 36.4% |
| 1× | 29.1% |
| 2× | 26.6% |
| 5× | 7.2% |
| 20× | 0.7% |

Rows with true win probability at least .50 account for **72.2%** of R. This is a scoring diagnostic using the released odds. Any predictor or routing rule must use observable support or forecasts instead of that label. Win-error mass is concentrated on jackpot machines while 1× error is more prominent on steady machines. Against M3, the hybrid lowers win-mass-weighted conditional payout SSE from .01020 to .00645 and p0 MSE from .00471 to .00447.

The exact validation payout mix has two conditional 1× levels within each type, with a fixed 2×:5×:20× ratio. The high-1× mode is informative for the win head as well. The hybrid overestimates win probability by **2.7 percentage points** on that mode and underestimates it by **3.2 points** on the low-1× mode. These groups come from the released targets. The underlying mechanism remains open. The pattern motivates testing a history-inferred state that affects both components. The factored GRU's conditional 1× forecast already discriminates the two modes with per-type AUC between .860 and .888. Any state model should compete with that neural signal as well as with the simpler logistic decoder. [Previous payout-state experiment](../02_payout_state/README.md).

A small diagnostic fit only the 2×:5×:20× ratios on **main training histories**, then applied them by type while preserving the leading hybrid's 0× and 1× probabilities. It scored **56.3859**, up .0448. ΔR was −.000847, paired 95% interval [−.001265, −.000449]. This is a cheap structural challenger. The ratio stability and gain need checking on fresh episodes.

## Research and experiments

### First wave

1. **Improve the conditional win table, focusing on (4,7).** Preserve M3's pair and flip posterior for the first ablation. Fit all four ordered binary lag cells with partial pooling toward the existing rates. Test a shared four-cell table, then one with pair-specific deviations and optional machine-type pooling. A second ablation lets the refined table change the pair and flip likelihood. Measure p0 error and contest R for (4,7) in every condition, and check that the other pairs retain their scores. Research sparse higher-order categorical transitions and conditional tensor factorization for a follow-up that learns additional lag interactions from data. [Sarkar and Dunson, Higher Order Markov Chains](https://arxiv.org/abs/1506.06268).
2. **Fit a joint two-state win and payout model.** Compare a persistent per-machine mixture against a sticky two-state HMM with the same state-specific emissions. Each emission uses the lag cell to estimate win probability and a type-specific positive payout profile. The forecast averages the two states' *five-class distributions* using the filtered state probabilities. This preserves the fact that a state changes both win and payout odds. Start with type-specific tail ratios and a small state effect on win log odds. Compare frozen best, tail projection, payout-only state, and joint state using the same histories. Test whether temporal transitions earn a gain over the static mixture before adding explicit duration. [Johnson and Willsky, Hidden Semi-Markov Models](https://www.jmlr.org/papers/v14/johnson13a.html).
3. **Recover evidence from sparsely observed histories.** Test a small shrinkage or blending model using effective pair count, posterior entropy, and posterior flip age. Use shared M4 and the full blend as candidate sources. Separately, prototype a finite-state forward pass that sums over unobserved binary outcomes in historical gaps. Use the last eight binary outcomes as memory, with an observation mask and a small set of pair, direction and flip hypotheses. Compare drop-missing, posterior-weighted historical counts, and the forward pass on identical masks. Keep cutoff and targets fixed while reducing only observations. Check recovery on generated histories with known rules, then compare released short histories and full histories. Estimate only eligible forecast weights on episode-held-out predictions. [Yao et al., Stacking Predictive Distributions](https://arxiv.org/abs/1704.02030).

### Second wave

4. **Test rule changes and new lag support.** Compare no flip, one polarity flip, and a run-length filter that can reset polarity, strength, or pair identity. Change one mechanism at a time. Learn a fresh 28-pair prior with a uniform floor and compare it with the carried main prior. Include independent versus shared episode hazards. Because recent flips are rare under the fitted model on main, use generated late flips, pair swaps, and repeated flips with exact cutoff odds, then test on fresh adaptation. [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742).
5. **Train a small sequence residual model.** Keep the structured win and payout baseline as an offset. Compare a masked GRU and a causal dilated convolution using the same inputs, parameter budget, training episodes and seeds. Inputs include payout category, observation mask, elapsed time since the last observed spin, and prequential lag-pair evidence. Apply synthetic short-history masks that match released observation layouts. Test changes one at a time against the existing factored GRU. An equal-weight blend stays the control for a learned support-dependent blend. [Che et al., Recurrent Networks with Missing Values](https://www.nature.com/articles/s41598-018-24271-9) · [Bai, Kolter and Koltun, Convolutional and Recurrent Sequence Models](https://arxiv.org/abs/1803.01271).

## Evaluation before promotion

- Freeze the 56.3410 hybrid and the train-tail diagnostic as separate controls. Record candidate formulas, features and fold membership before looking at held results.
- Use whole-episode folds. Fit population statistics, calibrators and blend weights inside the fitting folds. Hold complete evaluation episodes out of the main comparison. If self-supervised evaluation histories are used, score that transductive setting separately. Signal is observed at the real cutoff and condition labels are used for splitting and scoring.
- Save query-keyed forecasts and report the official R, paired episode bootstrap, four condition ratios, p0 MSE, win-mass-weighted conditional payout error, and scores for the short-history and inferred (4,7) groups. Check support from each subgroup and use the overall score for selection. Use generated odds and masked histories to test mechanisms, with their scores reported separately.
- Estimate saturation with nested history masks on the same complete-history cutoffs. Retain the last 40 spins and vary older observations from 24 to all. Separately vary the number of training episodes. Compare how the structured and neural models respond to each budget to separate limited prefix evidence from limited training data.
- Reserve fresh adaptation episodes for a final comparison once the kit is released. Public main has already informed the designs. Check row-order and padding invariance and forecast-only runtime before promoting a model.

## Reproduce the analysis

From the kit root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -B signal_experiments/ceiling_analysis/analyze.py
```

[Machine-readable diagnostics](diagnostics.json) contain the score decomposition, model and input hashes, exact per-condition values, and paired comparisons. [Slice CSV](slices.csv) contains query counts, observed lag support, shares of the remaining contest loss, and raw mean squared probability error by model.
