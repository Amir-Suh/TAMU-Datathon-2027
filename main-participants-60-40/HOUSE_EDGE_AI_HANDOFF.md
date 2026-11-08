# House Edge forecasting challenge — project handoff for another AI

**Main-round evidence snapshot, November 7, 2026 (Central Time).** This document contains the problem, data contract, methods, measured outcomes and open questions. The receiving model has only this text. File names near the end are for the team to locate artifacts later.

## 1. Where the project stands

- **Frozen, simple reference candidate.** Amir's **M3**, a Bayesian two-lag model with a possible single polarity reversal, scored **55.268370 / 60** on all 96 released main validation episodes. A fresh fit matched the archived forecast to about 1.3e-14. There is a format-checked main test CSV.
- **Best complete saved hybrid.** **56.341022 / 60**. Use M3's win probability with a small adjustment that distinguishes two past losses from two past wins. For payout sizes given a win, use the conditional mix from an equal average of the 31-leaf lag-pair LightGBM and a three-seed factored GRU ensemble. Gain over M3 **+1.072652 points** and paired change in loss ratio **−0.020668 [−0.025110, −0.016367]**. All four conditions improved. Saved main test CSV passed the official format checker.
- **Highest observed score, structural diagnostic.** **56.3859 / 60** after keeping the hybrid's probabilities of 0× and 1× and allocating its remaining mass using type-specific 2×:5×:20× ratios estimated from train histories. Gain **+0.0448 points** over 56.3410. Treat the projection as a small development challenger.
- **Evidence boundary.** The hybrid, projection and branch scores reuse public main validation. The older Bayesian-vs-boosted comparison had its own locked half-validation audit, described below. Hidden main test scores, actual adaptation results and organizer upload/reproduction confirmation are unavailable in the reviewed record.

## 2. Objective, inputs and scoring

Predict the exact five-category probability distribution of **one next spin** of a simulated machine. Categories are gross returns `[0, 1, 2, 5, 20]` on a one-chip wager. The team's models call any payout greater than zero a “win,” including 1×, which returns the wager. Expected payout alone is insufficient for scoring.

### Released files and joins

| File | CSV columns | Where available and how it is used |
|---|---|---|
| `queries.csv` | `query_id,episode_id,machine_id,machine_type,signal,cutoff` | All splits. One query per machine. Type is `steady`, `balanced` or `jackpot`. Signal is the observed integer −1, 0 or +1. Predict at tick `cutoff`. |
| `history.csv` | `episode_id,machine_id,tick,payout` | All splits. One observed spin per key. Join self-history on episode and full machine ID. Join peers within the episode and align on actual ticks. |
| `labels.csv` | `query_id,outcome` | Train and validation. One sampled cutoff outcome per query. |
| `conditions.csv` | `query_id,condition` | Train and validation. Labels are constant within an episode. Used for grouped splits and scoring. |
| `targets.csv` | `query_id,episode_id,p_0,p_1,p_2,p_5,p_20,outcome` | Validation only. The five probabilities are simulator truth. The sampled outcome is also included. |
| `reference.csv` | `query_id,p_0,p_1,p_2,p_5,p_20` | Validation only. Frozen strength-20 pooled forecasts fitted for the round. |
| `sample_submission.csv` | Same header as reference | All splits. The 0.2 values are format examples. |

Every episode has **eight machines at a common cutoff**. All episode and full machine IDs are split-disjoint. A suffix such as `m000` just denotes a position in the episode. Train teaches population parameters. A test forecast uses the new machine's own released history to infer its local rule. Example train query `main-train-e00000-m000-next` is steady, signal +1, cutoff 238. Its own payouts at 235–237 were 1, 0, 1, and the label at 238 is 2. An earlier supervised example at tick 237 must truncate **every** machine's features at tick 236. Example test query `main-test-e00000-m000-next` is balanced, signal −1, cutoff 221, with 64 observations including ticks 181–220.

| Main split | Episodes | Queries | History rows | Full / short machine histories | Cutoff min / median / max |
|---|---:|---:|---:|---:|---|
| Train | 96 | 768 | 139,944 | 576 / 192 | 192 / 220.5 / 256 |
| Validation | 96 | 768 | 141,392 | 576 / 192 | 192 / 223.5 / 256 |
| Test | 192 | 1,536 | 282,664 | 1,152 / 384 | 192 / 226 / 256 |

- **Full** prefixes contain every tick from 0 through `cutoff-1`. **Short** prefixes have the last **40 consecutive** ticks and **24 sampled older** ticks. Every actual query has all last eight lags available. A missing row is unknown. Preserve its time index and observation mask. Counting missing as 0× or compressing away gaps breaks lag features.
- The four conditions are `stable`, `short_history`, `drift`, `shared_change`. Train and validation each have 24 episodes per condition. Test has 48 each by metadata, with individual labels private. Short-history layout is observable. A full history alone cannot distinguish the other three labels. Avoid target-, reference- or condition-derived routing at inference.
- Histories are synchronous. Every feature for a forecast at tick `t`, including peer features, uses ticks **strictly before** `t`. Historical spin labels support causal training via pretend cutoffs. Treat the query's signal as a cutoff observation. Its availability at older historical endpoints is an assumption unless established separately.
- Main kit metadata is `experimentation-suite-v3`, event mode, season ID `5fffdc4d0c700bbf61c3f911`. Main is worth 60 points. Adaptation is a separate kit with new disjoint train, validation and test episodes and changed recurring patterns, worth 40 points. The kit schedule says main closes and adaptation opens Nov 7 at 9 PM Central, with adaptation due Nov 8 at 11 AM Central. Main data can be reused in adaptation, and adaptation uses its own frozen reference.

### Class balance and training units

| Source | 0× | 1× | 2× | 5× | 20× |
|---|---:|---:|---:|---:|---:|
| Train cutoff outcomes, counts | 431 | 161 | 115 | 49 | 12 |
| Train observed history, counts | 81,642 | 26,980 | 20,524 | 9,007 | 1,791 |
| Train history, % | 58.34 | 19.28 | 14.67 | 6.44 | 1.28 |
| Steady history, % | 42.20 | 35.65 | 17.51 | 4.55 | 0.09 |
| Balanced history, % | 59.26 | 14.35 | 18.23 | 6.97 | 1.19 |
| Jackpot history, % | 77.56 | 4.73 | 6.66 | 8.16 | 2.89 |

Train has 266 steady, 278 balanced and 224 jackpot queries. Signal counts for −1/0/+1 are 250/267/251. Validation has more jackpot machines by share (34.5% vs. 29.2%). No steady machine had a 20× cutoff training outcome, although steady histories include 20×. Train at natural class proportions or calibrate explicitly if changing sampling. Reweighting by rarity or dollar value changes the probability objective.

The initial shared training set used the last 32 observed history labels per machine (24,576 rows, 6,144 per condition, all eight immediate lags available) plus 768 actual cutoffs. Historical rows are correlated within machines and episodes. If using all eligible histories at tick ≥40, short history supplies only 6,174 of 110,790 endpoints with all eight lags observed (5.6%). Earlier pseudo-cutoffs are on average 16.5 ticks earlier than real cutoffs. Their last-40 observation counts average 36.4 and can be as low as 9, versus 40 at every actual cutoff. Verify performance separately at actual cutoffs.

### Official score and comparison discipline

For each query, compute `sum_k (forecast[k] - true_probability[k])^2` over all five categories. Average those losses over machines within each episode, then over episodes within condition `c` to get `E[c]`. Compute `B[c]` by applying the same procedure to the frozen reference. Then:

```text
R = mean over the four conditions of E[c] / max(B[c], 1e-9)
main points = 60 / (1 + R)
adaptation points = 40 / (1 + R)
total = main points + adaptation points
```

Lower `R` is better. The frozen reference scores 30 main points or 20 adaptation points. Exact simulator odds would score 60 or 40. Sampled outcomes are useful for training, binary/multiclass Brier and log loss. Those realized-label metrics are on a different scale from the contest loss. In expectation, one-hot multiclass Brier equals squared error to the true distribution plus the sampling variance `1 - sum_k true_p[k]^2`.

We distinguish three evaluation protocols:

1. **Earlier tuning / frozen audit.** The kit sorts episodes by SHA-256 of ID within condition, alternating into two 48-episode halves, each with 384 queries. Earlier team models competed on tuning. The two-model frozen audit later scored Bayesian `switch_96` and boosted/logistic after saving forecasts and hashes. That audit is already consumed.
2. **Full main validation.** All 96 episodes and 768 queries. Used for the Amir branch comparison, repaired M5/M6, Fredy replays and later ablations. A retrospectively printed “audit half” for these full-validation experiments gives no fresh holdout. Some later calibration/state fits cross-fit by episode within these 96 episodes. Fredy's development fits also use pre-cutoff validation and test histories as self-supervision. Keep that transductive protocol distinct from strictly held-episode learning.
3. **Grouped training checks and stress tests.** Hold whole episodes together. Fit priors, features of upstream models, calibrators and weights inside the appropriate fitting folds. Separately report masked-main and known-mechanism synthetic tests. Paired uncertainty intervals quoted here resample whole episodes within each condition with fixed forecasts and matching reference losses. They describe released-episode variation, without adjusting for the model search or projecting hidden test performance.

## 3. What the released data taught us

### The dominant signal is a machine-specific binary lag interaction

Define `x[t] = 1` for a positive payout and `0` for 0×. Candidate lag pairs satisfy `1 ≤ a < b ≤ 8`, giving 28 pairs. Many machines have high win odds when `x[t-a] == x[t-b]` and others when they differ. A binary win match can involve two different positive payout sizes. The main inferred pairs are **(1,3), (2,5), (4,7)**. Keep the other pairs in the hypothesis space, especially for adaptation. Stable train machines had the same best pair in their first and second halves 93% of the time versus 5% after shuffling. Single-lag models add little. M2 dropped to about 29.9 when validation histories were shuffled, yet scored about 51.2 after fitting population priors to shuffled train and using intact validation prefixes. This locates much of the usable evidence in each machine's own history.

Train diagnostics found the same pair with reversed polarity on roughly 89% of drift and 94% of shared-change machines. A step reversal usually fit better than a gradual ramp. Typical inferred flips were around tick 140–150, with median about 70 ticks from flip to query. Amir's corrected analysis found **about 1.2 ticks of inferred within-episode flip-time spread** for shared change versus 17.7–18.8 under shuffled episodes, and 13.7 versus 20.4–21.6 for drift. This supersedes an early crude 23-tick estimate. Sharing flip time still helps little with the normal cutoff history. Pair identity, polarity and payout mix showed little evidence of being shared across an episode. Peer-alignment experiments also found little forecast improvement.

### Observable `signal` and a distinct inferred payout state

- Signal mainly calibrates **win odds within type and lag context**. In an exploratory analysis of 733 high-confidence validation rows, true win odds rose monotonically from −1 to 0 to +1 in all six type × favorable/unfavorable combinations. For jackpot machines in the favorable cell, mean win probabilities were about 51%, 56%, 65%. A one-step signal increase corresponded to about 1.45× the win odds after accounting for the other structure. The association appears even in stable-machine histories more than 100 ticks before cutoff. Whether signal is fixed or is a readout of a persistent hidden variable is unknown.
- In matched full-validation refits, M1 scored **31.077** with no signal and **31.092** with signal in both heads. M3 scored **55.126** with no signal, **55.272** with win-only signal, **55.121** with payout-only signal, **55.268** with both. Permutation checks found little evidence that signal identifies pair, polarity, current favorable cell or flip time.
- True main validation odds reveal **two levels of `P(1× | win)` per type**, with fixed relative proportions among 2×, 5×, 20× within type. Low/high values are **.455/.754 steady**, **.214/.500 balanced**, **.121/.335 jackpot**. The high-1× mode occurred on about 54% of queries at each signal value. In an exploratory adjusted analysis it was associated with about **0.47× the odds of a win** controlling for type, inferred lag state, condition and signal. These are inferences from released odds. The mode labels are unavailable as test inputs. The early conclusion that payout-size variation was largely irreducible was revised after the history decoder and neural payout swaps.

## 4. Earlier custom suite — 48 tuning episodes unless marked audit

The useful modeling split is `w = P(payout > 0)`, `r = P(size | win)`, with output `[1-w, w*r1, w*r2, w*r5, w*r20]`. This made it possible to swap payout models while fixing win forecasts.

| Attempt | Main tuning points / 60 | What it established |
|---|---:|---|
| Pooled reference / kit pooled recent | 30.000 / 31.06 | Starter uses local counts with a type × signal prior. Kit recent half-life 24, strength 20. |
| Kit single lag / pair (1,2) | 30.85 / 30.74 | Best tested kit single lag was 1 with half-life 24. Fixed generic lag choices miss machine-specific rules. |
| Swept pooled recent | 31.133 | Half-life grid {4,8,12,24,48}, strength {4,8,20}. Winner 12/8. No-context ablation also 31.133. |
| Direct five-class logistic | 31.354 | Causal summary/lag features cannot capture local rule variation in this model. |
| Two-stage logistic | 31.444 self-only, 31.306 with peers, 31.290 unblended | Weak win forecasts, but self-only conditional payout model helped stronger win heads. |
| Fixed / adaptive lag-expert mixture | 31.237 fixed, 33.225 initial adaptive | Binary singletons, ordered four-cell pairs and no-lag experts. Decay 12/48 ticks, strength 8. Prequential excess Brier relative to no-lag changes the mixture within each machine. |
| Adaptive rate sweep | 43.682 at 384, 44.104 at 1536, 44.098 at 3072 | Training grouped-cutoff Brier selected 3072. Tuning favored 1536 by only .006 points. |
| Signed motifs | 31.503 singleton, 42.026 singleton + pair, 42.402 with logistic payout | Logit offsets with signed effects for binary lag features, discounted evidence half-life 48. Pair model strong on stable and short (.132/.188 R), poor on drift and shared (.779/.612). |
| Online regime filter | 43.950 vs. matched 44.098 base | Constant hazard and fixed-state gates lost to base at rates 48 and 3072. Peer-triggered reset was gated out. |
| Lagged peers | 44.117 individual, 44.091 pooled, 44.098 self-only | Matched aligned 44.061 vs 11-tick-shifted 44.010. Differences did not establish a reliable incremental peer gain. |
| Boosted win-logit correction | 37.775 at rate 48, **49.744** at 3072 | 60 shallow trees, causal features on a strong adaptive offset, inner and outer episode exclusion for population priors. At 3072, intercept/calibration 46.035, 30 trees 49.219, zero correction 44.098. Boosting improved all conditions; tuning `p_0` MSE .0434 → .0206. |
| Self-only causal CNN | 31.597 | 40-tick window, 16 channels, width 3, dilations 1/2/4/8, four epochs. This model/training run was weak. Later full-sequence GRUs did substantially better. |
| Transition retrieval | 31.438 fixed, 41.233 learned | Retrieves prior transitions from a machine's history using lag/context similarity and recency. Learning the similarity helped but trailed the strongest models. |
| Combine boosted, adaptive, signed motifs and logistic | **50.273957** fixed, **50.237370** episode-cross-fitted blend | Best fixed forecast used 60-tree rate-3072 win and self-only logistic size. +0.530 points over boosted alone, `ΔR=-.012718` [−.019674, −.006553]. Scalar blending of boosted and signed win forecasts gave little additional benefit. |

The full adaptive rate grid gave the following tuning points by rate. 0→31.237, 12→31.381, 24→31.666, 48→33.225, 72→35.828, 96→38.177, 144→41.114, 192→42.371, 288→43.325, 384→43.682, 512→43.883, 768→44.027, 1024→44.079, 1536→44.104, 2048→44.101, 3072→44.098. The boost runner always exports a best nonzero correction and the regime runner a best nonbase filter. Use their reported comparison against the base when interpreting exported CSVs. The 60-tree correction won all four grouped training folds.

### Earlier switching Bayesian follow-up and genuine audit

- **`switch_96` + self-only logistic payout head** uses posterior weights over all 28 binary lag pairs, signed logit effects and a null/control. An effect amplitude can reset each tick with probability **1/96**. Pair identity is fixed within a hypothesis. Posterior weights can shift across pairs, though old evidence can delay that shift. The uniform-prior run reproduced saved tuning forecasts exactly. Nineteen targeted tests passed in the review.
- Grouped train cutoff **binary Brier** ranked `switch_96` .082812, `switch_32` .087086, discounted signed .142102, widened discounted .143056, static signed .187806, parity table .194872, four-cell table .199040. This favored switching amplitudes over discounting or static tables in that implementation. These static tables differ from the later split-cell M3 ablation.
- The standalone model scored **53.382758** on tuning, the scalar cross-fitted blend **53.392796**, and the fixed 25% Bayesian / 75% boosted-logistic blend **51.712276**. Learned pair-frequency prior with 80% learned / 20% uniform support scored **53.799644**, `ΔR=-.008709` [−.014173, −.004372]. Most gain was short history. It missed the predeclared −.01 advancement gate and was never evaluated on the locked audit. Nested training Brier improved only .082812 → .082452, with mixed folds. The implemented hazard/grid gain gate also stopped the reset-144 and wider-grid follow-up. A more elaborate positive-head stage remained a plan in this run. In a controlled prior stress test at tick 40, learned vs uniform prior MSE improved from .00686 to .00319 for planted (2,5), and worsened from .01009 to .01919 for planted (1,8). The forecasts converged by tick 128.
- In the locked audit, Bayesian scored **52.665123** versus **49.212506** for boosted/logistic. Audit `R=.139274` vs `.219202`, paired `ΔR=-.079928` [−.124380, −.041979]. Condition ratios Bayesian vs boosted were stable .093617 vs .076837, short .225938 vs .277504, drift .114058 vs .271122, shared .123483 vs .251346. Bayesian won 30/48 episodes. On the joined full validation, these fixed models score **53.037228** and **49.766798**.
- In the input-only audit, recent 32-tick observed win rate was .42924 train vs .39378 test. About .00974 of the .03546 gap was explained by type/signal composition. That is a history shift, with no direct evidence for cutoff recalibration. Frozen Bayesian pair posterior mass on the three common pairs was 94.93% train, 93.68% validation, 93.99% test (80.89% on short test histories, 98.35% on full). The posteriors are inferred, and these checks cannot establish hidden accuracy.

## 5. Amir's M1–M6 family — full main validation

| Model | Points / 60 | Stable R | Short R | Drift R | Shared R |
|---|---:|---:|---:|---:|---:|
| Kit pooled recent | 30.825854 | | | | |
| M1 hierarchical frequency | 31.091807 | .9617 | .9793 | .9333 | .8448 |
| M2 discounted parity lag pair | 52.597509 | .0817 | .1192 | .2113 | .1508 |
| **M3 explicit flip** | **55.268370** | .077244 | .106894 | .079817 | .078493 |
| M4 independent/shared flip mixture | 55.282990 | .077314 | .105820 | .079818 | .078347 |
| M5 original boosted corrector | 54.987969 | .078697 | .111637 | .088928 | .085329 |
| M6 original ensemble | 55.296582 | .076507 | .105740 | .079860 | .078125 |
| M5 repaired boosted corrector | 54.841763 | .083822 | .116690 | .088514 | .087201 |
| M6 repaired ensemble | 55.282043 | .077150 | .106681 | .079534 | .078008 |

**M1.** Type × signal context priors. Recency-weighted Beta posterior for the local win rate, chosen strength 5 and half-life 16 ticks. Conditional size head is Dirichlet-smoothed local positive counts, chosen strength 100 with no decay. The size head is shared with M2–M4.

**M2.** All 28 pairs × two directions × three count half-lives (∞, 64, 24) plus corresponding null hypotheses. For each hypothesis, estimate two Bernoulli cell rates with shrinkage to learned type × signal high and low rates. Score predictions prequentially with a 64-tick evidence half-life. Use unconditional local win odds when a required lag is missing. Two empirical-Bayes iterations learn a pair prior with a 10% uniform floor. Chosen rate-prior strength 3. This handles changing rules by discounting old evidence.

**M3.** Infer a fixed pair, current polarity and flip time jointly. Consider no flip and a four-tick grid of flip times with at least eight post-flip ticks, plus a 5% null hypothesis. For a flip hypothesis, swap high/low cell assignments before the flip and pool those counts with post-flip evidence. Observed outcomes with unavailable lag context form a third group. Missing outcomes add zero evidence. Compute each hypothesis's Beta-binomial marginal likelihood on the same observed outcomes, via prefix/suffix cumulative counts, and posterior-average the cutoff win probabilities. Use current observed lags when available. Empirical priors from two training iterations give about 33%, 29%, 28% to the main three pairs, with a 10% uniform floor. Flip-time prior is smoothed 50/50 with uniform. The fitted flip mass around .62 includes early flips nearly indistinguishable from no flip. Chosen Beta strength 120 on train pseudo-cutoffs (20 also scored about 55.25 on validation). Approximate prior means by signal −1/0/+1:

| Type | Favorable cell | Unfavorable cell |
|---|---|---|
| Steady | .93/.94/.96 | .12/.15/.21 |
| Balanced | .78/.84/.87 | .04/.05/.07 |
| Jackpot | .50/.59/.65 | .01/.02/.02 |

M3 gains **2.231 points** over the earlier switching Bayesian on the same 96 episodes, wins 79/96, and improves all four conditions. Paired `ΔR=-.045669` [−.061022, −.031343], with `p_0` MSE .010378 → .004711. It recovers most of M2's loss on changing rules. M3 assumes a single abrupt reversal and constant pair within each hypothesis. The internal M1–M4 pseudo-query selection reuses priors estimated from training histories containing those endpoints. The external validation result and deterministic replay support the model separately.

**M4.** Compare an independent-flip model with one whose eight machines share a single episode flip time. For the shared branch multiply per-machine likelihoods at each candidate time after integrating each machine's pair, polarity and null. Evidence-weight the two alternatives. Learned shared prior about .67. It adds .014620 points at ordinary cutoffs, mainly short history. Paired `ΔR=-.000287` with interval about [−.00057, −.00007]. Practical effect is small despite the evidence for synchronized flips.

**M5/M6 and repair.** M5 trains a histogram GBDT to replace M4's win probability, using M4's win logit, independent forecast and posterior/support features. M6 blends M2–M5 win probabilities with nonnegative, sum-to-one weights learned by binary log loss and keeps the M4 size distribution. The original M5/M6 computed `n_recent` using the batch's final 64 padded columns, so the window could shift away from a row's own cutoff. It undercounted 749/768 validation rows, sometimes by 64. Direct M3/M4 forecasts were unaffected. Original M5 folds also used features from a base fit that had seen the held episodes.

The repair uses per-machine cutoff indexing, episode-held-out M4 features at every M5 stage, and whole-pipeline cross-fitting of M5 and other members for M6 weights. It preserves tree-count candidates {25,50,100,200}. Repaired M6 weights were about **67.04% M3 / 21.67% M4 / 11.29% M5**, negligible M2. Its tiny gain over M3, `ΔR=-.000268` [−.000796, +.000292], did not hold up in nested training diagnostics. Repaired M5 lost in all four conditions, `ΔR=+.008445` [+.003679, +.013916]. On genuine training cutoffs under two 48-episode outer fits, binary Brier was **.076629 / .077028 / .079130 / .076982** for M3/M4/M5/M6. The respective binary log losses were **.266357 / .267504 / .277766 / .267517**. M6 switched from mostly M3 in one outer fold to essentially all M4 in the other. Saved repaired forecasts passed row permutation, padding and intact-episode batching checks. Keep episode groups intact when predicting with M4.

## 6. Fredy's branch, reproduction and model combinations

The early full-validation branch review found pooled/context/recent controls at **29.685–30.826**, best GLM `l8` **31.093**, best regime `d2_routed` **30.779**, original GBDT `trans_64_l15` with neighbor features dropped **36.395**, and first three no-message GRU-128 logged seeds **53.098–53.578**. The regime detector's logged 73.958% accuracy was below the 75% always-negative shared-change baseline. Factorized GLM, GRU with peer messaging, GRU-192 and a separate learned stacker had no retained benchmark establishing an advantage. The blend fitting code can optimize weights against public validation, but the original pipeline did not establish a separate held-out score for such fitted weights. Its default final configuration still selected the weaker GBDT.

### What changed with the pair features and GRU replay

- **Pair LightGBM.** Use type, own lags, causal frequency/EMA features and engineered features for **(1,3), (2,5), (4,7)**. For each pair include the current binary match, prior all-history and last-40 accuracies of the match rule, and corresponding rule predictions with direction chosen by historical accuracy. Require both lags and outcome when scoring historical rule accuracy. `pair_l31` uses 31 leaves, `tau=64`, drops direct neighbor features, and early-stops on held training episodes. It uses observed history ticks from train, validation and test in dev mode, with zero actual cutoff labels. Both 15- and 31-leaf pair models replayed to six decimals. Other logged pair variants scored **52.559** without transduction, **53.138** with 7 leaves, **53.931** at `tau=∞`, **53.987** at `tau=24`, and **53.859** with neighbors. These three explicit pair features will need reconsideration if adaptation changes the recurring pairs.
- **GRU.** Causal full tick sequence with a missing token, learned payout-token and machine-type embeddings, 128 hidden units, no peer message in the leading runs, 100 epochs. Plain GRU emits five-class probabilities. The factored version emits a win logit and four conditional-size logits. Both use observed prefixes from all three splits plus the genuine **train** cutoff outcomes in dev mode. Neither Fredy base has an explicit signal input. Short-history augmentation is a separate variant, logged **53.078–53.605** for two runs. The primary factored replay used three unaugmented CPU seeds 1–3. Original factored single-run logs were **54.063–54.272**, and a later original plain seed reached **54.508**.
- **Replay provenance.** No original CUDA checkpoints and only three of seven original seed IDs were available. The replay used seeds 1–7, four CPU fits and three fused Apple MPS fits for plain GRU. Device and implementation affect optimization. Plain replay single-seed scores ranged **53.035–54.558**. The historical 15-leaf tree + seven-GRU mean + original M6 was reported at **55.92** and reconstructed at **55.860278**. The historical tree + original M6 replay reached **55.389536** (reported 55.39). The 12 saved Fredy components (two trees, seven plain GRUs, three factored GRUs) reproduce saved forecasts on checkpoint reload. Fredy's dev fits exclude validation cutoff answers and exact validation odds. Final mode can include validation supervision, so dev scores describe the saved dev configurations.
- **Packaging gap.** The old GRU ZIP builder rejects a training-time Torch import even though frozen inference uses NumPy. A portable final entry needs separated inference code and a clean offline rerun. The saved component forecasts and reload checks support reproducibility on the development machine.

### Full main validation combination scores

Let `P` be a pair-tree five-class forecast, `G` the equal average of chosen GRU forecasts, and `A` a fixed M6 forecast. A full three-way average is `(P+G+A)/3`. A **payout-head swap** takes `w=1-win_source[0]` and uses a five-class source forecast `F` only for its positive shape, setting `out[0]=1-w` and `out[k]=w*F[k]/sum(F[positive])` for positive categories. For tree/GRU swaps, use `F=(P+G)/2`. Condition after averaging the two full vectors. Averaging their separately normalized conditional vectors is a different diagnostic.

| Forecast | Points / 60 |
|---|---:|
| Frozen M3 | 55.268370 |
| Historical 15-leaf pair tree | 53.945873 |
| Current 31-leaf pair tree | 54.289888 |
| Seven plain GRUs, mean | 55.419191 |
| Three factored GRUs, mean | 55.406231 |
| Historical 15-leaf tree + plain GRUs + original M6 reconstruction | 55.860278 |
| 31-leaf tree + plain GRUs + repaired M6, full average | 55.973765 |
| 31-leaf tree + factored GRUs + repaired M6, full average | 56.064085 |
| M3 win + conditional mix from the current plain-GRU three-way full average | 55.994410 |
| M3 win + 31-leaf tree payout mix | 55.751846 |
| M3 win + 31-leaf tree/plain GRU payout mix | 56.172553 |
| M3 win + 31-leaf tree/factored GRU payout mix | 56.259032 |
| Split-match M3 win + 31-leaf tree payout mix | 55.833268 |
| Split-match M3 win + 31-leaf tree/plain GRU payout mix | 56.254033 |
| **Split-match M3 win + 31-leaf tree/factored GRU payout mix** | **56.341022** |
| Train-tail ratio projection on the best hybrid, diagnostic | **56.3859** |

The single-family GRU means beat M3 by .138–.151 points, with paired intervals crossing zero. The payout swap improves more than the individual arms because they contribute complementary information. Ordinary M3 win with the tree/plain-GRU payout mix retains M3's `p_0` MSE .004711 while reducing win-mass-weighted conditional payout SSE .010198 → .006784. Switching to the factored instead of plain ensemble in the best hybrid improved `R` by .001647 [−.002706, −.000621]. That comparison also changes seed count and training backend.

## 7. Targeted follow-ups and their measured results

### Split the 00 and 11 lag contexts in M3

For every pair/flip hypothesis, compute the observed **post-flip** historical rate difference `win_rate(11) - win_rate(00)`. No-flip hypotheses use the whole prefix. Require both lags and observed outcomes. Learn a train-wide contrast for hypotheses where match is favorable and a second contrast where match is unfavorable, weighted by frozen hypothesis posterior and effective paired count `n00*n11/(n00+n11)`, with 200 effective counts at zero. The learned global differences were **−.02368 / −.00209**, respectively. Blend each machine/hypothesis contrast with 20 effective counts at its global contrast. Shrink further by `(n00+n11)/(n00+n11+8)`. Center the two predictions around M3's match probability under smoothed subtype frequencies and bound the contrast to valid probabilities. Keep the original pair, direction, flip and null posterior weights, pooled 01/10 prediction and positive payout mix.

This alone moved **55.268370 → 55.350776**, `R=.085612 → .083996`, `ΔR=-.001616` [−.003049, −.000364], and `p_0` MSE **.004711 → .004468**. All four conditions improved. There were 288 top-pair 00 rows, 178 top-pair 11 rows and 302 differing rows. Mean absolute win adjustments were .0063 for 00 and .0126 for 11. Causal count and fallback tests passed. The leading hybrid above uses this win component.

### History-only payout-state classifier

Use type and four history windows (16, 40, 96 and full) with shrunken 1× share among positives, win count and shrunken win rate. Standardize and use ridge logistic classification, `C=.1`, without signal or condition inputs. Learn the two state levels and tail ratios from only the other 72 validation episodes in each 24-episode fold. Mix the two state payout profiles using the predicted high-state probability, with M3's exact win mass fixed.

- State AUC **.731**, state Brier **.2085** versus **.2501** for the constant per-type mixture. Official points **55.278099** vs M3 **55.268370**, `ΔR=-.000191` [−.002194, +.001924]. The constant mixture scored **55.016327**, and a classifier weighted by M3 win mass squared scored **55.262742**.
- Unweighted conditional SSE fell **.02553 → .02361**, while win-mass-weighted error rose slightly **.01020 → .01026**. State classification skill alone does not establish a contest gain. This was a payout-only change and used validation-derived labels in grouped folds. The joint-state model described below is a separate proposal.

### Cutoff-only signal calibration

Keep the positive payout mix fixed. Fit a regularized logistic correction to the base win logit at genuine cutoffs using observed binary labels, equal episode weight and priors with standard deviations 1.0 on the intercept and .5 on the signal slope. Compare raw, intercept-only and intercept+signal. GBDT also had a sensitivity run using actual train cutoff labels, since the dev tree used only histories.

| Base / calibration data | Raw | Intercept | Intercept + signal |
|---|---:|---:|---:|
| Pair tree, validation grouped folds | 54.28989 | 54.09749 | 54.03749 |
| Pair tree, train cutoff labels | 54.28989 | 54.27644 | 54.22677 |
| Seven plain GRUs, validation grouped folds | 55.41919 | 55.23294 | 55.24239 |
| Plain GRU seed 2, validation grouped folds | 53.77811 | 53.63676 | 53.52257 |
| Three factored GRUs, validation grouped folds | 55.40623 | 55.22761 | 55.28020 |
| Signal-aware M3, train out-of-fold forecasts | 55.26837 | 55.12163 | 55.08835 |

True validation mean win probability rose 7.35 percentage points from signal −1 to +1. Raw GBDT, plain GRU mean and M3 showed 5.00, 3.68 and 6.89 points of gap. Positive fitted signal slopes did not improve any raw score here. This result and the matched M3 signal ablation answer different questions. M3 benefited from signal inside its win priors, while a small after-the-fact correction to existing predictions did not help.

### Prior transfer and shared flips under stress

On a main-round proxy, fit both arms to the same 24 main training episodes. Only the 28-pair prior differs. The carry prior used all main training including those 24. Carry scored **55.26080** and refit **55.28414**, `ΔR=-.000458` [−.001176, +.000042]. Their mass on the main three pairs was 90.5% and 90.2%. This is same-round evidence.

Keep actual main cutoffs/targets/reference fixed and mask histories deterministically. Preserve 24 older observations and the specified number of most recent contiguous ticks.

| Mask | M3 points | M4 points | ΔR (M4 − M3) |
|---|---:|---:|---:|
| Released history | 55.26837 | 55.28299 | −.000287 [−.000571, −.000072] |
| 24 older + 16 recent | 50.03294 | 50.08748 | −.001306 [−.002987, +.000312] |
| 24 older + 8 recent | 38.26344 | 38.45602 | −.007853 [−.013850, −.002133] |

In a separate seeded synthetic test, use disjoint 24-episode train and evaluation sets, known cutoff odds and a static match/mismatch rule restricted to the four rarest pairs under the main prior. Their combined prior mass was 1.5% carried versus 91.4% after refitting. **Raw five-class SSE** carried → refitted was **.000605 → .000605** at full prefixes, **.212595 → .056175** at 24+16 and **.284113 → .205446** at 24+8. This tests vulnerability to pair-prior shift when evidence is scarce. Actual adaptation results remain open.

## 8. Remaining error and research queue

### Where the best hybrid still misses

- **Per-condition `R`.** Hybrid **.0511 stable / .0929 short / .0553 drift / .0604 shared**. The full tree/factored-GRU/repaired-M6 average has a slight short-history edge (.0918), with paired difference about −.00116 [−.01428, +.01050]. A support-dependent blend remains untested.
- **Headroom.** Using exact released win odds with the hybrid payout mix gives **58.0446**. Using exact conditional size with its win odds gives **58.1384**. Splitting the win-size interaction equally attributes about **48.7%** of current contest loss to win and **51.3%** to size. These substitutions are diagnostics using true odds.
- **Short history.** 25% of queries produce **35.8%** of the remaining contest loss. Median historical support with the leading pair's two lag observations is **36** short vs **220** full. Every final query already observes its last 40 ticks, so the distinction is mainly older evidence. On short histories, replacing win with truth reduces R from .0929 to .0417, versus .0518 when only size is corrected.
- **Inferred pair.** (4,7) is **32.3% of queries and 45.3% of remaining R**, with the highest error of the three leading pairs in each condition and a similar pattern within types. The leading pair has posterior mass ≥.99 on **717/768** queries, which account for 90.5% of remaining R. Rows below .90 account for 6.4% of R. Improving identity selection alone has limited reach on this split. Only three queries have >.5 posterior mass on a flip in the last 40 ticks.
- **Payout classes.** 0×/1×/2×/5×/20× contribute **36.4% / 29.1% / 26.6% / 7.2% / .7%** of remaining R. The top error tenth accounts for 47.1% of R. Rows with true win odds ≥.5 account for 72.2% of R. This last grouping uses released answers for diagnosis. The hybrid lowered weighted conditional payout SSE **.01020 → .00645** and `p_0` MSE **.00471 → .00447** versus M3.
- **Latent payout mode.** Hybrid overestimates win probability by **2.7 percentage points** in the high-1× state and underestimates it by **3.2 points** in the low-1× state. The factored GRU conditional 1× score distinguishes these states with per-type AUC **.860–.888**. That is a stronger history signal than the logistic state decoder. The train-tail projection keeps `p_0,p_1` fixed and reallocates `1-p_0-p_1` across 2×/5×/20× using train-history frequencies within type. In [2×, 5×, 20×] order, those normalized weights are [.79047, .20532, .00421] for steady, [.69085, .26403, .04511] for balanced and [.37606, .46085, .16309] for jackpot. Its `ΔR=-.000847` [−.001265, −.000449] is a small score gain to verify on fresh episodes.

### Proposed next experiments, with no measured contest results yet

1. **Pair-specific four-cell win emissions.** Keep M3's hypothesis posterior frozen initially, and fit partial pooling for 00/01/10/11 rates around the parent parity rates, especially for (4,7). Compare shared, pair-specific and type-pooled rates. Test asymmetric 01 vs 10 separately. Let refined emission likelihoods affect pair/flip inference in a subsequent ablation. This extends the completed 00/11 forecast-only split.
2. **Joint two-state win and size model.** Compare a persistent per-machine mixture with a sticky two-state HMM using the same emissions. Conditional on state, let lag/type/signal determine win odds with state-specific offsets and let type/state determine size mix. Average each state's **full five-class vector** by filtered state mass. Multiplying separately averaged win and size heads would drop the correlation. Compare with the frozen hybrid, tail projection, completed payout-only state model and factored GRU signal. See explicit-duration models only if transitions help over a static mixture.
3. **Sparse-history inference.** First test a simple blend using observable support, pair posterior entropy and estimated flip age. Then compare dropping missing cells with marginalizing unobserved binary outcomes across historical gaps. Lags up to eight allow a forward state over the last eight wins/losses (256 binary states), conditioned on a small set of pair/polarity/flip hypotheses and observed masks. Validate on known rules first. Use nested masks with the same cutoffs, keeping the last 40 ticks while varying older observations. Vary the number of independent training episodes separately from prefix length.
4. **Adaptation rule changes.** Compare frozen main vs adaptation-only refit vs pooled or warm-start variants on the new kit. Relearn all 28 pair weights with uniform support and verify what patterns recur before using main-specific engineered features. Compare no flip, one flip and a filter permitting resets of polarity, strength or pair identity, with and without shared episode hazards. Deliberately generate late flips, multiple flips and pair swaps with known exact odds for mechanism checks.
5. **Small causal neural residual.** Compare a masked GRU and dilated convolution with matched data, seeds and capacity, using the structured forecast as an offset. Supply observation masks, elapsed time and causal pair evidence. Check against the existing factored GRUs, with synthetic short-history masks and equal-weight blend controls.

Earlier discussion also covered **TypeSafe Jev** as a possible standalone or payout-head benchmark and **LLM-generated validation**. There is no recorded Jev fit or score. Hosted inference cannot satisfy the offline submission contract. An LLM can help write a seeded simulator for controlled stress tests. Invented targets test the invented generator. Walk-forward predictions of real observed spins in held-out episodes add evaluation events, with dependence preserved by episode grouping. Fresh organizer episodes are the strongest confirmation for performance against true odds.

## 9. Delivery and evidence map

Organizer submission requires a CSV with exactly `query_id,p_0,p_1,p_2,p_5,p_20`, one row per test ID and finite [0,1] probabilities summing to 1 within `1e-6`. CSV limit 10 MB. Code ZIP limits are 100 MB compressed, 512 MiB unpacked and 5,000 entries, with `README.md` and `reproduce.json` at ZIP root, image `python` (Python 3.13), exact dependency versions in root `requirements.txt` and an offline command writing `/output/submission.csv`. Organizer mounts current data at `/data`, code at `/submission`, main data for adaptation at `/main_data`, and gives two CPU cores, 16 GiB RAM and 30 minutes. Predictions must reproduce within `1e-8` after normalization. Local format and checkpoint replays are separate from clean-ZIP offline reproduction and Form upload.

The following artifact paths are relative to the main participant kit.

- **Organizer contract.** `DATA.md`, `RULES.md`, `SUBMISSION_README.md`, `public/metadata.json`.
- **Early suite and audit.** `../ML_TRAINING_PLAN.md`, `../EXPERIMENT_RESULTS.md` (historical), `../runs/`, `adversarial_review/REVIEW.md`, `adversarial_review/verified_scores.json`. The audit in `../runs/bayesian-lag-pairs/audit-switch-96/` used separate prepare/freeze and score phases.
- **Branch review and repair.** `adversarial_review/branches/REVIEW.md` and `cross_branch_scores.json` in that directory, `m5_m6_repair/results/repaired-m5-m6/` for the comparison, frozen `models.joblib`, M3 `m3/submission.csv`, training folds and invariance checks. Amir's initial reviewed source was `c339b46`, with the corrected shared-flip analysis at `bb0357e`. Fredy's replay source is `bbff737`.
- **Latest replays and ablations.** `signal_experiments/README.md` and numbered experiment directories. Best saved test CSV is `signal_experiments/combined/m3_match_split_win_pair_l31_factor_gru_size_test.csv`. The corresponding validation CSV has the same name without `_test`. Verify with `python3 -B check_submission.py <csv>` (add `--split validation` for a validation forecast) or use `signal_experiments/score.py` for paired diagnostics.
- **Remaining-error analysis.** `signal_experiments/ceiling_analysis/README.md`, `diagnostics.json` and `slices.csv` in that directory.

**How to use this handoff.** If asked for ideas, distinguish replicated results, validation-only hypotheses and new proposals. Use the saved 56.341 hybrid and frozen M3 as controls, with the 56.3859 tail projection as a separate diagnostic comparator. Design tests around whole held-out episodes, legal inference inputs and the official probability loss. If proposing a model change, specify its prediction formula, how it uses missing history, the simplest matched ablation and which condition or error slice it is meant to improve.
