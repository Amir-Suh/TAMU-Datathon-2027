# House Edge model training plan

Main round, `experimentation-suite-v3` · November 7, 2026 · Central time

**Train a machine-adaptive lag model and a regularized win/payout model first.** Use recency-weighted pooling as the fallback and as the control for every ablation. It scored **31.13 / 60** after a small sweep on the public tuning episodes. The biggest opportunity is forecasting whether the next spin pays at all. On those episodes, the pooled reference's error fell by **92–96%** when given the true no-payout probability while retaining its own conditional mix of positive payouts. That diagnostic uses validation answers and serves only to prioritize modeling work.

This is the original training plan. See the [latest experiment results](EXPERIMENT_RESULTS.md) for the measured comparison.

## 1. What the query files give us

The kit contains CSV query manifests and a Python loader. No SQL query files were found in the checkout. Each row in `queries.csv` asks for the five probabilities `[p_0, p_1, p_2, p_5, p_20]` for one machine at its `cutoff` tick.

| File | Grain and join | Role |
| --- | --- | --- |
| `queries.csv` | One row per `query_id`, one query per machine in an episode | `machine_type` and `signal` are observable predictors. Episode and machine IDs group histories. Cutoff gives the prediction time. |
| `history.csv` | One observed payout per `(episode_id, machine_id, tick)` | Join on episode and machine for self-history, or episode for peer history. All input ticks must be strictly before the prediction tick. |
| `labels.csv` | One realized payout per query in train and validation | Train cutoff supervision. A historical payout can also label a prediction made at that earlier tick. |
| `conditions.csv` | One condition per query in train and validation, constant within each episode | Stratify folds, weight training samples by condition if useful, and score slices. The test manifest supplies no condition field. |
| `targets.csv` and `reference.csv` | One true and one frozen reference distribution per validation query | Measure actual probability error and score against the round's reference. |

The query `main-train-e00000-m000-next` is a steady machine with signal `1` at tick 238. Its own ticks 235–237 yielded 1×, 0×, and 1×, and its training label is 2× at tick 238. The features for this example can use other machines' ticks through 237. If tick 237 becomes a historical training label, every machine's feature cutoff moves back to 237.

### Verified inventory

| Split | Episodes | Queries | Observed spins | Full / short machine histories | Cutoff min / median / max |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 96 | 768 | 139,944 | 576 / 192 | 192 / 220.5 / 256 |
| Validation | 96 | 768 | 141,392 | 576 / 192 | 192 / 223.5 / 256 |
| Test | 192 | 1,536 | 282,664 | 1,152 / 384 | 192 / 226 / 256 |

Each episode has eight machines at a common cutoff, and episode and query IDs are disjoint across splits. The kit loader verified the observation keys and cutoff bounds. Every short history has 40 consecutive observations ending at `cutoff - 1` and 24 older observations. Preserve their original tick positions and masks. Training query counts by type are 266 steady, 278 balanced, and 224 jackpot. Counts by signal `(-1, 0, 1)` are `(250, 267, 251)`. Validation has a larger jackpot-machine share than train (34.5% vs. 29.2%), so track results within type alongside the official condition score.

### Imbalance and what it changes

| Observations | 0× | 1× | 2× | 5× | 20× |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train cutoff labels, counts | 431 | 161 | 115 | 49 | 12 |
| Train history, counts | 81,642 | 26,980 | 20,524 | 9,007 | 1,791 |
| Train history, % | 58.34 | 19.28 | 14.67 | 6.44 | 1.28 |
| Steady history, % | 42.20 | 35.65 | 17.51 | 4.55 | 0.09 |
| Balanced history, % | 59.26 | 14.35 | 18.23 | 6.97 | 1.19 |
| Jackpot history, % | 77.56 | 4.73 | 6.66 | 8.16 | 2.89 |
| Validation tuning true probabilities, mean % | 54.63 | 20.14 | 16.40 | 7.36 | 1.47 |

- **The cutoff labels are sparse.** Steady machines have zero 20× cutoff labels in train, but their histories contain 20× payouts. Use historical labels and smooth rare positive probabilities toward a type/signal prior. Use natural payout proportions in the loss. Inverse-frequency class weighting, oversampling jackpots, or multiplying error by payout size changes the probability target.
- **The conditions are balanced by episode and skewed by spin count.** Each training condition has 24 episodes and 192 queries. Of 110,790 historical labels at `t >= 40` with all eight immediate lags observed, only 6,174 (5.6%) are short-history examples. The last 32 observed ticks per machine offer a balanced starting set of **24,576 historical labels, 6,144 per condition**, with lags 1–8 observed for every label. Add the 768 real cutoff labels. This is 96 independent episode groups with many correlated observations per group.
- **Local pair counts can be thin even when the query lags are present.** For the starter's three-bin payout context `(0, 1, 2+)` at lags `(1, 2)`, short-history machines had a median of seven matching historical examples before decay. 59 of 192 had fewer than five. The medians for the other conditions were 30.5–38.5. Every pair model needs support-based backoff.
- **Win probability varies much more than its average suggests.** The 10th and 90th percentiles of true `p_0` were 3.95% and 98.87% on the 384 tuning queries. `p_0` contributed 67–69% of the pooled reference's squared error in each condition. The 20× component contributed 0.25–0.37%, though true 20× odds reached 11.87% on one tuning query. Track jackpot calibration by type while prioritizing the much larger win-rate errors.
- **Peer features need a controlled test.** In training, the mean absolute change in a machine's observed nonzero rate between the first and last 40 ticks was 11.8 percentage points in stable, 16.5 in drift, and 15.7 in shared change. This coarse comparison includes sampling noise. The signed changes were essentially uncorrelated between peer machines in shared-change episodes (pooled pair correlation −0.02). Try peer lag context and measures of change magnitude or timing alongside signed peer means.

Sources for the audit and file contract. [Data dictionary](main-participants-60-40/DATA.md), [main metadata](main-participants-60-40/public/metadata.json), [train queries and logs](main-participants-60-40/public/train/), [validation queries and logs](main-participants-60-40/public/validation/), [test queries and logs](main-participants-60-40/public/test/), [schema checks](slot_forecasting/schema.py).

## 2. Measured starting point

These scores use the kit's deterministic tuning half, 12 validation episodes per condition and 384 queries total. Lower condition ratios are better. The pooled predictions matched the frozen reference to floating-point precision. The decay sweep compared half-lives `{4, 8, 12, 24, 48}` and prior strengths `{4, 8, 20}`. Treat the winning setting as a provisional fallback since the same tuning episodes selected it.

| Model | Points / 60 | Mean ratio | Stable | Short | Drift | Shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen pooled | 30.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Kit pooled recent, half-life 24, strength 20 | 31.06 | 0.932 | 0.947 | 0.981 | 0.916 | 0.883 |
| Best fixed-lag kit variant, lag 1, half-life 24 | 30.85 | 0.945 | 1.002 | 0.964 | 0.939 | 0.876 |
| Kit pair `(1, 2)`, half-life 24 | 30.74 | 0.952 | 0.963 | 0.998 | 0.944 | 0.903 |
| Swept pooled recent, half-life 12, strength 8 | **31.13** | **0.927** | 0.952 | 0.964 | 0.917 | 0.876 |

The transition starter backs its conditional counts with a long-memory pooled forecast. To isolate the value of lag context, give the next transition model and its no-lag control the **same** recency prior and decay. [Frequency starter](slot_forecasting/baseline.py), [transition starter](slot_forecasting/transition.py), [episode selection](slot_forecasting/experiments.py).

## 3. Build the training set once

1. **Use one point-in-time feature builder.** Build examples as `(episode, machine, t)`, with `t = cutoff` for real queries and `t = observed tick` for historical labels. Index histories as tick-addressed arrays or maps with a separate observed mask. Compute windows in ticks, lag lookups at exact `t - lag`, and support counts from observed payouts. Treat cutoff as a time index and test any use of absolute tick as a predictor separately.
2. **Start with the balanced endpoint sample.** Use the 32 observed historical labels closest to cutoff and the real cutoff label per machine. Build all rolling and local-fit features again as of each example's tick. Historical short-history prefixes typically have fewer than 40 contiguous recent inputs, so retain prefix support features and compare to actual short-history cutoffs. If this mismatch hurts, try training-only augmentation of full prefixes with 40 recent and 24 sampled older ticks where sufficient history exists. Choose masks independently of outcomes. Group all augmented views with their original episode.
3. **Add feature families in order.** Start with type, signal, self and peer payout rates over the previous 8, 16, 32, and 64 ticks and the whole prefix, sample counts, and fast-minus-slow differences. Keep self and peers in separate aggregates, exclude the focal machine from peer summaries, and include peer support. Next add exact self lags 1–8, three-bin and win/loss contexts, then local conditional forecasts and their support. Compare self-only with peer features, including type-aware peer rates. As a diagnostic, compare aligned peers with peers shifted to older ticks.
4. **Check point-in-time behavior and split provenance.** A synthetic edit to any payout at or after `t`, including a peer payout, must leave the features and forecast at `t` unchanged. Removing a row must change its observed mask without creating a 0×. Fit global preprocessing and models inside each training fold. When a fitted prior or prediction feeds a second-stage model, generate those training features out of fold by episode so the prior or prediction cannot use that episode's later spins or the row's own label. Keep `query_id`, `episode_id`, and `machine_id` as keys. Exclude ID strings, suffixes, condition labels, targets, and reference predictions from model inputs.

## 4. Models to train, in order

| Priority | Model | First experiment | Decision |
| --- | --- | --- | --- |
| 0 | **Recency-weighted pooling** | Freeze half-life 12 and prior strength 8 as the main fallback candidate. Keep the original pooled reference as a score and serialization check. | Produce a reproducible CPU fallback before experimenting. |
| 1 | **Per-machine lag mixture with shrinkage** | Fit count experts for no lag, singles 1–8, and 28 pairs of distinct lags. Start with binary lag contexts for the win head, then try three-bin contexts. Use causal prequential errors to weight experts within each machine. | Keep only if local weighting beats both the same-decay no-lag control and a fixed global mixture. Inspect low-support short-history queries. |
| 2 | **Regularized logistic models, then shallow CatBoost** | Fit a binary win model and a four-class conditional payout model on the shared features. Use L2 logistic regression first, then compare CatBoost depth 4 and 6 with 300–600 trees and two CPU threads. Run a direct five-class model with the same inputs as a control. | Add blocks one at a time. Retain interactions and peer features when they improve true-probability score over the simpler version. |
| 3 | **Small causal temporal CNN** | If the lag or peer ablations help, train a masked CNN with a 40-tick recent window, 16–32 channels, kernel width 3, and causal dilations 1, 2, 4, 8. Provide long-history summaries on the side. Encode each machine with shared weights, then pool only peers at aligned pre-cutoff ticks. | Run self-only first. Require a gain over the tabular model or a gain from blending. |
| 4 | **Online change detection** | If fast/slow features mainly help on drift or shared-change episodes, compare a causal change score and short/long memory gating. Start with per-machine differences and pooled magnitudes of peer changes. Try Bayesian run-length tracking only if those features help. | Keep the gate only if it improves drift/shared change with acceptable stable and short-history performance. |

**How to fit the lag mixture.** At historical tick `s`, forecast from counts at ticks strictly earlier than `s` before using its realized outcome to update the counts. Compare all experts over a common recent scoring window and use the no-lag forecast whenever an expert lacks its input lags. Shrink pair estimates toward their singleton and no-lag estimates, with strength increasing as effective support falls. Test a short and a long decay (e.g. 12 and 48 ticks). Exponentially weight each expert using its recent excess Brier loss over the no-lag forecast, shrink expert weights toward the common mix when there is little evidence, and use normalized weights. Keep priors for positive size separately by type/signal and adapt them only with sufficient positive observations. This borrows the idea of weighting contexts of different complexity from context-tree methods. The proposed nonconsecutive lag mixture is a separate implementation. [Bayesian Context Trees](https://arxiv.org/abs/2007.14900).

**How to fit the two heads.** Predict `w = P(payout > 0)` and a softmax vector `r = P(payout = k | payout > 0)` over `k = [1, 2, 5, 20]`. Emit `[1 - w, w*r_1, w*r_2, w*r_5, w*r_20]`. Use binary log loss on all observed labels and multiclass log loss on positives. Fit a type/signal prior on training folds, then compare shrinkage of the positive head toward that prior against the learned head alone. Start with no class reweighting. Compare a win-logit offset from the local lag model as an ablation so the global learner can correct a machine-specific forecast. Include peer feature counts and category masks. CatBoost can learn nonlinear interactions on the resulting feature table. Episode separation still belongs in the evaluation design. [CatBoost paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [multiclass objectives](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification).

**How to fit the sequence model.** Use one channel per payout class plus an observed mask, type and signal embeddings, and summaries of older history. Encode each peer with the same network and mean-pool or pool by type, carrying the number of contributing peers. This handles arbitrary ordering of peer machines. If predicting multiple historical labels in a batch, shift inputs by one tick so the output at `t` sees at most `t - 1`. Use the same two heads and episode folds as the tabular baseline. Keep hidden width, depth, and training seeds fixed during the first feature comparison. [Causal temporal convolutions](https://arxiv.org/abs/1803.01271), [Deep Sets](https://arxiv.org/abs/1703.06114), [Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742).

## 5. Evaluate probabilities and freeze the decision

1. **Develop inside train first.** Use four fixed episode folds stratified by the training condition (six episodes per condition per fold). Compare feature variants using held-out historical and cutoff outcomes. Report episode-aggregated one-hot Brier and log loss, plus performance on real cutoff queries separately. Use equal episode contribution if adding unequal numbers of history examples. These are noisy development proxies. For an outcome drawn from true odds `q`, expected one-hot Brier equals `sum((p - q)^2) + 1 - sum(q^2)`, so it targets the same probabilities in expectation. [Gneiting and Raftery on proper scoring](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
2. **Rank a short list on the 48 public tuning episodes.** Use the kit's official scorer. It sums squared error across the five probabilities, averages over machines within each episode and then across episodes within each condition, divides by that condition's frozen reference error, and averages the four ratios to get `R`. Round points are `60/(1+R)` for main and `40/(1+R)` for adaptation. Report each condition ratio, each class's error, win calibration and 20× calibration by type, plus runtime. The scoring targets are the true distributions. The realized jackpots serve as labels and development diagnostics.
3. **Keep post-processing small.** For the best two complementary forecasts, try convex weights of 0, .25, .5, .75, and 1 on tuning. Compare with a single win-logit temperature or a prior blend if win calibration calls for it. If learning calibration or more flexible weights from tuning soft targets, fit them on whole-episode folds and compare out-of-fold tuning predictions before freezing. Skip flexible calibrators for rare payouts on 48 episodes. [Guo et al. on temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).
4. **Lock candidate, settings, and fallback before opening the audit.** Score the candidate and fixed fallback on the remaining 48 validation episodes once. Bootstrap paired differences in `R` by resampling whole episodes within each condition, using the same sampled episodes for both forecasts and their frozen reference. A practical predeclared promotion rule is to reduce audit `R` by at least 0.01 while keeping every condition ratio within 0.05 of the fallback or better. Record the interval and flag uncertainty if it spans zero. If the model misses the gate, use the fallback. The audit half was reserved from score comparisons and target diagnostics in this analysis.

[Round scorer](slot_forecasting/rounds.py), [condition aggregation](slot_forecasting/contest.py), [experiment runner](slot_forecasting/experiments.py), [official rules](main-participants-60-40/RULES.md).

## 6. Main and adaptation execution

- **Main, before Saturday 9 PM Central.** Save a valid baseline first. Implement the shared builder, then run the adaptive lag and two-head tabular models before stretching to a CNN or change detector. Budget the last hour for the audit, fresh-directory reproduction, ZIP creation, and upload. Record the candidate configuration, fold manifest, data version, seeds, dependency versions, validation results and hashes.
- **Adaptation, Saturday 9 PM through Sunday 11 AM Central.** Use the new public kit in a separate directory and verify its round metadata. Re-run the audit of train class rates, mask support, and which lag experts help. Compare frozen main weights, adaptation-only refit, and main initialization or pooled training with adaptation upweighted. Keep decay and lag mixture tunable. Refit the type/signal priors on adaptation data as the default and use its own frozen reference. Make the transfer decision on its tuning episodes and reserve the audit. Generate an adaptation-only fallback immediately.
- **Package and rerun.** Check `query_id,p_0,p_1,p_2,p_5,p_20` with the kit checker. The main test CSV needs 1,536 unique query IDs with finite probabilities summing to one. Package the prediction command and any saved checkpoint, pin dependencies for the organizer's Python 3.13 environment, and test the offline command against a fresh output path. The documented runner has two CPU cores, 16 GiB RAM, 30 minutes, and a normalized probability tolerance of `1e-8`. Keep the exact CSV and ZIP together after the Form confirms receipt. [Participant guide](main-participants-60-40/README.md#7-submitting), [submission template](main-participants-60-40/SUBMISSION_README.md).

### Commands for the kit controls

```sh
cd main-participants-60-40
python3 run_starter.py --out outputs-pooled-recent --method pooled_recent
python3 check_submission.py outputs-pooled-recent/submission.csv
python3 build_entry.py --starter outputs-pooled-recent --out entry-pooled-recent.zip
```

These commands reproduce and package the **kit default** pooled-recent settings (24, 20). The swept candidate (12, 8) needs an explicit `FrequencyForecaster("pooled_recent", half_life=12, prior_strength=8)` configuration and matching prediction command. The starter prints full-validation results. If preserving the audit, use `episode_folds()` and `subset_score()` to report only tuning results until the settings are locked.
