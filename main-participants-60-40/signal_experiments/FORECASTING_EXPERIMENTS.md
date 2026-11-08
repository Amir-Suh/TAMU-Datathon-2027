# House Edge forecasting research and experiment plan

November 7, 2026 · Main evidence and adaptation protocol

## Starting point

The job is to forecast five next-spin probabilities from an observed prefix, one machine type, and one signal recorded at the actual cutoff. Main validation has 96 episodes and 768 queries. The strongest saved model uses split-match M3 for win mass and the 31-leaf tree plus three factored GRUs for the conditional payout mix. Its score is **56.341022/60**, with **R = 0.0649434**. The train-only tail projection is a development challenger at **56.3859**. The simpler split-match M3 plus tree payout mix is a fallback at **55.8333**. [Saved model comparisons](README.md) · [Tail projection and residual diagnostics](ceiling_analysis/README.md).

| Evidence | Implication for the experiment queue |
|---|---|
| True win odds substituted into the best forecast give 58.045; true conditional payout odds give 58.138. | Give both components a research track. The substitutions use released targets and measure each component's separate headroom. |
| The inferred (4,7) lag pair accounts for 45.3% of remaining R on 32.3% of queries. High-confidence pair assignments are common. | Improve win rates within a rule before making pair selection more complex on main. |
| Short histories account for 35.8% of remaining R on 25% of queries. The last 40 ticks are observed; usable historical lag-pair counts have medians of 36 versus 220. | Test the use of older partial evidence and a blend based on observable support. |
| The high-1× state has a +2.7 percentage-point mean win bias; the low state has a −3.2-point bias. The factored GRU's conditional 1× estimates distinguish the two states with .86–.89 AUC by type. | Try the GRU's payout-state estimate as a win correction before implementing an HMM. Score joint forecasts, not state classification alone. |
| 0×, 1× and 2× account for 92% of remaining R. A train-history tail projection adds .045 points. | Preserve the useful payout structure. Put early compute into win mass and the 1× versus larger-win split. |

The current evidence already covers static frequency and logistic baselines, recency weighting, lag-pair inference, explicit flips, shared flips, boosting, plain and factored GRUs, a generic CNN, peer-lag corrections, and signal calibration. A prior amplitude-reset model keeps the lag pair fixed within each hypothesis. The open changepoint experiment is therefore a change in **what can switch**. The new four-cell experiment extends the existing 00/11 split to the ordered differ cells and pair-specific rates. [Original model review](../adversarial_review/REVIEW.md) · [00/11 experiment](04_four_lag_cells/README.md) · [Signal calibration](03_signal_calibration/README.md).

## Run order and gates

| Order | IDs | Deliverable | Advance when |
|---|---|---|---|
| 0 | B0, P0 | Frozen controls, exact score replay, train-only tail structure check | The run reproduces all query IDs, probabilities and published scores. |
| 1 | W1, Z1, S1 | Four-cell win ablation, GRU-state win offset, support-based blend | At least one variant beats its matched control on grouped development data, with evidence in its target slice. |
| 2 | Z2, S2, W2 | Joint static/HMM payout model, exact missing-spin filter, refined pair evidence | The mechanism works on known-law simulations and improves the official score on a separate held-episode comparison. |
| 3 | C1, C2, N1, W3 | Switching pair model, shared hazard, neural residual, sparse categorical alternatives | Each variant beats a matched simpler control in a regime the simpler model struggles with. |
| 4 | X1 | One bounded pretrained comparator | The mapping to categorical probabilities is valid and a frozen one-seed run is competitive enough to justify further work. |

Start **W1 and Z1** as independent jobs after B0. Run S1 with precomputed, eligible forecasts while those fits run. W2 waits for W1, S2 waits for a verified synthetic generator, and N1 uses the better structured control after the first wave. A first pass uses one seed for deterministic models, three paired seeds for the two neural architectures, and at most the listed configurations per ablation. Measure runtime on a 12-episode smoke subset before expanding any dynamic-programming or neural grid. Current in-repo components and frozen checkpoints make the first wave feasible without retraining a seed farm.

## Data and evaluation contract

1. **Freeze the inputs.** Record the round and season ID, branch and commit, source and data hashes, dependency versions, folds, seed, fit configuration, and parent forecasts. Keep the five-class ordering `0,1,2,5,20`. Reuse the published main folds and the comparison wrapper with the round's frozen reference. [Episode folds](folds.csv) · [Scorer](score.py).
2. **Keep episodes together.** Fit rate priors, cross-fitted features, learned states, calibrators, and stack weights inside the fitting split. When an outer fold is held out, refit every learned layer upstream of its predictions or use a checkpoint that excluded the outer episodes. Use grouped inner folds for model selection. When predicting earlier historical endpoints, construct each prefix separately and check causality with a future-outcome perturbation. The actual cutoff signal is available for actual cutoff rows. [Bürkner, Gabry and Vehtari, leave-future-out validation](https://arxiv.org/abs/1902.06281).
3. **Separate evaluation tiers.** Known-law simulations test the inference algorithm. Public main validation measures development performance on previously examined episodes. For adaptation, assign a condition-balanced lockbox of one quarter of validation episodes by stable episode hash before viewing its soft probabilities. Fit on adaptation train histories and realized cutoff labels. If a method needs soft labels for calibration or stacking, use only the development portion of adaptation validation for that fit. Open the lockbox for a frozen shortlist of at most two challengers and their controls. Report its episode count and condition balance. Fit a final model on additional public data only after retaining the lockbox result for the evaluated fit.
4. **Declare history exposure.** Default to a strict whole-episode holdout. Report a separate transductive comparison when a model self-supervises on evaluation histories. Use only pre-cutoff history for all forecasts. Condition, inferred truth-defined payout mode, and simulator odds are reserved for fitting where explicitly allowed or for scoring. Deployment routing uses observed counts, inferred posteriors, or other causal features.
5. **Use the proper target.** Score five-class squared error against released probabilities, averaged within episode and normalized by the reference per condition. Report overall `R`, points on the proper round scale, condition ratios, 10,000 paired condition-stratified episode bootstrap draws, p0 MSE, win-mass-weighted payout SSE, and the (4,7) and short-history slices. Historical Bernoulli and categorical log loss are useful fitting diagnostics. Preserve the official metric for selection. [Gneiting and Raftery, proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
6. **Set a practical promotion threshold.** Proposed default for added complexity is a fresh-lockbox improvement of at least **0.002 in R**, with a paired 95% interval below zero and no condition R deterioration above **0.005** unless that trade-off was specified in advance. Treat a wide interval as unresolved and retain the simpler incumbent. Run the fixed-formula replay and a two-core CPU timing before promotion. The organizer forecast-only budget is **30 minutes, two CPU cores, 16 GiB RAM**, with a code ZIP cap of **100 MB** and replay probability tolerance of **1e−8** after normalization. [Kit submission contract](../SUBMISSION_README.md) · [Rules](../RULES.md).

A cutoff-only result calibrated on reused main soft labels stays in the development tier. Whole-episode cross-fitting there protects the prediction of each held fold from its labels. The earlier model and hypothesis selection still informs the analysis.

## Experiment cards

### B0 · Reproduce the benchmarks

- **Controls.** Frozen M3, shared-flip M4, repaired M6, 31-leaf GBDT, factored GRU mean, full three-way blend, 56.341 hybrid, and split-match plus tree. Preserve each forecast's original fit scope. Record the 56.3859 tail projection as a separate challenger once recreated.
- **Deliverables.** A hash-checked validation replay, reference denominators by condition, forecast IDs, model settings and inference timing. Assert that the original M3 forecast scores 55.268370 and the hybrid 56.341022 on main. Keep adaptation points on their native /40 scale. [Replay code](01_fredy_replay/README.md) · [Scorer](score.py).

### P0 · Constrain the positive payout tail

- **Question.** Can a type-specific tail remove predictable variance while preserving the model's two strongest outputs?
- **Arms.** Compare the existing payout head with a projection that keeps p0 and p1 and allocates the remaining mass according to `N(2×), N(5×), N(20×)` from training histories within each type. Compare raw counts with one fixed lightly smoothed variant. Fit the ratio inside each training fold. Check stability across machine type, condition, and round using released training observations and development targets where available.
- **Readout.** Four condition ratios, 2×/5×/20× squared errors, and exact invariance of p0 and p1 within floating-point tolerance. If the tail ratios vary meaningfully on adaptation, compare a type-pooled shrinkage prior against the constrained head before carrying it over. [Existing projection probe](ceiling_analysis/diagnostics.json).

### W1 · Four ordered lag cells with pair-specific shrinkage

- **Question.** Do 01 and 10, and the two matching cells, need different rates conditional on the pair and its current polarity?
- **Arms.** `W1a` extends the existing 00/11 refinement with a pooled 01/10 contrast. `W1b` adds pair-level deviations for all four cells, shrunk toward the pooled contrast. `W1c` adds machine-type shrinkage only if W1b improves. Keep each pair, direction and flip posterior fixed. Use post-flip counts for local contrasts, retain the pre-flip counts in the parent, and keep conditional payout mix fixed. Start with effective local sample strengths **8, 20, 50** and select a single strength on inner episode folds. Collapse the correction to the parent when child-cell support is absent.
- **Readout.** Overall and (4,7) p0 MSE and R, by condition. Log cell sample counts and posterior mass for corrections. Check that zero contrast reproduces the parent exactly, and that cutoff/future payouts never enter child counts. [Current refinement implementation](04_four_lag_cells/run.py) · [Sarkar and Dunson, higher-order categorical transitions](https://arxiv.org/abs/1506.06268).

### W2 · Update rule inference with the four-cell likelihood

- **Question.** Does improved emission likelihood identify the right pair or flip sooner?
- **Arms.** Keep the winning W1 table and compare frozen-posterior prediction against allowing the new cells to change hypothesis evidence. Keep the same 28-pair support, null component, cutoff and flip grid. Validate likelihood normalization on a tiny history by explicit enumeration. Leave strength and pair priors fixed in the primary contrast.
- **Readout.** Paired score, change in posterior entropy, true pair recovery on generated sequences, and real short-history performance. Compare to W1 rather than a generic four-cell table. An earlier stationary four-cell implementation was weak on grouped training, which makes the matched changepoint comparison important. [Earlier Bayesian comparison](../adversarial_review/REVIEW.md).

### Z1 · Reuse the GRU's payout state to calibrate win odds

- **Question.** Can a state score inferred from history remove the opposite win biases across payout modes?
- **Arms.** Use the GRU's predicted conditional 1× probability, normalized or standardized by machine type from fitting episodes, as one state feature. Fit a regularized win-logit offset with an intercept and one state term using precomputed eligible forecasts. Compare raw, intercept-only, and intercept-plus-state with the conditional mix fixed. An optional fourth arm adds a single state × predicted-win-regime interaction. Apply the same regularization to the intercept across arms. Use actual cutoff labels, or released development odds only inside the fitting fold, and record the choice.
- **Readout.** Official R and p0 MSE first, then residual win bias by released payout mode and calibration by predicted win bin. Include the previous cutoff signal calibration as a negative comparison. If state AUC remains high and the offset fails, test whether the state feature measures payout behavior without enough independent information to improve win odds. [Payout-state baseline](02_payout_state/README.md) · [Existing calibration runner](03_signal_calibration/calibrate.py).

### Z2 · Joint payout and win state

- **Question.** Is a persistent machine trait sufficient, or do the observations support transitions within the history?
- **Emission.** For state `z`, let `w_z` be win mass after the lag-cell effect, `s_z` be the share of wins that pay 1×, and `r_type` the three-way tail. The three probabilities used in filtering are `1 − w_z`, `w_z s_z`, and `w_z (1 − s_z)`. Split the last category by `r_type` when returning five classes. Average the **joint five-class forecasts** over predicted state probabilities. This handles the association between state and winning.
- **Arms.** `Z2a` uses a constant latent state per machine. `Z2b` uses two states with symmetric expected dwell times **16, 64, or 128 ticks**, selected in inner folds. Include the payout-only emission as a control with its win-state effect set to zero. Start with a shared win-state coefficient shrunk toward zero and ordered 1× emission rates to identify the states across fits. Initialize from P0 and the eligible state proxy, then fit with regularized likelihood on training histories. The cutoff prediction filters the prefix. Smoothing is useful only inside training on fitting histories. Carry the transition through missing ticks. Use a marginal win rate for unavailable lag contexts in the first prototype, and integrate missing lag bits in S2.
- **Readout.** Check label-order invariance, normalization and agreement with an explicit two-state path sum on a tiny prefix. Compare static versus sticky on both full and sparse histories, and compare joint against the payout-only arm. Test explicit durations only if the sticky model wins and its inferred dwell behavior supports that extra structure. [Johnson and Willsky, explicit-duration HMMs](https://www.jmlr.org/papers/v14/johnson13a.html) · [Dynamax forward and backward recursions](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference.py).

### S1 · Blend forecasts using observed support

- **Question.** Can a simple support rule use the small short-history edge of the full blend without giving up the hybrid's advantage on long histories?
- **Arms.** For two frozen parent forecasts, compare equal weighting, one learned global weight, and two weights shrunk toward the global value. Set the first support boundary at **40 usable lag triples** before scoring. Keep posterior entropy as a separate one-feature alternative. Compare separate win and size blends only after the full-distribution comparison. Fit convex weights on eligible episode-held-out predictions with the same quadratic objective as scoring. Use the condition reference scales from the fitting fold.
- **Readout.** Overall and short-history paired changes, weight stability across folds, and fallback when support is missing or outside the fitting range. Each candidate uses the same frozen parents. [Yao et al., stacking predictive distributions](https://arxiv.org/abs/1704.02030) · [Stan loo weight-optimization reference](https://github.com/stan-dev/loo/blob/master/R/loo_model_weights.R).

### S2 · Sum over missing historical wins

- **Question.** How much pair evidence is lost when a historical spin is observed but one of its lagged inputs is missing?
- **Arms.** Compare drop-incomplete counts, expected historical counts under a fixed baseline, and an exact finite-state filter for a fixed pair and polarity. For largest lag `b`, the filter stores a distribution over the last `b` binary wins, with at most `2^b ≤ 256` states. Each state has two possible shift-register transitions. An observed payout restricts the next bit and contributes its emission likelihood. A missing tick sums both bits and advances time. Start with fixed emissions and no flip, then add a small set of pair, polarity and flip hypotheses. Keep a null or broad-prior alternative so short-history pruning retains uncertainty. Combine with Z2 only after the simpler filter works.
- **Readout.** Match brute-force enumeration for `b≤4` and short prefixes, match the complete-data likelihood with all ticks observed, and verify that consecutive missing ticks evolve time. Compare the same cutoffs under nested masks with **0, 12, 24, 48 and all** older observations plus the last 40 spins. Measure pair recovery and official score using the same held episodes across masks. Use sparse shift transitions and benchmark before adding more state dimensions. [Dynamax filtering reference](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference.py) · [Doucet et al., marginalizing tractable components in particle filters](https://arxiv.org/abs/1301.3853).

### C1 · Identify what changes in a regime

- **Question.** Does adaptation need a polarity flip, a gradual strength change, or a different lag pair?
- **Arms.** Compare M3's one-flip model, the existing amplitude-reset control, and online run-length inference that resets one component at a time. Start with mean run lengths **32 and 96** plus a no-change arm. Fit 28-pair priors using training data with a uniform floor, and compare refitted versus carried priors. Generate independent test episodes with a step flip, a 16-tick ramp, an actual pair swap, and two flips. Place changes 8, 16 or 40 ticks before cutoff. Preserve exact probabilities and retain a stable control. Keep multi-change and particle approximations behind a runtime check after the exact small model.
- **Readout.** Forecast loss, pair or polarity recovery, and detection lag on generated cases. Check stable false resets. Use fresh adaptation to decide which mechanism transfers. [Adams and MacKay, online changepoint detection](https://arxiv.org/abs/0710.3742) · [Fearnhead and Liu, multiple changepoints](https://doi.org/10.1111/j.1467-9868.2007.00601.x) · [Python reference](https://github.com/hildensia/bayesian_changepoint_detection).

### C2 · Share evidence across related machines

- **Question.** Can other machines help locate a recent shift when their own histories contain more evidence?
- **Arms.** Compare independent C1, existing shared-flip M4, and a partially shared hazard that allows independent deviations. Compare actual peer prefixes with a control that shifts observations toward older source ticks and a whole-episode peer permutation matched on history support. Apply peer updates with information available strictly before the prediction tick. Any extra lagged peer term competes with the shared-hazard arm so the two mechanisms can be distinguished.
- **Readout.** Shared and staggered synthetic change cases, M4's existing sparse masks, and real condition ratios. Compare the size of the effect against the negative controls. Treat peer dependence as predictive evidence within an episode. [Tank et al., categorical multivariate transitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10586348/) · [Prior adaptation stress test](05_adaptation_stress/README.md).

### W3 · Alternative sparse categorical model

- **Question.** Can a learned lag representation discover interactions beyond the three common pairs and transfer more reliably?
- **Arms.** Fit a small penalized categorical GLM or GBDT using frozen structured predictions. Compare explicit pair-parity terms for all 28 candidates with a train-selected sparse subset. Include the history-inferred direction and support so opposite directions across machines can produce opposite corrections. Allow full observed outcome categories only for selected pairs, with shrinkage toward binary parity. A second-stage research variant uses low-rank conditional tensor factors. Add a depth-8 binary context tree as an inexpensive contiguous-memory control if the initial comparison is inconclusive. Keep main hand-selected pairs and adaptation train-selected pairs as distinct arms.
- **Readout.** Same seed and fold controls, held-history R, pair recovery, and speed. Report memory and runtime before increasing lag depth. Compare to the 31-leaf tree, whose strong result depends on three preselected pairs. [Sarkar and Dunson, conditional tensor factorization](https://arxiv.org/abs/1506.06268) · [Willems, Shtarkov and Tjalkens, context-tree weighting](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf) · [LightGBM implementation and parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

### N1 · Masked neural residual forecast

- **Question.** Can a small global sequence model improve the structured forecast under sparse observations or unfamiliar lag pairs?
- **Arms.** Start from the existing factored GRU. Compare a GRU with observation masks and elapsed-time inputs to a causal dilated TCN at matched parameter count and training budget. Use payout embeddings, machine type, history masks, and prequential pair evidence. Let the network learn a zero-initialized residual on the win logit and a separate residual on the payout head. Compare raw sequence input against adding pair evidence, then test short-history augmentation as one final ablation. Fit three paired seeds per finalist. Use training histories for historical next-tick loss and genuine cutoff rows for any cutoff-only supervision.
- **Readout.** Performance by observation count and pair confidence, both heads' errors, seed dispersion, CPU runtime and forecast replay. Use the existing NumPy GRU exporter or a small CPU-friendly inference path. Fredy's ZIP builder at `bbff737` rejects Torch imports in the shipped graph, so verify inference isolation early. [Che et al., GRU-D paper and design](https://www.nature.com/articles/s41598-018-24271-9) · [GRU-D code](https://github.com/PeterChe1990/GRU-D) · [Bai, Kolter and Koltun, TCN](https://arxiv.org/abs/1803.01271) · [Reference causal TCN](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py) · [Fredy runtime builder](https://github.com/Amir-Suh/TAMU-Datathon-2027/blob/bbff7379e57984eb623588273ed1c958bb8b4f47/src/tdx/package/adapters.py).

### X1 · Bounded external forecasting comparator

- **Question.** Can pretrained temporal features add independent value after the small categorical models have been compared?
- **Arms.** Run one offline benchmark using a frozen Chronos model on pre-cutoff payout values or binary win histories. For the payout forecast, define the discrete conversion before scoring. Original Chronos sample forecasts can be binned using fixed midpoints between payout categories, with endpoint bins absorbing both tails. Compare its stand-alone probabilities and one fixed blend. For Chronos-2, check the quantile interface before using it. A coarse quantile grid only approximates bin probabilities. A DeepAR-style global autoregressor with a native categorical or factored head is the train-from-scratch comparator.
- **Readout.** Probability validity, held R, history support sensitivity, inference latency, and package size. Keep X1 a research benchmark until it matches the contest output and offline budget. [Original Chronos paper](https://arxiv.org/abs/2403.07815) · [Chronos-2 paper](https://arxiv.org/abs/2510.15821) · [Chronos repository](https://github.com/amazon-science/chronos-forecasting) · [Salinas et al., DeepAR](https://doi.org/10.1016/j.ijforecast.2019.07.001) · [GluonTS DeepAR module](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deepar/module.py).

## Known-law test matrix

Use separate generated fit and evaluation episodes. Draw each observed payout from the specified five-class distribution and save that distribution as the exact target. Reuse the same generated episodes, observed outcomes, cutoff, and mask seeds across each paired comparison.

| Fixture | Vary one mechanism | What a passing model should do |
|---|---|---|
| Stationary parity | Balanced binary contexts, then unequal 00/11 and 01/10 rates | Recover the ordered-cell correction when present and shrink toward parity in the equal-rate case. |
| Payout-state persistence | Constant, sticky, and independent state paths with identical emissions | Recover the constant-state control; show a held loss gain from transitions when they are present. |
| Observational sparsity | Known full prefixes with nested old-tick masks | Match explicit enumeration on tiny histories and improve pair evidence as observations are added. |
| Late and shared change | Polarity flip, amplitude ramp, pair swap, or a second flip | Distinguish which parameter changed, while keeping stable forecasts calibrated. Stagger episode changes to test how much sharing to retain. |
| Pair prior shift | Move prevalence from the three common main pairs to rare pairs | Preserve support for the new pairs and quantify how much target-training refits help under each history budget. |

Gate each statistical experiment on finite, normalized probabilities and on unchanged predictions after padding, row reordering, and at-cutoff or future-outcome perturbations of otherwise identical prefixes. For peer models, preserve whole episodes when testing row order and batching.

## Papers and repositories to inspect while building

| Reference | Read or inspect | Local use and limitation |
|---|---|---|
| [Linderman et al., Dynamax](https://joss.theoj.org/papers/10.21105/joss.07069) and [inference code](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference.py) | Forward filtering, predicted versus filtered state, time-dependent transition support | Numerical reference for Z2 and S2. Start with a small NumPy/SciPy inference implementation so JAX stays optional. |
| [hmmlearn CategoricalHMM](https://github.com/hmmlearn/hmmlearn/blob/main/src/hmmlearn/hmm.py) and [pomegranate HMM tutorial](https://github.com/jmschrei/pomegranate/blob/master/docs/tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.ipynb) | Plain categorical likelihood and dense versus sparse transitions | Cross-check the homogeneous HMM special case. Pair-conditioned emissions and latent lag bits require a custom wrapper. Pomegranate's README lists limitations on missing categorical distributions in its masked-tensor interface. [Missing-data notes](https://github.com/jmschrei/pomegranate). |
| [Johnson and Willsky](https://www.jmlr.org/papers/v14/johnson13a.html) and [pyhsmm](https://github.com/mattjj/pyhsmm) | Explicit duration, geometric dwell time in ordinary HMMs, and segment simulation | Use as a design reference for Z2 only after a transition model improves the constant-state comparison. |
| [Adams and MacKay](https://arxiv.org/abs/0710.3742), [Fearnhead and Liu](https://doi.org/10.1111/j.1467-9868.2007.00601.x), and [BOCPD code](https://github.com/hildensia/bayesian_changepoint_detection) | Run-length recursion, hazard, and multiple-change inference | Adapt the predictive likelihood to parity-conditioned Bernoulli or categorical outcomes. The supplied example likelihoods are reference implementations. |
| [Sarkar and Dunson](https://arxiv.org/abs/1506.06268) and [Tank et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10586348/) | Lag selection, interaction shrinkage, and multivariate categorical transitions | W3 needs explicit pair interactions. A mixture of single-lag effects by itself cannot express a pure parity interaction. |
| [Yao et al.](https://arxiv.org/abs/1704.02030), [loo implementation](https://github.com/stan-dev/loo/blob/master/R/loo_model_weights.R), and [LFO paper code](https://github.com/paul-buerkner/LFO-CV-paper) | Out-of-sample stacking and chronological validation | Borrow the evaluation design. Use exact episode and prefix splits plus contest-weighted quadratic loss in the local implementation. |
| [GRU-D repository](https://github.com/PeterChe1990/GRU-D), [TCN code](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py), and [GluonTS](https://github.com/awslabs/gluonts) | Observation masks, elapsed-time input, causal padding, global autoregressive batching | Match the task's discrete distribution and reuse local fit/export infrastructure. External code serves as an architectural reference. |
| [Chronos forecast pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) and [model card](https://huggingface.co/amazon/chronos-2) | Quantile API, local checkpoint loading, and covariate inputs | Benchmark separately. Check model terms, file size and offline runtime before considering an entry. |

## Adaptation intake and evidence of a ceiling

- After release, validate schema, episode grouping, support and payouts. Re-estimate pair prevalence over all 28 candidates from train prefixes, then check type-specific tail ratios and inferred flip behavior. Use the [adaptation stress runner](05_adaptation_stress/README.md) on the released data. Refit the structured control on that round before comparing a new model.
- Test two learning curves separately. For the **history curve**, mask nested older observations on the same complete-history cutoffs while keeping the last 40 spins. For the **training curve**, refit on **24, 48 and 96** episodes where available and score the same held episodes. Keep conditions and cutoff targets fixed across masks. Plot median and paired episode variability for both heads and for (4,7).
- If the training curve still improves, increase available training or reduce model variance. If only the history curve improves, prioritize inference from partial observations and priors for short histories. If both flatten and matched alternatives agree, test on generated sequences with known parameters and then with the latent state exposed. The gap between those two generated controls estimates the cost of hidden information under that generator.
- For handoff, each run should save `plan.json`, `folds.csv` or its hash, `validation_predictions.csv`, `scores.json`, a short findings note, and a forecast replay command. Record synthetic scores in their own file. Re-run the official format checker and an offline, fresh-folder replay for any candidate selected for submission.

### Existing reproduction commands

Run from the participant-kit root.

```sh
python3 -B signal_experiments/ceiling_analysis/analyze.py
python3 -B signal_experiments/score.py \
  best=signal_experiments/combined/m3_match_split_win_pair_l31_factor_gru_size.csv
```

[Verified score decomposition](ceiling_analysis/diagnostics.json) · [Slice data](ceiling_analysis/slices.csv) · [Saved branch and checkpoint replay](01_fredy_replay/README.md).
