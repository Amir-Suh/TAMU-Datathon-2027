# Assignment 03 · Sparse lagged peer effects

## Goal

Test whether individual peers carry predictive signal at particular tick delays after accounting for the focal machine's own history. The earlier logistic model scored 31.306 with pooled peer features and 31.444 without them. Simple averaging can dilute a relevant neighbor or cancel effects with different signs. Use the adaptive model at **33.224516/60** as the main control. [Previous features](../slot_forecasting/causal.py), [comparison](../runs/model-comparison/comparison.json).

Categorical mixture transition models use sparse contributions from individual lagged series, and generalized linear autoregressions extend this idea to Bernoulli sequences. Use these papers as guidance for a small number of lagged sources and strong shrinkage. In this challenge, an informative peer can reflect shared hidden state, so interpret the result as incremental forecasting evidence. [Tank et al., Convex Mixture Distribution](https://pmc.ncbi.nlm.nih.gov/articles/PMC10586348/), [Hall, Raskutti and Willett, autoregressive GLMs](https://arxiv.org/abs/1605.02693).

## Build

1. For each focal machine, enumerate each of the other seven machines at exact delays 1–8. Use binary win/loss first. Candidate feature `x_s` is the peer's lagged win flag centered by that peer's smoothed win rate before `s-lag`. When that spin is missing, skip the candidate's likelihood or gradient update. Also build a cheap control using lagged leave-one-out peer win rates and contributor counts pooled by type.
2. Compute an adaptive self-only win forecast `b_s` at each historical pseudo-cutoff. A correctness-first implementation can replay truncated manifests in batches, one offset from the released cutoff per pass, keeping one query per machine and truncating all peers at the same tick. Use the last 32 historical endpoints as the first window and cache inside your lane. For each peer/lag, estimate a shrunk local coefficient from pairs observed before the query. A one-step ridge-logistic estimate is `beta = sum(w_s*x_s*(y_s-b_s)) / (lambda + sum(w_s*x_s^2*b_s*(1-b_s)))`, with `w_s` a fixed 24-tick decay and `lambda = 2` as a starting point. Clip coefficients to `[-2,2]`. Zero corrections with fewer than eight matched observations. Fit train-fold type/signal priors without the held-out episode.
3. Require prospective evidence before selecting a peer/lag. Score the candidate's correction on the last 12 eligible forecast ticks of each prefix using estimates fit strictly before each of those ticks. Compare its observed Brier to the same self-only forecast on the same ticks. Pool the top one or two supported corrections after shrinking their contribution toward zero. Use a deterministic average over tied candidates, or use soft weights, so peer ordering is immaterial. When no peer is observed or none helps, return the adaptive forecast. Start with win only and keep its conditional positive mix fixed.
4. Run three matched comparisons on train folds before the tuning run. Compare self-only, pooled lagged peers, and selected individual peer/lag effects. Add a time-shift control using older peer ticks with an offset such as 11, holding the evaluated rows and validity mask as close as possible. If there is enough support, try one prespecified extension to 0× / 1× / 2× or more. Treat a gain shared by the aligned and shifted peer variants as weak evidence of a specific lagged link.

## Verification

- On a seeded synthetic episode where one peer's lag-3 outcome helps and another has the opposite sign, recover the planted direction on sufficiently long prefixes. Check an unrelated-peers case for false corrections.
- Shuffle the order of machines and change peer IDs consistently in queries and histories. Probabilities mapped back to the same machines must agree to numerical precision. Changing the focal history should affect the self forecast but should never put that machine in its own peer set.
- Change every peer outcome at the forecast tick and later and verify unchanged features, selected peers, and predictions at the forecast tick. Test fully missing peer histories and confirm exact adaptive fallback.
- Report support, selected lag counts and signs, and the aligned-versus-shifted result. Report shared-change and short-history ratios as well as overall points. Compare candidate, self-only, and pooled peer control on identical tuning IDs.

## Execution and handoff

Own `slot_forecasting/peer_lags.py` and `run_peer_lags.py`. Write to `runs/03-peer-lags/<run-id>/`. Use train episodes to select settings, then use `run_model_comparison.tuning_data()` and the official `contest_score` on the 48 tuning episodes. Reserve audit and test. Return runnable code, config, verification, `tuning_predictions.csv`, and `report.json` with points, condition ratios, paired episode uncertainty and runtime. See the [shared experiment contract](README.md#shared-experiment-contract).
