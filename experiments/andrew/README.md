# Andrew's early experiment track

These experiments establish the controls that preceded Amir's M3 and the joint win/payout hybrid. Each runner keeps its original model code and archived inputs. Its default data path points to the canonical main kit at the repository root. [Early results](EXPERIMENT_RESULTS.md) · [Training plan](ML_TRAINING_PLAN.md) · [Model briefs](model_briefs/)

| Experiment family | Runner |
| --- | --- |
| Frequency, logistic, and adaptive lag baselines | `run_model_comparison.py` |
| Signed motifs, peer lags, and transition retrieval | `run_signed_motifs.py`, `run_peer_lags.py`, `run_transition_retrieval.py` |
| Regime filters and adaptive update rates | `run_regime_filter.py`, `run_adaptive_weight_sweep.py` |
| Boosted win corrections and temporal CNN | `run_boosted_correction.py`, `run_temporal_cnn.py` |
| Bayesian lag pairs and hyperparameter selection | `run_bayesian_lag_pairs.py`, `run_bayesian_lag_pair_tuning.py` |
| Win and payout combinations | `run_combine_forecasts.py` |

Archived forecasts, settings, and scores are under [runs](runs/). The model comparison starts from an earlier split protocol, so use the full-validation row when comparing with Amir or Fredy. [Bayesian experiment plan](BAYESIAN_LAG_PAIR_PLAN.md) · [Tuning and audit plan](BAYESIAN_LAG_PAIR_TUNING_PLAN.md)

Example from the repository root with the [model requirements](requirements-models.txt) installed. Use a fresh output directory.

```sh
out=$(mktemp -d)
python3 -B experiments/andrew/run_model_comparison.py --out "$out/comparison"
```

The later M5/M6 repairs and the hybrid work have separate [kit-relative locations](../../docs/EXPERIMENTS.md#experiment-families).
