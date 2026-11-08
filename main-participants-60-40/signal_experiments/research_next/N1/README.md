# N1 · Masked neural residual forecasts

The best residual was the GRU with explicit lag-pair evidence, at **55.121922 / 60**. It improved the matched causal structure baseline by ΔR −0.007000 [−0.011150, −0.002895] and trailed the frozen hybrid by +0.023553 [+0.016610, +0.030468]. The hybrid remains the incumbent. [Official scores and paired comparisons](scores.json)

| Forecast | Points / 60 | R | p0 MSE | Win-weighted payout SSE |
|---|---:|---:|---:|---:|
| Frozen hybrid | 56.341022 | 0.064943 | 0.004468 | 0.006448 |
| Matched structural baseline | 54.769686 | 0.095497 | 0.005971 | 0.010441 |
| GRU, raw sequence | 55.076123 | 0.089401 | 0.005275 | 0.010147 |
| GRU, pair evidence | 55.121922 | 0.088496 | 0.005181 | 0.010077 |
| TCN, raw sequence | 54.968844 | 0.091527 | 0.005521 | 0.010198 |
| TCN, pair evidence | 55.052316 | 0.089872 | 0.005371 | 0.010089 |

Explicit pair evidence lowered R versus raw inputs by 0.000905 for the GRU (paired interval −0.001841 to +0.000055 for ΔR) and 0.001655 for the TCN (−0.002756 to −0.000608). With evidence, TCN versus GRU ΔR was +0.001376 [−0.000578, +0.003387]. Single-seed points ranged from 55.048465 to 55.080524 for the evidence GRU and from 54.854940 to 55.023038 for the evidence TCN. The table reports the equal probability average over the three seeds. Intervals use 10,000 condition-stratified paired episode bootstrap draws on the fixed predictions. [Per-seed forecasts and paired scores](scores.json)

The evidence GRU lowered p0 MSE from 0.007917 to 0.007602 on 192 rows with 64 observations and from 0.008179 to 0.006943 on the 248 frozen-M3 inferred (4,7) rows. Hybrid MSE was 0.006791 and 0.006453, respectively. On the 29 queries with structural pair confidence below .7, the evidence GRU's p0 MSE was 0.010024 versus 0.012936 for the hybrid, and its total loss contribution declined by 0.000603 in R [−0.004241, +0.001778]. The wide hybrid gap on the full comparison and the 64-observation slice ended the ablation before sparse augmentation. [Observation support, pair confidence and lag slices](slices.csv)

## Matched design

- The shared structural offset tracks 28 lag pairs and both directions with a null component. Fixed population rates and pair priors come from M3 fits on training histories. Each direction can change each tick with probability 1/256. Missing outcomes advance that transition without a likelihood update. An unavailable lag uses the marginal type win rate. The size offset uses type-pooled positive counts and recency-weighted machine counts with a 32-tick half-life and prior strength 24.
- The historical training offsets come from a four-fold training cross-fit that excludes the target episodes from population priors. A tick's structural and neural predictors use observations strictly before that tick. Historical ticks use neutral signal. The actual training cutoff signal enters only at the matching cutoff, and the realized training cutoff supplies one extra supervised outcome per machine. All main training histories determine the priors used at validation inference.
- Both networks receive a payout embedding from the previous tick, a machine-type embedding, observation mask, elapsed time and the cutoff-only signal. The evidence arms also receive four pre-outcome structure summaries. The raw arms use zeros in those channels. The GRU has 20 hidden units and 2,430 parameters. The 12-channel TCN has 2,414 parameters and seven causal width-2 blocks with dilations 1 through 64, for a 128-tick receptive field. Both add zero-initialized win and conditional-payout logit corrections to the same offsets.
- Both models use identical eight-epoch budgets, batch size 48, AdamW with learning rate .003, and three paired seeds. All neural supervision comes from training episodes. Official metrics use main public validation, and the frozen hybrid is the common incumbent with its previously documented transductive history exposure. [Inputs, configuration and versions](plan.json)

## Replay

The 12-episode smoke runs took 0.585 seconds for the GRU and 0.043 seconds for the TCN for one epoch including their optimizer setup. All 12 fits passed the future-outcome perturbation check at an earlier cutoff. Torch and NumPy logits agreed within 2.09e−7, and forecast inference uses the NumPy path in both evaluation and replay. The NPZ contains the learned residual weights and frozen structural priors. The final two-thread training and scoring run took 38.44 seconds. A 2.64-second fresh-folder replay blocked Torch imports and training-data reads and reproduced all 17 forecast CSVs byte for byte. The official checker accepted the structural baseline and all four seed averages. [NumPy inference](infer.py) · [Exported weights](weights.npz)

```sh
d=signal_experiments/research_next/N1
python3 -B "$d/run.py" --smoke
python3 -B "$d/run.py"
replay_dir=$(mktemp -d)
python3 -B "$d/run.py" --replay "$d/weights.npz" --out "$replay_dir"
cmp "$d/gru_pair_evidence.csv" "$replay_dir/gru_pair_evidence.csv"
python3 -B check_submission.py "$d/gru_pair_evidence.csv" --split validation
```
