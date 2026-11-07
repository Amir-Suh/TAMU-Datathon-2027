# Model 4: flip-aware lag-pair model with a shared episode flip

**Validation: 55.28 / 60** (model 3: 55.27). Code: [model.py](model.py), a thin subclass of model 3.

## Idea

The rules say some relationships "change together across an episode". Model 4 adds an alternative to model 3's independent flips: **all 8 machines in an episode share one flip time**. The two are combined by Bayesian evidence.

- **Shared model:** the episode's posterior over the flip time τ is the τ prior times the product of every machine's likelihood at that τ, summed over each machine's pair, direction and the null. Each machine's forecast then averages over this shared τ posterior.
- **Mixing:** P(shared | episode) compares the shared evidence with the product of the machines' independent evidences, under a prior learned on train (0.67).

## Result

| | Stable | Short history | Drift | Shared change |
|---|---|---|---|---|
| Model 3 | 0.077 | 0.107 | 0.080 | 0.078 |
| Model 4 | 0.077 | 0.106 | 0.080 | 0.078 |

**No meaningful gain.** The difference of 0.01 points is far inside the noise (an episode bootstrap of about ±0.5 points).

**Why:**
- After a flip, each machine typically has about 70 ticks of post-flip data, so it can locate its own flip without help.
- Shared-change machines do flip at essentially the same tick (spread within an episode about 1 tick, permutation p < 0.001; [FINDINGS.md](../FINDINGS.md) §2). But since each machine already finds that tick on its own, sharing adds no new information.

Model 4 is kept for comparison, and as a candidate for round two. Synchronized flips would matter there if flips come later or histories are shorter, leaving fewer post-flip spins per machine.

## Run it

```sh
python -m models.run --models m4
```
