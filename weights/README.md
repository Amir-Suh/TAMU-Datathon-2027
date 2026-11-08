# Trained model weights (dev mode: fit on train labels + unlabelled train/validation/test histories; no validation answers)

All members are numpy float64 exports (`model.npz` + `meta.json`), loadable WITHOUT torch. Trained on the main round (`data/main`, metadata sha256 7979323052df471e6b8b2c985cb48faa89e287ac6b0fd7789f24bccd544ce7bf).

| Folder | Arm / loader | Members | Training | Validation points (dev mode) |
|---|---|---|---|---|
| `seqens_ens12_dev/` | `tdx.arms.seqens.arm.SeqEnsArm.load(dir)` | 12 GRUs, `member_00..11` = seeds 11..22 | `seq` variant `gru128_factor` (h=128, factored win/size head, plain cross-entropy, 100 epochs) | 55.773 |
| `tfmens_ens8_dev/` | `tdx.arms.tfmens.arm.TfmEnsArm.load(dir)` | 8 transformers, `member_00..07` = seeds 21..28 | `tfm2` variant `d128l3` (d=128, 3 layers, 4 heads, causal conv k=9, ALiBi, warmup, plain cross-entropy, 100 epochs) | 56.241 |
| `seqens_size2_ens/` | `SeqEnsArm.load(dir)` | 12 GRUs, `member_00..11` = seeds 41..52 | `seqloss` variant `size2` (same GRU, loss = win BCE + 2 x size CE on wins; see each member's `loss.json`) | 56.405 |

The first two are exactly the members packaged inside `submissions/candidate3b_*/entry_main.zip` (the original 3b and, with a fixed m3, Andrew's clean refit on `andrew-exp`). `seqens_size2_ens/` is NOT in any submitted zip yet.

Verify: `cd weights && sha256sum -c SHA256SUMS`, then
```
PYTHONPATH=src python - <<'PY'
from tdx.data import load_round, with_answers; from tdx.score import score
from tdx.arms.seqens.arm import SeqEnsArm
r = load_round('data/main', answers=False); t = with_answers(r)
print(score(t, SeqEnsArm.load('weights/seqens_size2_ens').predict(r.splits['validation']))['points'])   # 56.405
PY
```
Regenerate from scratch: `claude_output/sh/size2_ensemble.sh` (size2 GRUs), `claude_output/sh/round2.sh` (all ensembles on a new round). Training is deterministic for a fixed seed on the same hardware (two same-seed `seq` runs matched exactly); not guaranteed across GPUs.
