# W1a + P0 main entry package

**Advance.** The frozen [test CSV](../../coordinator/assembly/test_W1a_P0.csv) and [code ZIP](w1a_p0_code.zip) passed the local organizer-style reproduction gate. The code runs from fitted state and released histories.

| Check | Result |
|---|---:|
| Released validation points | 56.425331 / 60 |
| Test IDs | 1,536 exact |
| Validation IDs | 768 exact |
| Test maximum normalized probability difference | 1.3763e-14 |
| Validation maximum normalized probability difference | 9.6589e-15 |
| Test wall time | 9.106 s |
| Test peak RSS | 593.5 MiB |
| Test CSV SHA-256 | `c7c34efd491648b5a7cf97bfa04df090427fc528dde31a28800211575a856b8b` |
| ZIP SHA-256 | `e4610accbfb69604d7497b01afb2e59682df10ca9ec32459dd733fe59cf78451` |

The ZIP reconstructs W1a's pooled ordered-cell win head around the frozen M3 posterior, using strength 20 and its fitted train contrasts. It reconstructs the payout donor with the archived 31-leaf booster and three factored NumPy GRUs, then uses the P0 training type counts to redistribute the 2x, 5x and 20x mass. It reads only the split query and pre-cutoff history files. The exact-command and guarded test replays produced identical bytes. The kit checker ran separately from inference and passed all four frozen and replay CSVs.

The replay used official Linux/aarch64 Python 3.13.18, pinned dependencies, network disabled, a two-core CPU quota, a 16 GiB RAM cap and a 30-minute timeout. Docker mounted the extracted package and a scratch copy of the full released round read-only. The host-backed output directory was writable and generated files were below the 32 MiB organizer allowance.

[Parity and resource report](w1a_p0_replay.json) · [Test access trace](w1a_p0_test_access.jsonl) · [Validation access trace](w1a_p0_validation_access.jsonl) · [Manifest](manifest.json) · [Replay procedure and fallback pairs](../01_frozen_controls/REPORT.md) · [Paired comparison against hybrid](../../coordinator/assembly_vs_hybrid.json) · [Paired comparison against M3](../../coordinator/assembly_vs_m3.json)
