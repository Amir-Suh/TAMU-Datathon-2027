# Frozen main fallback packages

Both fallback pairs passed the organizer-style replay gate. Pair the code ZIP with its archived test CSV.

| Arm | Archived test CSV | Code ZIP | Test max normalized difference | Wall time | Peak RSS |
|---|---|---|---:|---:|---:|
| Hybrid | [Test CSV](../../../../combined/m3_match_split_win_pair_l31_factor_gru_size_test.csv) | [Code ZIP](hybrid_code.zip) | 1.3763e-14 | 8.743 s | 594.9 MiB |
| M3 | [Test CSV](../../../../../m5_m6_repair/results/repaired-m5-m6/m3/submission.csv) | [Code ZIP](m3_code.zip) | 1.3552e-14 | 1.044 s | 399.9 MiB |

Each replay had 1,536 exact test IDs and 768 exact validation IDs. The validation replay max normalized differences were 9.6589e-15 and 9.8255e-15. Both candidate and replay CSVs passed the kit checker. The original hybrid scored 56.341022 / 60 and M3 scored 55.268370 / 60 on released main validation.

Both archives contain source and fitted state. M3 state was exported losslessly from the hash-checked joblib checkpoint into JSON. Hybrid adds a JSON match-cell contrast, the archived LightGBM booster and three factored NumPy GRU weight files. The archive requirements are pinned and inference uses one numerical thread.

The replays used official Linux/aarch64 Python 3.13.18 with no network, read-only `/submission` and `/data`, a writable `/output`, and a 128 MiB scratch mount. The measured process used the exact argument array in each ZIP. The Docker CPU and memory cgroups confirmed 2 cores and 16 GiB, with a wall timeout of 1,800 seconds. Docker Desktop could not directly bind the host Documents folder. The harness mounted a read-only scratch copy of the full released round and extracted each ZIP into a fresh folder. The host-backed output directory was writable and the generated files were below the 32 MiB organizer output allowance.

A guarded replay regenerated each test forecast byte-for-byte and read only its two test inputs. The guarded validation replay read only its two validation inputs. Both loaders reject observations at or after the cutoff. Validation odds were read afterward by the separate kit checker. This rehearsal used the official Python base plus pinned dependencies, with the local Docker host as the execution environment.

[Hybrid replay and hashes](hybrid_replay.json) · [M3 replay and hashes](m3_replay.json) · [Format log](format_and_parity.log) · [Manifest](manifest.json) · [Reproduction steps](REPLAY.md)
