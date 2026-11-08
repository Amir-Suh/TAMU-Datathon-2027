# Main candidate and replay

**W1a + P0** combines the pooled ordered-cell win forecast with the hard train-history tail projection. It scored **56.425331 / 60** on the 96 released main validation episodes (768 queries). The hybrid scored 56.341022 and frozen M3 scored 55.268370. [Scores versus the hybrid](assembly_vs_hybrid.json) · [Scores versus M3](assembly_vs_m3.json)

| Comparison | ΔR | Paired 95% interval |
| --- | ---: | ---: |
| W1a + P0 vs hybrid | −0.001591 | [−0.002449, −0.000821] |
| W1a + P0 vs W1a | −0.000846 | [−0.001258, −0.000453] |
| W1a + P0 vs P0 | −0.000744 | [−0.001433, −0.000123] |
| W2 + P0 vs W1a + P0 | −0.000020 | [−0.000046, +0.000002] |

W1a + P0 reduced condition R in stable, short_history, drift, and shared_change by 0.002831, 0.001669, 0.000263, and 0.001603 respectively. On the 248 frozen-M3 (4,7) queries, p0 MSE dropped from 0.006453 to 0.006210. The overall mean tail SSE dropped from 0.004275 to 0.004039. W2's point gain over the combination was 0.001054 and its paired interval crossed zero. [Tail increment](assembly_vs_W1a.json) · [Win increment](assembly_vs_P0.json) · [W2 increment](assembly_vs_W1a_P0.json)

The [matching test CSV](assembly/test_W1a_P0.csv) and [code ZIP](../package/02_w1a_p0/w1a_p0_code.zip) passed a clean offline Linux ARM64 Python 3.13.18 replay with read-only data and source, a two-CPU cgroup, a 16 GiB memory limit, and a 30-minute timeout. Inference took 9.11 seconds. All 1,536 test IDs matched and the maximum difference after probability normalization was 1.38e−14. The independent coordinator check found 37 ZIP members, with no forecasts packaged as inputs. The output directory was host-backed and the forecast was below the organizer's output limit. [Package report](../package/02_w1a_p0/REPORT.md) · [Independent package audit](package_audit.json) · [Verified hybrid and M3 fallbacks](../package/01_frozen_controls/REPORT.md)

The coordinator re-scored immutable forecasts after the fit phase. The training episode folds are disjoint. The main score is reused development evidence with archived tree/GRU exposure to pre-cutoff histories across splits. The new fitting and packaged inference traces show no protected-answer reads. Test inference accessed the test queries and pre-cutoff history files. [Input audit](input_audit.json) · [Main assembly trace summary](preliminary_trace_audit.json) · [Container test access trace](../package/02_w1a_p0/w1a_p0_test_access.jsonl)
