# Combined win and payout heads

The four-cell M3 experiment changes the win probability and preserves M3's positive payout mix. The Fredy experiment found that the tree's positive payout mix improves M3 with win probabilities held fixed. The forecasts here combine those changes without fitting another model or blend weight. Public main validation serves as a development comparison.

```sh
d=signal_experiments
python3 -B "$d/combine_heads.py" --queries public/validation/queries.csv \
  --win "$d/04_four_lag_cells/validation_predictions.csv" \
  --size "$d/01_fredy_replay/forecasts/pair_l31.csv" \
  --out "$d/combined/m3_match_split_win_pair_l31_size.csv"
python3 -B "$d/combine_heads.py" --queries public/test/queries.csv \
  --win "$d/04_four_lag_cells/test_predictions.csv" \
  --size "$d/01_fredy_replay/test_forecasts/pair_l31.csv" \
  --out "$d/combined/m3_match_split_win_pair_l31_size_test.csv"
```

The [combined forecast](m3_match_split_win_pair_l31_size.csv) scored **55.833268 / 60**. Against frozen M3, ΔR was −0.010984 (paired 95% interval −0.015999 to −0.006288). Against M3 win plus the same tree payout mix, ΔR was −0.001569 (−0.003005 to −0.000302). See [comparison against M3](pair_comparison.json) and [increment from the four-cell win head](pair_increment.json). The [test forecast](m3_match_split_win_pair_l31_size_test.csv) passed the kit format checker for 1,536 queries.

The same swap with the equal tree and seven-seed GRU mixture scored **56.254033 / 60**. Against frozen M3, ΔR was −0.019022 (paired 95% interval −0.023330 to −0.014954). All four conditions improved. Against the same payout mixture with the unmodified M3 win head, splitting 00/11 adds 0.081480 points (paired ΔR −0.001547, interval −0.002967 to −0.000302). [Full comparison](full_comparison.json) · [Win-head increment](gru_increment.json).

For this version, use `forecasts/pair_l31_gru.csv` and `test_forecasts/pair_l31_gru.csv` as the size-source files in the same commands. [Validation forecast](m3_match_split_win_pair_l31_gru_size.csv) · [Format-checked test forecast](m3_match_split_win_pair_l31_gru_size_test.csv).

The factored GRU comparison uses the same composition with the 31-leaf tree and the three-seed factored GRU mean. The resulting forecast scored **56.341022 / 60**, **1.072652 points above frozen M3**, with lower loss in all four conditions. Against frozen M3, ΔR was −0.020668 (paired 95% interval −0.025110 to −0.016367). Against the seven-seed plain-GRU swap, ΔR was −0.001647 (−0.002706 to −0.000621). That comparison changes GRU formulation, seed count, and training backend together. [Comparison against M3](factored_comparison.json) · [Comparison against plain GRUs](factored_vs_plain.json).

Use `forecasts/pair_l31_factor_gru.csv` and `test_forecasts/pair_l31_factor_gru.csv` as the size-source files to recreate this version. [Validation forecast](m3_match_split_win_pair_l31_factor_gru_size.csv) · [Format-checked test forecast](m3_match_split_win_pair_l31_factor_gru_size_test.csv).
