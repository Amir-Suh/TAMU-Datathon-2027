# Pilot results (validation, dev mode, main round)

Generated 2026-11-07 with `python -m tdx.tools.compare runs/ --baseline runs/pooled/pooled_recent`. Points = 60/(1+R); ratios are per-condition loss vs the frozen pooled baseline (lower is better). Intervals are paired episode bootstrap (2000 resamples) vs pooled_recent. The seq epoch count was picked by watching validation, so its figures are mildly optimistic. Rows repeat when a variant was run more than once (seq seeds 1-3, gbdt reruns). runs/<arm>/ folders are git-ignored; the score log is runs/LOG.md.

```
arm     variant         points  stable  short  drift  shared  seconds  Δpoints  95% interval      
seq     gru128_nomsg    53.578  0.090   0.179  0.121  0.088   199.74   +22.752  [+21.998, +23.468]
seq     gru128_nomsg    53.382  0.113   0.171  0.124  0.088   201.84   +22.557  [+21.635, +23.436]
seq     gru128_nomsg    53.098  0.085   0.217  0.110  0.109   199.39   +22.272  [+21.175, +23.255]
gbdt    trans_64_l15    36.395  0.604   0.725  0.682  0.583   96.28    +5.570   [+4.760, +6.374]  
gbdt    trans_64_l15    36.395  0.604   0.725  0.682  0.583   84.25    +5.570   [+4.760, +6.374]  
gbdt    trans_64_l15    36.226  0.620   0.734  0.684  0.587   99.11    +5.400   [+4.631, +6.157]  
gbdt    trans_64_l15    36.191  0.613   0.738  0.697  0.584   97.90    +5.365   [+4.575, +6.172]  
gbdt    all_inf_l15     35.123  0.660   0.790  0.749  0.633   26.95    +4.298   [+3.528, +5.053]  
gbdt    all_64_l15      34.362  0.689   0.837  0.788  0.671   25.57    +3.536   [+2.818, +4.243]  
gbdt    last128_64_l15  34.253  0.723   0.841  0.782  0.660   16.50    +3.428   [+2.686, +4.138]  
gbdt    trans_24_l7     34.180  0.737   0.833  0.790  0.662   94.56    +3.354   [+2.694, +3.999]  
gbdt    all_64_l7       33.526  0.763   0.863  0.839  0.693   22.01    +2.700   [+2.063, +3.331]  
gbdt    all_24_l15      33.506  0.780   0.856  0.842  0.685   15.96    +2.680   [+2.042, +3.319]  
gbdt    last64_24_l15   33.088  0.799   0.866  0.870  0.719   5.36     +2.262   [+1.627, +2.884]  
glm     l8              31.093  0.954   0.971  0.944  0.850   66.95    +0.267   [-0.426, +0.997]  
glm     l4              30.963  0.983   0.956  0.945  0.867   55.51    +0.137   [-0.348, +0.633]  
glm     lambda100       30.878  0.981   0.997  0.937  0.858   36.64    +0.052   [-0.316, +0.439]  
glm     l2              30.844  1.006   1.006  0.926  0.843   44.89    +0.018   [-0.363, +0.455]  
pooled  pooled_recent   30.826  0.958   1.001  0.946  0.880   0.07     +0.000   [+0.000, +0.000]  
pooled  pooled_recent   30.826  0.958   1.001  0.946  0.880   0.07     +0.000   [+0.000, +0.000]  
regime  d2_routed       30.779  0.970   1.013  0.950  0.864   1.80     -0.046   [-0.355, +0.244]  
glm     tau32           30.745  1.002   0.965  0.975  0.864   32.87    -0.080   [-0.628, +0.470]  
pooled  recent          30.743  0.986   1.018  0.938  0.865   0.09     -0.083   [-0.291, +0.142]  
regime  d2_routed       30.476  0.986   1.023  0.966  0.900   2.26     -0.350   [-0.674, -0.033]  
regime  d2_d3_mix       30.187  0.997   1.024  0.993  0.936   2.14     -0.639   [-1.006, -0.287]  
pooled  pooled          30.000  1.000   1.000  1.000  1.000   0.07     -0.826   [-1.138, -0.512]  
glm     lambda1         29.759  1.021   1.016  1.062  0.966   47.77    -1.067   [-1.865, -0.260]  
regime  d3_changepoint  29.710  1.019   1.028  1.038  0.993   2.13     -1.116   [-1.576, -0.678]  
pooled  context         29.685  0.990   1.080  1.030  0.985   0.08     -1.141   [-1.520, -0.773]  
```
