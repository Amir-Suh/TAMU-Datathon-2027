# gbdt

Run the eight pilots with `~/.venvs/tdx/bin/python -m tdx.sweep --arm gbdt`.
All training and prediction uses one LightGBM thread by default. `threads` may be
set to 1..8; pin 1 for the verified deterministic CPU path. No GPU is used.

Feature schema: 46 columns, documented in features.py. Lags use calendar ticks;
EMA half-lives and win counts use observed observations. Neighbour snapshots
use each other machine's last observed payout, while 5/10 windows pool observed
cells in calendar windows. Empty rates/aggregates are -1; empty divergence is
NaN (LightGBM missing). All features exclude the current tick. Signal is unused.
Training needs at least eight prior own observations. last_n is a calendar-tick
restriction relative to that machine's last observed tick, not a row sample.

W2's specific final-only cutoff rule takes precedence over the common contract's
permission to use train cutoff labels in dev. Final adds permitted hard labels
and validation soft targets as weighted class rows; dev uses history alone.
Early stopping holds out the SHA256-sorted first floor(0.2 * train episodes)
episodes. No validation conditions or targets select boosting rounds. All rows
are refitted at the selected round count. Set-derived supervision split names
are sorted to keep final row order stable across Python hash seeds.

`GBDTArm.save` stores model.txt and features.json. `GBDTArm.load` predicts using
the same causal features and clipping. Include the adjacent requirements.txt
pin in W6's submission requirements; root packaging files belong to W6.
