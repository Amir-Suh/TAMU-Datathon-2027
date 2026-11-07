"""Model 4: model 3 plus an episode-level 'shared flip time' alternative.

All the machinery lives in FlipLagPairModel; this model only switches on
``share_episode`` so the two can be compared side by side.
"""
from __future__ import annotations

from ..m3_lag_pair_flip.model import FlipLagPairModel


class SharedFlipLagPairModel(FlipLagPairModel):
    name = "m4_lag_pair_flip_shared"

    def __init__(self, **kwargs):
        kwargs.setdefault("share_episode", True)
        super().__init__(**kwargs)
