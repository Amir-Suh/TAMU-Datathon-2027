"""Vectorized versions of the organizer's frequency baselines."""

from .arm import PooledArm

VARIANTS = {
    "pooled": {"method": "pooled"},
    "pooled_recent": {"method": "pooled_recent"},
    "recent": {"method": "recent"},
    "context": {"method": "context"},
}


def build(method="pooled", prior_strength=20.0, half_life=24.0,
          transductive=False):
    return PooledArm(method, prior_strength, half_life, transductive)
