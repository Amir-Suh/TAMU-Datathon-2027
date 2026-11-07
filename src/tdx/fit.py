"""Explicit supervision permissions for every model fit."""
from dataclasses import dataclass

@dataclass(frozen=True)
class FitSpec:
    tick_splits: tuple[str, ...]
    label_splits: tuple[str, ...]
    soft_splits: tuple[str, ...]

def dev_spec(transductive=False):
    return FitSpec(('train', 'validation', 'test') if transductive else ('train',), ('train',), ())

def final_spec():
    return FitSpec(('train', 'validation', 'test'), ('train', 'validation'), ('validation',))
