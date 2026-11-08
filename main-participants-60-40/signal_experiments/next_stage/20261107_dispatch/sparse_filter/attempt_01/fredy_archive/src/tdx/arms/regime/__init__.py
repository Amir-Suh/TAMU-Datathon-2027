"""Condition routing, local transition specialists and pooled changepoints."""
from .arm import RegimeArm

VARIANTS = {
    'd2_routed': {'method': 'd2_routed'},
    'd3_changepoint': {'method': 'd3_changepoint'},
    'd2_d3_mix': {'method': 'd2_d3_mix'},
}


def build(method='d2_routed', specialists=None, transductive=False):
    return RegimeArm(method=method, specialists=specialists, transductive=transductive)
