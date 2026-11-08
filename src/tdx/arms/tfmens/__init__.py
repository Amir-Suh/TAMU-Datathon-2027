from .arm import TfmEnsArm

VARIANTS = {'ens': {}}


def build(**params):
    return TfmEnsArm(**params)
