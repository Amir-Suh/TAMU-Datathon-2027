from .arm import SeqEnsArm

VARIANTS = {'ens': {}}


def build(**params):
    return SeqEnsArm(**params)
