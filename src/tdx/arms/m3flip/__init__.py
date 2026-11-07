from .arm import M3FlipArm

VARIANTS = {'m3': {}}
RUNTIME_FIT = True   # packaging: fitted on the released train split at inference time (about 12 s)


def build(**params):
    return M3FlipArm(**params)
