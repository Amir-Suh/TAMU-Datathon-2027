from .arm import TfmArm

VARIANTS = {
    'tfm_d64_l2': {'d': 64, 'L': 2, 'heads': 4, 'epochs': 100, 'lr': 1e-3, 'wd': 1e-2, 'batch': 32, 'transductive': True, 'factor': False},
    'tfm_d128_l3': {'d': 128, 'L': 3, 'heads': 4, 'epochs': 100, 'lr': 1e-3, 'wd': 1e-2, 'batch': 32, 'transductive': True, 'factor': False},
    'tfm_d128_l3_fac': {'d': 128, 'L': 3, 'heads': 4, 'epochs': 100, 'lr': 1e-3, 'wd': 1e-2, 'batch': 32, 'transductive': True, 'factor': True},
}


def build(**params):
    return TfmArm(**params)
