from .arm import Tfm2Arm

VARIANTS = {
    'conv9': {
        'd': 64, 'L': 2, 'heads': 4, 'epochs': 60, 'lr': 2e-3, 'wd': 1e-2,
        'batch': 32, 'transductive': True, 'conv': 9, 'alibi': False, 'warmup': True, 'dropout': 0.1
    },
    'alibi': {
        'd': 64, 'L': 2, 'heads': 4, 'epochs': 60, 'lr': 2e-3, 'wd': 1e-2,
        'batch': 32, 'transductive': True, 'conv': 0, 'alibi': True, 'warmup': True, 'dropout': 0.1
    },
    'conv9_alibi': {
        'd': 64, 'L': 2, 'heads': 4, 'epochs': 60, 'lr': 2e-3, 'wd': 1e-2,
        'batch': 32, 'transductive': True, 'conv': 9, 'alibi': True, 'warmup': True, 'dropout': 0.1
    },
    'd128l3': {
        'd': 128, 'L': 3, 'heads': 4, 'epochs': 100, 'lr': 2e-3, 'wd': 1e-2,
        'batch': 32, 'transductive': True, 'conv': 9, 'alibi': True, 'warmup': True, 'dropout': 0.1
    },
}


def build(**params):
    return Tfm2Arm(**params)
