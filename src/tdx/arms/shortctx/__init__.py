from .arm import ShortCtxArm

VARIANTS = {
    # first agy pilot: 20 epochs, small; validation 44.4, short_history backtest Brier 0.4032 (worse than gru_fac / m3)
    'pilot20': {
        'd': 64,
        'h': 64,
        'mlp_h': 64,
        'epochs': 20,
        'lr': 2e-3,
        'wd': 1e-2,
        'batch_size': 256,
        'dropout': 0.1,
        'transductive': True,
        'seed': 7,
        'batches_per_epoch': 250,
        'exact_ratio': 0.5,
    },
    # longer / wider follow-up (5x epochs, 2x width); this is what tail_eval and short_backtest score as 'shortctx'
    'pilot': {
        'd': 128,
        'h': 128,
        'mlp_h': 128,
        'epochs': 100,
        'lr': 2e-3,
        'wd': 1e-2,
        'batch_size': 256,
        'dropout': 0.1,
        'transductive': True,
        'seed': 7,
        'batches_per_epoch': 250,
        'exact_ratio': 0.5,
    },
}


def build(**params):
    return ShortCtxArm(**params)
