from .arm import SeqLossArm

_B = {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'factor': True}
VARIANTS = {
    'ce': dict(_B),                                              # reference: identical to seq gru128_factor
    'brier': dict(_B, loss='brier'),                             # pure Brier on the realized next outcome
    'ce_brier2': dict(_B, loss='ce_brier', beta=2.0),            # cross-entropy + 2 * Brier
    'size2': dict(_B, size_w=2.0),                               # win BCE + 2 * size CE (on wins)
    'size4': dict(_B, size_w=4.0),
    'rec64': dict(_B, recency_tau=64.0),                         # ticks near each episode's cutoff weigh more
    'lenw': dict(_B, len_pow=0.5),                               # machines with short observed histories weigh more
    'valw': dict(_B, val_pow=0.5),                               # weight by payout value ** 0.5
}


def build(**params):
    return SeqLossArm(**params)
