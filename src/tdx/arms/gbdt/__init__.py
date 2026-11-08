from .arm import GBDTArm

VARIANTS = {
    'all_inf_l15': {},
    'all_64_l15': {'tau': 64},
    'all_24_l15': {'tau': 24},
    'last64_24_l15': {'tau': 24, 'last_n': 64},
    'last128_64_l15': {'tau': 64, 'last_n': 128},
    'all_64_l7': {'tau': 64, 'leaves': 7},
    'trans_64_l15': {'tau': 64, 'transductive': True},
    'pair_trans_64_l15': {'tau': 64, 'transductive': True, 'drop_pair': False, 'drop_neighbour': True},
    'pair_tau24': {'tau': 24, 'transductive': True, 'drop_pair': False, 'drop_neighbour': True, 'threads': 3},
    'pair_tauinf': {'transductive': True, 'drop_pair': False, 'drop_neighbour': True, 'threads': 3},
    'pair_l31': {'tau': 64, 'leaves': 31, 'transductive': True, 'drop_pair': False, 'drop_neighbour': True, 'threads': 3},
    'pair_l7': {'tau': 64, 'leaves': 7, 'transductive': True, 'drop_pair': False, 'drop_neighbour': True, 'threads': 3},
    'pair_nbr': {'tau': 64, 'transductive': True, 'drop_pair': False, 'threads': 3},
    'pair_notrans': {'tau': 64, 'drop_pair': False, 'drop_neighbour': True, 'threads': 3},
    'trans_24_l7': {'tau': 24, 'leaves': 7, 'transductive': True},
}


def build(**params):
    params.pop('device', None)
    return GBDTArm(**params)
