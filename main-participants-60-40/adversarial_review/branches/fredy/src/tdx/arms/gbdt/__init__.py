from .arm import GBDTArm

VARIANTS = {
    'all_inf_l15': {},
    'all_64_l15': {'tau': 64},
    'all_24_l15': {'tau': 24},
    'last64_24_l15': {'tau': 24, 'last_n': 64},
    'last128_64_l15': {'tau': 64, 'last_n': 128},
    'all_64_l7': {'tau': 64, 'leaves': 7},
    'trans_64_l15': {'tau': 64, 'transductive': True},
    'trans_24_l7': {'tau': 24, 'leaves': 7, 'transductive': True},
}


def build(**params):
    params.pop('device', None)
    return GBDTArm(**params)
