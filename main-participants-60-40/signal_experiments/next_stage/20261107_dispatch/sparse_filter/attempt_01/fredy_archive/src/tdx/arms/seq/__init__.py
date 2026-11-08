from .arm import SeqArm

VARIANTS = {
    'gru128_nomsg': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True},
    'gru128_msg': {'h': 128, 'msg': True, 'epochs': 100, 'transductive': True},
    'gru192_nomsg': {'h': 192, 'msg': False, 'epochs': 100, 'transductive': True},
    'gru128_aug': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'aug': 0.3},
    'gru128_factor': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'factor': True},
}


def build(**params):
    return SeqArm(**params)
