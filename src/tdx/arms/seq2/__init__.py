from .arm import Seq2Arm

VARIANTS = {
    'base': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True},
    'aug3': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'aug': 0.3, 'aug_target': 'masked'},
    'aug3all': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'aug': 0.3, 'aug_target': 'all'},
    'aug5all': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'aug': 0.5, 'aug_target': 'all'},
    'flag_aug5all': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'short_flag': True, 'aug': 0.5, 'aug_target': 'all'},
    'flag_aug5all_dist': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'short_flag': True, 'aug': 0.5, 'aug_target': 'all', 'distill': 0.5},
    'flag_aug5all_fac': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'short_flag': True, 'aug': 0.5, 'aug_target': 'all', 'factor': True},
    'tokdrop2_flag_aug5all': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'tokdrop': 0.2, 'short_flag': True, 'aug': 0.5, 'aug_target': 'all'},
    'all_in': {'h': 128, 'msg': False, 'epochs': 100, 'transductive': True, 'short_flag': True, 'aug': 0.5, 'aug_target': 'all', 'distill': 0.5, 'factor': True, 'tokdrop': 0.1},
}


def build(**params):
    return Seq2Arm(**params)
