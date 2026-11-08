from .arm import MoeGate4Arm, forward_np, compute_pair_stats

VARIANTS = {
    'pilot': {
        'K': 4,
        'lam_cond': 0.3,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    },
    'k3_ref': {
        'K': 3,
        'lam_cond': 0.0,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    },
    'k4_nocond': {
        'K': 4,
        'lam_cond': 0.0,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    },
    'k4_cond03': {
        'K': 4,
        'lam_cond': 0.3,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    },
    'k4_cond10': {
        'K': 4,
        'lam_cond': 1.0,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    },
}

_last_arm = None


def build(**params):
    global _last_arm
    arm = MoeGate4Arm(**params)
    _last_arm = arm
    return arm


def usage(split, arm=None):
    """Return mean gate weights per expert for each episode (shape: [E, K])."""
    global _last_arm
    if arm is None:
        arm = _last_arm
    if arm is None:
        raise RuntimeError("No arm instance provided or built yet. Pass an arm or call build/fit first.")
    return arm.usage(split)
