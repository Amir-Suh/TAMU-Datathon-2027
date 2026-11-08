from .arm import MoeGateArm, forward_np, compute_pair_stats

VARIANTS = {
    'pilot': {
        'K': 3,
        'h': 128,
        'd_expert': 64,
        'epochs': 100,
        'transductive': True,
        'factor': True,
    }
}

_last_arm = None


def build(**params):
    global _last_arm
    arm = MoeGateArm(**params)
    _last_arm = arm
    return arm


def usage(split, arm=None):
    """Return mean gate weights per expert for each episode."""
    global _last_arm
    if arm is None:
        arm = _last_arm
    if arm is None:
        raise RuntimeError("No arm instance provided or built yet. Pass an arm or call build/fit first.")
    return arm.usage(split)
