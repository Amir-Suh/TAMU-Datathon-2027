"""Two-state emissions, scaled forward-backward likelihood, and joint forecasts."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

STATE_SIGN = np.array([-.5, .5])
PRIOR_SD = np.r_[np.full(3, .35), np.full(3, 1.0), np.full(3, .75), .50]


@dataclass
class History:
    payout: np.ndarray
    kind: np.ndarray
    cutoff: np.ndarray
    base_logit: np.ndarray
    count: np.ndarray

    @classmethod
    def build(cls, split, base_win):
        before = np.arange(split.X.shape[1])[None, :] < split.cutoff[:, None]
        payout = np.where(before, np.minimum(split.X, 2), -1).T
        return cls(payout, split.machine_type, split.cutoff, logit(base_win).T,
                   (payout >= 0).sum(axis=0))


def parameters(theta):
    intercept, midpoint, log_gap = theta[:3], theta[3:6], theta[6:9]
    gap = np.exp(log_gap)
    q1 = expit(midpoint[:, None] + gap[:, None] * STATE_SIGN)
    return intercept, q1, gap, theta[9]


def emissions(history, theta):
    intercept, q1, _, shift = parameters(theta)
    win = expit(history.base_logit[..., None] + intercept[history.kind, None] + shift * STATE_SIGN)
    p1 = q1[history.kind]
    payout = history.payout[..., None]
    prob = np.where(payout == 0, 1 - win, np.where(payout == 1, win * p1, win * (1 - p1)))
    return np.where(payout < 0, 1.0, prob), win, p1


def advance(state, switch):
    return (1 - switch) * state + switch * state[..., ::-1]


def forward(emission, switch):
    """Return the prior at each tick, the filtered state, and each likelihood factor."""
    prior = np.empty_like(emission)
    filtered = np.empty_like(emission)
    scale = np.empty(emission.shape[:2])
    state = np.full(emission.shape[1:], .5)
    for tick, evidence in enumerate(emission):
        prior[tick] = state
        weighted = state * evidence
        scale[tick] = weighted.sum(axis=1)
        filtered[tick] = weighted / scale[tick, :, None]
        state = advance(filtered[tick], switch)
    return prior, filtered, scale


def smoothed(emission, filtered, scale, switch):
    post = np.empty_like(filtered)
    back = np.ones(filtered.shape[1:])
    for tick in range(len(filtered) - 1, -1, -1):
        post[tick] = filtered[tick] * back
        back = advance(back * emission[tick], switch) / scale[tick, :, None]
    return post


def history_prior(history):
    one_rate = []
    for kind in range(3):
        outcomes = history.payout[:, history.kind == kind]
        one_rate.append(((outcomes == 1).sum() + .5) / ((outcomes > 0).sum() + 1))
    return np.r_[np.zeros(3), logit(one_rate), np.zeros(4)]


def objective(theta, history, switch, prior_mean):
    likelihood, win, q1 = emissions(history, theta)
    _, filtered, scale = forward(likelihood, switch)
    post = smoothed(likelihood, filtered, scale, switch)
    observed = history.payout[..., None] >= 0
    won = history.payout[..., None] > 0
    one = history.payout[..., None] == 1
    win_grad = (post * (win - won) * observed).sum(axis=0)
    size_grad = (post * (q1 - one) * won).sum(axis=0)
    _, _, gap, _ = parameters(theta)
    def aggregate(x):
        return np.bincount(history.kind, weights=x, minlength=3)

    grad = np.r_[
        aggregate(win_grad.sum(axis=1)),
        aggregate(size_grad.sum(axis=1)),
        gap * aggregate(size_grad @ STATE_SIGN),
        (win_grad @ STATE_SIGN).sum(),
    ]
    penalty = (theta - prior_mean) / PRIOR_SD
    loss = -np.log(scale).sum() + .5 * (penalty @ penalty)
    return loss, grad + penalty / PRIOR_SD


def fit(history, dwell, joint):
    switch = 0 if dwell is None else 1 / dwell
    bounds = [(-2, 2)] * 3 + [(-5, 5)] * 3 + [(-4, 2)] * 3 + [(-2, 2) if joint else (0, 0)]
    prior_mean = history_prior(history)
    result = minimize(objective, prior_mean, args=(history, switch, prior_mean), jac=True,
                      method="L-BFGS-B", bounds=bounds,
                      options={"ftol": 1e-10, "gtol": 1e-4, "maxiter": 100})
    if not result.success:
        raise RuntimeError(f"State emission fit failed for dwell={dwell}, joint={joint}: {result.message}")
    return result.x, result.nit


def infer(history, theta, dwell):
    switch = 0 if dwell is None else 1 / dwell
    emission, _, _ = emissions(history, theta)
    prior, _, scale = forward(emission, switch)
    return prior[history.cutoff, np.arange(len(history.cutoff))], -np.log(scale).sum(axis=0)


def joint_forecast(base_win, prior, conditional_one, shift, tail):
    """Construct each state's joint distribution before averaging."""
    win = expit(logit(base_win)[:, None] + shift * STATE_SIGN)
    win_and_one = win * conditional_one
    win_and_tail = win * (1 - conditional_one)
    return np.column_stack(((prior * (1 - win)).sum(axis=1), (prior * win_and_one).sum(axis=1),
                            (prior * win_and_tail).sum(axis=1)[:, None] * tail))
