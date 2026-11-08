"""Train shallow regularized corrections to point-in-time adaptive win odds."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeRegressor

from .causal import feature_rows
from .peer_lags import adaptive_trace, adjust_win

SEED = 314159


@dataclass(frozen=True)
class BoostInputs:
    features: np.ndarray
    offset: np.ndarray
    base: np.ndarray


def build_inputs(queries, spins, machines, ticks, model, *, local_features=None):
    """Use the forecast and all history features as they stood at each endpoint."""
    if local_features is None:
        local_features = feature_rows(queries, spins, machines, ticks, peers=False)
    trace = adaptive_trace(queries, spins, model)
    base = trace.base[machines, ticks]
    win = 1 - base[:, 0]
    no_lag = trace.no_lag_win[machines, ticks]
    offset = logit(np.clip(win, 1e-12, 1 - 1e-12))
    no_lag_logit = logit(np.clip(no_lag, 1e-12, 1 - 1e-12))
    # The existing local features include differences between short and long payout rates.
    features = np.column_stack([local_features, win, no_lag, offset - no_lag_logit,
                                np.log1p(trace.context_support[machines, ticks]),
                                np.log1p(trace.no_lag_support[machines, ticks])])
    return BoostInputs(features, offset, base)


class OffsetCalibration:
    """Ridge-regularized intercept and slope correction fitted with binary log loss."""

    def __init__(self, *, ridge=.01):
        self.ridge = ridge

    def fit(self, offset, outcomes, weights):
        design = np.column_stack([np.ones(len(offset)), offset])
        weights = weights / weights.sum()

        def objective(beta):
            logits = offset + design @ beta
            loss = weights @ (np.logaddexp(0, logits) - outcomes * logits) + .5 * self.ridge * (beta @ beta)
            gradient = design.T @ (weights * (expit(logits) - outcomes)) + self.ridge * beta
            return loss, gradient

        result = minimize(objective, np.zeros(2), method="L-BFGS-B", jac=True,
                          options={"ftol": 1e-12, "gtol": 1e-8})
        if not result.success:
            raise RuntimeError(f"Offset calibration fit failed: {result.message}")
        self.coefficients = result.x
        return self

    def correction(self, offset):
        return self.coefficients[0] + self.coefficients[1] * offset


class OffsetTrees:
    """Depth-limited trees fit to weighted Newton steps from the supplied offset."""

    def __init__(self, *, rounds=60, max_depth=2, min_samples_leaf=100,
                 shrinkage=.05, step_limit=1., curvature_floor=1e-3, seed=SEED):
        self.rounds = rounds
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage = shrinkage
        self.step_limit = step_limit
        self.curvature_floor = curvature_floor
        self.seed = seed

    def fit(self, features, offset, outcomes, weights):
        logits = offset.copy()
        self.trees = []
        for i in range(self.rounds):
            probability = expit(logits)
            curvature = np.maximum(probability * (1 - probability), self.curvature_floor)
            target = (outcomes - probability) / curvature
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         random_state=self.seed + i)
            tree.fit(features, target, sample_weight=weights * curvature)
            logits += self.shrinkage * np.clip(tree.predict(features), -self.step_limit, self.step_limit)
            self.trees.append(tree)
        return self

    def corrections(self, features, budgets):
        """Return the additive log-odds changes at the requested tree budgets."""
        budgets = set(budgets)
        if not budgets or min(budgets) < 1 or max(budgets) > len(self.trees):
            raise ValueError("Tree budgets must refer to fitted rounds.")
        steps = np.zeros(len(features))
        predictions = {}
        for i, tree in enumerate(self.trees[:max(budgets)], 1):
            steps += self.shrinkage * np.clip(tree.predict(features), -self.step_limit, self.step_limit)
            if i in budgets:
                predictions[i] = steps.copy()
        return predictions


def corrected_probabilities(inputs, calibration, trees, *, budgets=(30, 60)):
    predictions = {"zero_correction": inputs.base.copy(),
                   "calibration": adjust_win(inputs.base, calibration.correction(inputs.offset))}
    if trees is not None:
        for rounds, correction in trees.corrections(inputs.features, budgets).items():
            predictions[f"trees_{rounds}"] = adjust_win(inputs.base, correction)
    return predictions
