# Model 5: boosted win classifier

Model 5 replaces Model 4's win probability with a histogram gradient boosting classifier. Model 4 supplies its win logit, independent win logit, and posterior summaries; the classifier can change the probability in any way supported by those features. P(size | win) still comes from Model 4.

## Episode crossfit

Training uses eight pseudo cutoffs per training query. All machines and pseudo cutoffs from one episode stay together in every fold.

For each of four round-selection folds:

1. Fit Model 4 on the other episodes to produce held-fold features. This refits Model 4 priors and its selected hyperparameters.
2. Within those other episodes, make classifier training features with three episode folds. Each feature row comes from a Model 4 fit that excludes its episode and the outer held episodes.
3. Fit a classifier for each round choice `{25, 50, 100, 200}` on those crossfit training features and predict the outer held fold.

The chosen round count minimizes the combined selection-fold binary log-loss. The reported minimum is a **round-selection loss**, so it is optimistic as an estimate of a fully tuned model's future loss.

The final classifier is trained on crossfit Model 4 features for all training episodes. Its deployment Model 4 is fitted on all training episodes. Any supplied prefit base is accepted only when its data fingerprint matches that exact final training partition.

The model uses only the training split for fitting and selection. Current validation and test results are reported by the repaired comparison run; scores from the archived implementation do not apply to this fit.

## Repaired run

The fixed repair run scores **54.841763 / 60** on the previously used public validation set, versus M3 at **55.268370**. The selected classifier has **100 trees**; it remains worse than M3 on validation and held-episode training diagnostics. See the [repair results](../../results/repaired-m5-m6/README.md).
