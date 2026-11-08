# W1 independent numerical review

Implementer: `gpt-6.1-sol`, high. Reviewer: `gpt-6-astra`, high, read-only.
Both were selected explicitly with `collaboration.spawn_agent(model=..., reasoning_effort=..., fork_turns="none")`.

Independent sparse-history feature reference differed by at most 2.22e-16.
Finite-difference weighted soft-target objective gradient L2 error was 3.34e-6 using forward differences; the repository unit test uses central differences and tighter component tolerances.
Corrupting all storage at/after cutoff left pooled fitting and local predictions exactly unchanged (maximum absolute difference 0.0).
An opposite-sign lag-2 test with neutral pooled weights recovered the probability side for 32/32 machines; mean absolute error .030583, maximum .092005.

The review found two iteration-limit problems before any successful pilot:

| Probe | Old setting | Evidence | Resolution |
|---|---|---|---|
| Pooled, first 8 train episodes | 150 iterations | Loss 7656.2350; gradient maximum 17.7920; unconverged | Cap raised to 1500; sample converged at 489 iterations, loss 7654.7244 |
| Local lambda=1, validation | 40 iterations | 644/768 unsuccessful; raising cap changed probabilities by up to .010420 | Cap raised to 150; all 768 converged |
| Local lambda=1, test | 40 iterations | 1302/1536 unsuccessful; raising cap changed probabilities by up to .008075 | Cap raised to 150; all 1536 converged, at most 119 iterations |

Local convergence probes used pooled weights fitted on the first eight training episodes, not the complete pilot checkpoints. Lambda=10 probes were essentially unchanged by the increased local cap (one validation status failure at 40 iterations, no prediction change; no test failures).

The implementer then checked the full 96-episode training fit: 47.980 seconds on two BLAS threads; scipy success, 1150 iterations, objective 91808.8689274637, gradient maximum 2.479780907180862. Termination was relative objective tolerance, not the gradient threshold. Pooled status, objective, iteration count, gradient maximum and message are persisted in model.npz.

Parent review also found and resolved two integration details: build now consumes the shared sweep's device/smoke hints, and disabling EMA rates retains feature group 5 (last-20 observed win rate). No test expectation was weakened; a sweep-hint regression test was added.

After the final local-cap change: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ~/.venvs/tdx/bin/python -m pytest -q tests/glm` printed `7 passed in 0.50s`.
