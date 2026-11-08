# AGENTS.md — House Edge (TAMU Datathon 2026), repo TAMU-Datathon-2027

Fast-paced datathon repo. Plans live in `claude_output/` (start with `claude_output/00_README.md`). This file is the contract for every coding agent (Codex, Claude Code, Gemini).

## Task
Predict next-spin probabilities `[p_0, p_1, p_2, p_5, p_20]` for each query (one machine at its `cutoff` tick). Score = squared error against the simulator's TRUE probabilities, averaged per episode, then per condition (stable, short_history, drift, shared_change), divided by the frozen pooled baseline's error, `R = mean ratio`, points = `60/(1+R)` (main) or `40/(1+R)` (adaptation). Pooled baseline = 30.000 points on main validation. Lower error is better, nothing else counts.

## Deadlines (Central)
- UPDATED by organizers on 2026-11-07 ~20:40 (user-reported): main submission now closes **Sun 2026-11-08 10:30**. Round two (adaptation) kit opens **Sun 10:30** and closes **Sun 12:00** (90 minutes, fit + package + submit). Older notes below that say Sat 21:00 / Sun 11:00 are superseded.
- The latest Form response counts even if invalid: always run `kit/check_submission.py` and a fresh-folder reproduction before uploading.

## Facts that decide designs (all measured on the kit, see `claude_output/01_ADVERSARIAL_REVIEW.md`)
- History files contain only ticks `< cutoff`; the last tick is always `cutoff-1`. One cutoff per episode; 8 machines per episode.
- `signal` exists ONLY in `queries.csv` (value at the cutoff). Tick-level training rows built from history have no signal. Use it only in cutoff-level stages.
- `short_history` episodes have exactly 64 observed ticks per machine (40 consecutive recent + 24 sampled older). Detect with `n_obs == 64`.
- Train: 96 episodes, 768 labelled queries (only 12 jackpots), 139,944 history rows. Validation: 96 episodes, 768 TRUE probability vectors, 141,392 history rows. Test: 192 episodes, 1,536 queries, 282,664 history rows. 24 / 24 / 48 episodes per condition.
- Effective sample size for regime effects is the number of EPISODES, not ticks. Always split by whole episode.
- Using test/validation HISTORY (ticks < cutoff) as self-supervised next-tick training data is allowed and is the largest data source. Using test answers is not (none exist locally anyway).
- Lag effects are machine-specific and can flip sign (measured: 46–63% of machines show a positive "won last tick" effect depending on type). Do not hard-code pooled lag rules; learn them per machine/episode with shrinkage.
- Organizer rerun: Python 3.13, 2 CPU cores, 16 GiB, 30 min, offline, `/data` read-only, output must match the uploaded CSV within 1e-8. Pin extra packages in `requirements.txt`.

## How we work here (deliberately light)
- Pilot first, read outputs after. Run the experiment, write numbers to `runs/<arm>/<variant>/`, append one line to `runs/LOG.md` (the harness does this), compare afterwards with `tools/compare.py`. No pre-run reading rules, no approval gates, no questions unless blocked.
- You may write your own reading rule: after a pilot wave put a 10-line `runs/READING_RULE.md` (what you compared, which interval you used, what you concluded) before acting on the numbers. Default PROPOSED rule: paired bootstrap over validation episodes of the per-condition loss-ratio difference vs the incumbent; treat differences inside the 95% interval as ties and prefer the simpler or faster model.
- Only three checks are mandatory: (1) `python -m tdx.tools.leak_test` passes for any new arm, (2) `python kit/check_submission.py <csv>` on anything you propose to upload, (3) fresh-folder reproduction (`claude_output/sh/repro_check.sh`) before the final upload of each round.
- Build all dependence on the round through `--round`/`--data` arguments, never hard-coded paths. Round two must be a one-command refit.
- Each arm lives in `src/tdx/arms/<name>/` and implements `build(**variant) -> Arm` (see `claude_output/specs/COMMON_CONTRACT.md`). Do not edit another arm's directory.
- Stay inside your clone, keep stdout short, write durable results under `runs/` and notes under `claude_output/notes/`. Do not commit (`.git` is read-only in the sandbox); end with a short list of changed paths.
- Environment: `~/.venvs/tdx` (created by `claude_output/sh/00_setup_env.sh`), Python 3.13 to match the organizer image. Nothing installs inside the Codex sandbox. Data: `data/main` and `data/adaptation` are symlinks to the kit `public/` folders (git-ignored).
