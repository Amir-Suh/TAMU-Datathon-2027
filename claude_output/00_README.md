# claude_output — start here

Prepared 2026-11-07 ~14:00 CT for House Edge (main closes 21:00 CT today; round two 21:00 → Sunday 11:00).

| File | What it is |
|---|---|
| `01_ADVERSARIAL_REVIEW.md` | Claim-by-claim check of `gemini_output/*` against the kit (17 rows, evidence per row) |
| `02_BAKEOFF_PLAN.md` | Data construction, five architectures, harness, decision flow, timeline, round-two design |
| `specs/COMMON_CONTRACT.md` + `W0…W6_*.md` | Codex stream specs (interfaces, rules, work packages, acceptance commands, model routing) |
| `sh/` | Setup, safe entry, Codex launchers (guarded), collect, review, reproduction check, round-two refit |
| `../AGENTS.md`, `../CLAUDE.md` | Light-weight repo contract for any coding agent |

## What the review changed in the plan (three bullets)
1. The cutoff targets are near-deterministic (445/768 validation queries have true `p_0` < 0.2 or > 0.8), so per-machine lag models are the primary bet, not noise filters. Gemini's "20% → 42% neighbour jump" and "win drops after a win" were not reproduced; lag effects differ per machine and flip sign.
2. Validation and test HISTORIES (424k extra rows) are legal self-supervised data; `signal` exists only at the cutoff, so it enters in a second stage; validation targets are used in the final fit and in the stacker (with grouped CV for an honest score).
3. Round two changes the lag structure and the organizer reruns offline to 1e-8: arms must refit in one command, and inference ships as numpy.

## Commands you can run yourself (in order)
```
cd /home/ganfwe/Downloads/TAMU-Datathon-2027
! bash claude_output/sh/00_setup_env.sh                 # ~5-10 min, creates ~/.venvs/tdx (Python 3.13, torch cu128). Do this first.
! bash claude_output/sh/01_link_data.sh                 # already run once; symlinks data/main and kit/public
! bash claude_output/sh/02_safe_entry.sh                # already run once: pooled_recent, validation 30.826. Upload outputs/safe/submission.csv + outputs/safe_entry.zip to the Main form NOW
! codex-guard --status                                  # plan window before launching
! bash claude_output/sh/10_launch_stream.sh W0 --dry-run
! bash claude_output/sh/11_launch_wave.sh w0             # starts W0 in tmux session tdx-w0 (attach: tmux attach -t tdx-w0)
# after W0 finishes (about 45 min): read /tmp/tdx_w0_orchestrator.txt, then
! bash claude_output/sh/13_review_stream.sh W0          # optional, different-model review
! bash claude_output/sh/12_collect_stream.sh W0
! bash claude_output/sh/11_launch_wave.sh pilots --dry-run
! bash claude_output/sh/11_launch_wave.sh pilots         # W1 glm, W2 gbdt, W3 seq, W4 regime, W6 package skeleton in parallel
# each: bash claude_output/sh/12_collect_stream.sh W1 ... W6, then
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
```
Already done in this session: data linked, safe entry built and checked (`check_submission.py`: VALID, 30.825854 / 60; ratios stable 0.958, short_history 1.001, drift 0.946, shared_change 0.880), all scripts pass `bash -n`, and every launcher's `--dry-run` prints its Codex command. Venv `~/.venvs/tdx` is built (python 3.13.15, numpy 2.5.4, lightgbm 4.7.0, torch 2.11.0+cu128, `requirements-dev.lock` written). Not done: any Codex launch (yours to start), and the guard-visibility check below.

## Things to know before launching
- **Guard visibility check (skill rule):** after the first stream is running for a few minutes, `codex-guard --status` must show `newest usage record: … (0 min ago)`; if it is old, stop and tell me (the guard is blind).
- **GPU (verified 2026-11-07):** host torch 2.11+cu128 sees the RTX 5080 (sm_120) and runs matmul and GRU backward; the Codex `workspace-write` sandbox hides `/dev/nvidia*` so CUDA is off there (`danger-full-access` works but is not used). W3 builds and CPU-smokes the model; you run the full sweep with `bash claude_output/sh/14_gpu_sweep.sh seq dev` after collecting W3. No model tokens are spent on GPU training.
- **Token policy (user, 2026-11-07):** Claude stays light; implementation, sweeps, reviews and refinement specs' code go to Codex.
- **Streams run in rsync copies** (`../TAMU-Datathon-2027-w<n>`), not git clones; nothing needs committing before launch. `12_collect_stream.sh` copies only each stream's owned paths back and prints any file outside them.
- **Cost:** token use per stream is UNKNOWN (no measurements on this repo). Check `codex-guard --status` between waves; the guard stops a run at 95% of the plan window.
- **Models** are routed per `sh/streams.sh` (all slugs verified with `codex debug models` today). Override one launch with `CODEX_MODEL=… CODEX_REASONING_EFFORT=…`.
- **No git commits were made.** After waves, ask me for a commit plan (explicit paths) and I will use `meaningful-commits`.

## Decisions I made that you can veto
- Streams are copies, not worktrees, to avoid committing before launch.
- Pilot grids are capped (≤ 12 variants per arm, 5–12 min each); the analysis rule is PROPOSED (paired episode bootstrap) and any agent may replace it with its own written `runs/READING_RULE.md`.
- `seq` and `gbdt` are kept in the bake-off even though the evidence favours `glm`: the pilots are cheap and the stack benefits from different errors.
- Simulator reverse-engineering, Time-LLM, JEPA and distillation are parked (reasons in the review).

## Effort level
High effort was used for the review and planning. Switch back to **medium** now: everything that remains in this session (running launchers, reading `compare` tables, writing refinement specs) is mechanical. Switch to high again only for the post-pilot decision (choosing arms and the blend) if the table is close.
