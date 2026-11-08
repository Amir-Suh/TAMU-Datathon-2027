House Edge datathon, repo root = cwd. Read AGENTS.md, claude_output/notes/EVAL_PROTOCOL_FROZEN.md (frozen rules + decision logs) and claude_output/notes/session_log.md only; do not re-read old chats.
State: branch fredy/bakeoff-gbdt-seq (`git status -sb` for HEAD; uncommitted: round2.sh, round2_blend.py, moegate4 parity 1e-6, overnight scripts). Main closes Sun 10:30 CT; round two 10:30-12:00 (90 min).
Submitted: Andrew's clean-refit 3b. Candidate 4 = B in runs/entry_cand4_B (check_submission VALID, repro OK 246/233 s, OOF 57.283 vs 3b recipe 57.106); NOT uploaded: the user decides.
Running unattended since 03:45: overnight_chain.sh (round2 rehearsal -> moegate4 leak tests -> frozen MoE retest), then queue_epoch150.sh. First run: bash claude_output/sh/overnight_status.sh
Read first: rehearsal TOTAL time vs 90 min (runs/r2_rehearsal.log); its predictions should match cached ones (determinism); MoE result in claude_output/notes/cv_blend_moegate_results.txt; epoch150 in runs/epoch_tail150 (reading rule in queue_epoch150.sh).
Round-two plan: run packaged candidate 4 on the new data first (no-retrain fallback, minutes), then round2.sh (retrain), choose with the paired-bootstrap gate (>= 0.2 and interval excludes 0). Fixed candB_main blend weights.
Waiting on the user: upload B or keep 3b; confirm fixed blend weights for round two. Launch nothing long without a say-so.
Corrected, do not trust older notes: size x2 helps only slightly (matched +0.079); the transductive gap is data volume, not own-history; CE-80 is worse than 100.
