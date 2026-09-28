#!/bin/bash
# One-shot watcher for Phase E2 (2 cotrain-ee trainings). Exits — and so notifies —
# on EITHER outcome, so a crash is never mistaken for "still running":
#   DONE    both policy_last.ckpt written
# WARNING: do not create placeholder policy_last.ckpt files in these dirs while this
# watcher is armed -- a DRY_RUN probe did exactly that on 2026-08-26 and fired a false
# "E2 DONE". Test the F3 gate against a scratch copy instead.
#   STALLED a job's .err stopped advancing for >20 min (dead / preempted / hung)
R=/coc/flash7/bli678/Projects/RoboTwin
C=$R/XPolicyLab/policy/ACT/checkpoints
A=$C/place_burger_fries/demo_clean-place_burger_fries-cotrain4-ee-0/policy_last.ckpt
B=$C/press_stapler/demo_clean-press_stapler-cotrain4-ee-0/policy_last.ckpt
E1=$R/sbatch_logs/act_train_ct_pl_3720329.err
E2=$R/sbatch_logs/act_train_ct_pr_3720330.err
while true; do
  if [ -f "$A" ] && [ -f "$B" ]; then echo "E2 DONE: both cotrain-ee policy_last.ckpt written"; exit 0; fi
  now=$(date +%s)
  for f in "$E1" "$E2"; do
    [ -f "$f" ] || { echo "E2 STALLED: $f missing"; exit 1; }
    age=$(( now - $(stat -c %Y "$f") ))
    if [ "$age" -gt 1200 ]; then
      echo "E2 STALLED: $(basename "$f") idle ${age}s - $(tail -c 200 "$f" | tr '\r' '\n' | tail -1)"; exit 1
    fi
  done
  sleep 120
done
