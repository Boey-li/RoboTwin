#!/bin/bash
# Move 3 Control-B runs from preemptible overcap to rl2-lab WITHOUT losing progress.
#
# A running job cannot change partition, so this is cancel+resubmit. Doing that now
# would discard ~600 epochs, since save_freq=1000 and nothing is on disk yet. So we
# wait for policy_epoch_1000, then resubmit with RESUME_CKPT/START_EPOCH (added after
# the earlier preemption) so the new job picks up exactly where the old one stopped.
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
# jobid:task:arm
TARGETS="3724707:press_stapler:scene40k1 3724708:press_stapler:scene40k2 3724709:press_stapler:scene40k4"
for spec in $TARGETS; do
  JID=${spec%%:*}; rest=${spec#*:}; T=${rest%%:*}; ARM=${rest#*:}
  CK="ckpts/ACT/$T/demo_clean-$T-$ARM-ee-0/policy_epoch_1000_seed_0.ckpt"
  echo "[migrate] $JID $T/$ARM -- waiting for epoch-1000 checkpoint"
  for i in $(seq 1 240); do
    [ -f "$CK" ] && break
    timeout 30 squeue -h -j "$JID" >/dev/null 2>&1 || { echo "[migrate] $JID vanished; skipping"; break; }
    sleep 60
  done
  if [ ! -f "$CK" ]; then echo "[migrate] $JID: no ckpt after wait, leaving on overcap"; continue; fi
  echo "[migrate] $JID: ckpt present, cancelling and resuming on rl2-lab"
  timeout 30 scancel "$JID" 2>/dev/null; sleep 10
  RESUME_CKPT="$R/$CK" START_EPOCH=1000 \
  sbatch --partition=rl2-lab --account=rl2-lab --qos=short \
    --gpus-per-node="l40s:1" --exclude=dynamics \
    --job-name="cb_${T:0:2}_${ARM}" \
    sbatch/train_act.sh "$T" 0 demo_clean "$ARM" ee 2>&1 | tail -1
done
echo "[migrate] done"
