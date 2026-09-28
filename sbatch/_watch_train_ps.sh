#!/bin/bash
# Fires when the press_stapler single-robot N=160 training finishes, or stalls.
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
CK="ckpts/ACT/press_stapler/demo_clean_n200-press_stapler-aloha_agilex-ee-0/policy_last.ckpt"
while true; do
  [ -f "$CK" ] && { echo "TRAIN DONE: policy_last.ckpt written"; exit 0; }
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    grep -q '^n200_ps' <<<"$q" || { echo "TRAIN STALLED: n200_ps gone, no policy_last.ckpt"; exit 1; }
  fi
  sleep 300
done
