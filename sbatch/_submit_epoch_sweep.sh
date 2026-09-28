#!/bin/bash
# Submit the epoch sweep once Slurm is healthy again.
#
# The primary slurmctld (marvin) went down mid-session and sbatch hangs even via the
# backup, so this waits for the controller and then submits. Idempotent: a cell is
# skipped if its job is already queued or its result already exists, so a partial
# earlier attempt cannot produce duplicates.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
LOG="$R/sbatch_logs/epoch_sweep.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

SWEEP=(
  "grab_roller|arx_x5|demo_clean_arx|joint|demo_clean-grab_roller-arx_x5-joint-0"
  "adjust_bottle|aloha_agilex|demo_clean|ee|demo_clean-adjust_bottle-aloha_agilex-ee-0"
)
EPOCHS="1000 2000 3000 4000 5000"

say "waiting for slurmctld"
while :; do
  if timeout 30 scontrol ping 2>/dev/null | grep -q "is UP" && timeout 30 squeue -u "$USER" -h >/dev/null 2>&1; then
    # a real submission is the only proof it works; probe with --test-only
    if timeout 30 sbatch --test-only --partition=overcap --account=rl2-lab --qos=short \
         sbatch/eval_act.sh grab_roller 1 0 arx_x5 demo_clean_arx joint >/dev/null 2>&1; then
      break
    fi
  fi
  say "  slurm not accepting submissions yet"
  sleep 300
done
say "slurm healthy - submitting"

Q="$(timeout 60 squeue -u "$USER" -h -o '%j' 2>/dev/null)"
i=0
for spec in "${SWEEP[@]}"; do
  IFS='|' read -r TASK ENVCFG TASKCFG ACTION RUN <<< "$spec"
  short=$( [ "$TASK" = grab_roller ] && echo grj || echo abe )
  for EP in $EPOCHS; do
    NAME="ep_${short}${EP}"
    CK="${RUN}-ep${EP}"
    if echo "$Q" | grep -qx "$NAME"; then say "  $NAME already queued - skip"; continue; fi
    if ls eval_result/"$TASK"/ACT/"$TASKCFG"/"$CK"/*/_result.txt >/dev/null 2>&1; then
      say "  $NAME already has a result - skip"; continue; fi
    if [ "$i" -lt 2 ]; then P=--partition=rl2-lab; else P=--partition=overcap; fi
    out=$(timeout 120 sbatch $P --account=rl2-lab --qos=short --job-name="$NAME" \
        --export=ALL,CKPT_OVERRIDE="$CK" \
        sbatch/eval_act.sh "$TASK" 100 0 "$ENVCFG" "$TASKCFG" "$ACTION" \
        "$R/seed_sets/common__${TASK}.txt" 2>&1)
    say "  $NAME -> $out"
    i=$((i+1))
  done
done
say "epoch sweep submission finished"
