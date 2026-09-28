#!/bin/bash
# Wait for the seed scans, draw one common scene set per task, rerun all 8 evals on it.
# Selection uses the scripted expert only - no policy result feeds into it.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
LOG="$R/sbatch_logs/rerun_pinned.log"
RNG_SEED="${RNG_SEED:-20260821}"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "waiting for seed scans (sc_/sc2_)"
while :; do
  n=$(squeue -u "$USER" -h -o "%j" | grep -c '^sc2\?_') || n=0
  [ "$n" -eq 0 ] && break
  say "  $n scan job(s) still running"
  sleep 120
done
say "scans finished"

if ! "$R/.venv/bin/python" scripts/build_common_seeds.py --n 100 --rng-seed "$RNG_SEED" >> "$LOG" 2>&1; then
  say "ABORT: could not draw 100 common seeds for every task (see above)"
  exit 1
fi

# rl2-lab caps this account at 2 concurrent a40s; the rest start now on overcap.
i=0
for spec in \
  "adjust_bottle|aloha_agilex|demo_clean|joint|rp_ab_al_j" \
  "adjust_bottle|aloha_agilex|demo_clean|ee|rp_ab_al_e" \
  "adjust_bottle|arx_x5|demo_clean_arx|joint|rp_ab_ax_j" \
  "adjust_bottle|arx_x5|demo_clean_arx|ee|rp_ab_ax_e" \
  "grab_roller|aloha_agilex|demo_clean|joint|rp_gr_al_j" \
  "grab_roller|aloha_agilex|demo_clean|ee|rp_gr_al_e" \
  "grab_roller|arx_x5|demo_clean_arx|joint|rp_gr_ax_j" \
  "grab_roller|arx_x5|demo_clean_arx|ee|rp_gr_ax_e" ; do
  IFS='|' read -r TASK ENVCFG TASKCFG ACTION NAME <<< "$spec"
  if [ "$i" -lt 2 ]; then P=(--partition=rl2-lab); else P=(--partition=overcap); fi
  out=$(sbatch "${P[@]}" --account=rl2-lab --qos=short --job-name="$NAME" \
      sbatch/eval_act.sh "$TASK" 100 0 "$ENVCFG" "$TASKCFG" "$ACTION" \
      "$R/seed_sets/common__${TASK}.txt" 2>&1)
  say "  $NAME -> $out"
  i=$((i+1))
done
say "submitted 8 pinned-seed evals"
