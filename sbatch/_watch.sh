#!/bin/bash
# Between-ticks watcher: runs the supervisor every 5 min so a finished training
# job gets its eval submitted promptly instead of idling until the next hourly
# tick. Matters most for rl2-lab, whose 2-job cap means a freed slot is the
# scarcest resource we have - an hour of it idle is an hour lost.
#
# Safe to run alongside the cron tick: _supervise.sh takes an flock.
# Exits once every run in the table has a result, or after MAX_HOURS.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
MAX_HOURS="${MAX_HOURS:-14}"
DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 ))
N_RUNS=$(grep -cE '^  "[a-z_]+\|' sbatch/_supervise.sh)

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  out=$(bash sbatch/_supervise.sh 2>&1)
  printf '%s\n\n' "$out" >> sbatch_logs/watch.log
  done_n=$(grep -c "eval DONE" <<< "$out")
  if [ "$done_n" -ge "$N_RUNS" ]; then
    echo "[$(date)] all $N_RUNS runs have results - watcher exiting" >> sbatch_logs/watch.log
    exit 0
  fi
  sleep 300
done
echo "[$(date)] watcher hit MAX_HOURS=$MAX_HOURS" >> sbatch_logs/watch.log
