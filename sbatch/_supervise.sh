#!/bin/bash
# Hourly supervisor for the ACT clean2clean reproduction.
#
# For each (task, embodiment, action space) it reports training progress and, the
# moment a run finishes, submits the 100-trial rollout. "Finished" means
# policy_last.ckpt exists: detr/act_policy.py:140 hard-loads that exact filename,
# so intermediate policy_epoch_*.ckpt files are never what eval reads.
#
# Partition policy: submit to rl2-lab; pass FORCE_OVERCAP=1 to go straight to
# overcap (preemptible, much larger a40 pool) when rl2-lab is saturated.
#
#   bash sbatch/_supervise.sh              # report + submit anything ready
#   DRY_RUN=1 bash sbatch/_supervise.sh    # report only
set -uo pipefail

R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
DRY_RUN="${DRY_RUN:-0}"

# The hourly cron tick and the between-ticks watcher can land together. Both
# submit, and squeue lags a few seconds behind sbatch, so without a lock the
# duplicate-guard below can read a stale queue and submit the same eval twice.
if [ "$DRY_RUN" != "1" ]; then
  exec 9>"$R/.supervise.lock"
  flock -w 300 9 || { echo "another supervise run holds the lock; skipping"; exit 0; }
fi

# task | env_cfg_type | task_config | action_type | eval job-name
RUNS=(
  "adjust_bottle|aloha_agilex|demo_clean|joint|ev_adjbot"
  "grab_roller|aloha_agilex|demo_clean|joint|ev_roller"
  "adjust_bottle|arx_x5|demo_clean_arx|joint|ev_arxbot"
  "grab_roller|arx_x5|demo_clean_arx|joint|ev_arxrol"
  "adjust_bottle|aloha_agilex|demo_clean|ee|ev_eebot"
  "grab_roller|aloha_agilex|demo_clean|ee|ev_eerol"
  "adjust_bottle|arx_x5|demo_clean_arx|ee|ev_eearxbot"
  "grab_roller|arx_x5|demo_clean_arx|ee|ev_eearxrol"
)

QUEUE="$(squeue -u "$USER" -h -o '%i|%j|%T|%P|%R' 2>/dev/null)"

job_state() { echo "$QUEUE" | awk -F'|' -v n="^$1$" '$2 ~ n {print $1" "$3" "$4" "$5; exit}'; }

submit() {  # $1=jobname, rest = eval_act.sh args
  local name="$1"; shift
  local part=(--partition=rl2-lab --account=rl2-lab --qos=short)
  [ "${FORCE_OVERCAP:-0}" = "1" ] && part=(--partition=overcap --account=rl2-lab --qos=short)
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY_RUN: sbatch ${part[*]} --job-name=$name sbatch/eval_act.sh $*"; return
  fi
  sbatch "${part[@]}" --job-name="$name" sbatch/eval_act.sh "$@"
}

# Match a training .out header to this run. Logs written before action_type was
# threaded through carry no "action=" token; treat those as joint.
train_job_for() {  # $1=task $2=envcfg $3=action -> jobid
  local f
  for f in $(ls -t sbatch_logs/act_train_*.out 2>/dev/null); do
    awk -v t="task=$1" -v e="env=$2" -v a="action=$3" '
      index($0,t) && index($0,e) {
        if (index($0,"action=")==0) { if (a=="action=joint") { print "1"; exit } }
        else if (index($0,a)) { print "1"; exit }
      }' "$f" | grep -q 1 || continue
    local j="${f##*_}"; echo "${j%.out}"; return
  done
}

echo "=== $(date '+%F %H:%M') ==="
[ -n "$QUEUE" ] && { echo "queue:"; echo "$QUEUE" | sed 's/^/  /'; } || echo "queue: empty"
echo

for spec in "${RUNS[@]}"; do
  IFS='|' read -r TASK ENVCFG TASKCFG ACTION EVNAME <<< "$spec"
  CKPT="demo_clean-${TASK}-${ENVCFG}-${ACTION}-0"
  DIR="ckpts/ACT/$CKPT"
  echo "[$TASK / $ENVCFG / $ACTION]"

  RES=$(ls -1 eval_result/"$TASK"/ACT/"$TASKCFG"/"$CKPT"/*/_result.txt 2>/dev/null | tail -1)
  if [ -n "$RES" ]; then echo "    eval DONE -> $(tail -1 "$RES")"; continue; fi

  EV=$(job_state "$EVNAME")
  if [ -n "$EV" ]; then
    # rl2-lab caps this account at 2 concurrent a40s, so a third eval sits on
    # QOSGrpGRES until one finishes - potentially hours. overcap is preemptible
    # but has a far larger pool, and starting now beats waiting. Prefer rl2-lab
    # at submit time; demote only once the queue says it cannot run there.
    read -r EVID EVST EVPART EVREASON <<< "$EV"
    if [ "$EVST" = PENDING ] && [ "$EVPART" = rl2-lab ] \
       && [[ "$EVREASON" =~ QOSGrp|Resources|Priority ]] && [ "$DRY_RUN" != "1" ]; then
      echo "    eval $EVID stuck on rl2-lab ($EVREASON) -> moving to overcap"
      scancel "$EVID"
      FORCE_OVERCAP=1 submit "$EVNAME" "$TASK" 100 0 "$ENVCFG" "$TASKCFG" "$ACTION"
    else
      echo "    eval in queue: $EV"
    fi
    continue
  fi

  if [ -f "$DIR/policy_last.ckpt" ]; then
    echo "    training COMPLETE -> submitting eval"
    submit "$EVNAME" "$TASK" 100 0 "$ENVCFG" "$TASKCFG" "$ACTION"
    continue
  fi

  J=$(train_job_for "$TASK" "$ENVCFG" "$ACTION")
  if [ -z "$J" ]; then echo "    NOT STARTED (no training job found)"; continue; fi
  PROG=$(tr '\r' '\n' < "sbatch_logs/act_train_"*"_${J}.err" 2>/dev/null | grep -o '[0-9]*/[0-9]* \[[^]]*\]' | tail -1)
  ST=$(echo "$QUEUE" | awk -F'|' -v j="$J" '$1==j {print $3"/"$4; exit}')
  latest=$(ls -1 "$DIR"/policy_epoch_*.ckpt 2>/dev/null | tail -1)
  echo "    train job $J ${ST:-not-in-queue} ${PROG:-(no progress line)}${latest:+ | last ckpt: $(basename "$latest")}"
done
