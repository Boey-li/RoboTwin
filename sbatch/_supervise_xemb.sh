#!/bin/bash
# Supervisor for the cross-embodiment ACT study (plan: wiggly-cooking-moon.md).
#
# Reports the Progress table and chains phases: C -> E1 -> F1/F2, D -> E2 -> F3.
# Phase gates are hard: it will not advance past a gate whose verification failed.
# Placement: rl2-lab first (2-job a40 cap), overflow to overcap (a40 or l40s).
#   bash sbatch/_supervise_xemb.sh            report + submit what is ready
#   DRY_RUN=1 bash sbatch/_supervise_xemb.sh  report only
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
DRY_RUN="${DRY_RUN:-0}"
LOG="$R/sbatch_logs/xemb_supervise.log"
TASKS="place_burger_fries press_stapler"
EMB="aloha_agilex arx_x5 franka piper ur5"
ACTS="joint ee"
declare -A CFG=( [aloha_agilex]=demo_clean [arx_x5]=demo_clean_arx [franka]=demo_clean_franka
                 [piper]=demo_clean_piper [ur5]=demo_clean_ur5 )
declare -A SHARDS=( [place_burger_fries]=2 [press_stapler]=2 )

if [ "$DRY_RUN" != "1" ]; then
  exec 9>"$R/.xemb.lock"; flock -w 300 9 || { echo "another supervise run holds the lock"; exit 0; }
fi
# A failed/timed-out squeue returns empty output, which is indistinguishable from
# an idle queue -- and every gate below would then read "nothing running" and
# resubmit live work. Fail closed: retry, and abort the tick if we never get a
# trustworthy answer. (Seen 2026-08-26: controller slow, 2 healthy jobs invisible.)
Q=""; QOK=0
for _try in 1 2 3; do
  if Q="$(timeout 45 squeue -u "$USER" -h -o '%i|%j|%T|%P|%R' 2>/dev/null)"; then QOK=1; break; fi
  sleep 5
done
if [ "$QOK" != "1" ] && [ "$DRY_RUN" != "1" ]; then
  echo "ABORT tick: squeue unavailable after 3 tries - refusing to act on unknown queue state"
  [ "$DRY_RUN" = "1" ] || echo "[$(date '+%m-%d %H:%M')] ABORT tick: squeue unavailable" >> "$LOG"
  exit 0
fi
inq() { echo "$Q" | awk -F'|' -v n="^$1$" '$2 ~ n {print $1; exit}'; }
nq()  { echo "$Q" | grep -c "|$1" ; }

# rl2-lab holds 2 a40s for this account; everything else goes to overcap.
sub() { # sub <jobname> <script> <args...>
  local name="$1"; shift
  local used; used=$(echo "$Q" | awk -F'|' '$4=="rl2-lab"' | wc -l)
  local P=(--partition=overcap); [ "$used" -lt 2 ] && P=(--partition=rl2-lab)
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: sbatch ${P[*]} --job-name=$name $*"; return; fi
  sbatch "${P[@]}" --account=rl2-lab --qos=short --job-name="$name" "$@" >>"$LOG" 2>&1
  Q="$Q"$'\n'"0|$name|PENDING|${P[0]#--partition=}|new"
}
say() { echo "$@"; [ "$DRY_RUN" = "1" ] || echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

echo "=== $(date '+%F %H:%M')  cross-embodiment study ==="
[ -n "$Q" ] && echo "queue: $(echo "$Q" | wc -l) jobs" || echo "queue: empty"

## ---- Phase B: seed scans -> common set -------------------------------------
scan_running=$(nq "sc_" || true)
nseedset=0; for t in $TASKS; do [ -f "seed_sets/common__$t.txt" ] && nseedset=$((nseedset+1)); done
if [ "$scan_running" -gt 0 ]; then
  echo "B  scans: $scan_running running"
elif [ "$nseedset" -lt 2 ]; then
  echo "B  scans done -> drawing common seed sets"
  [ "$DRY_RUN" = "1" ] || "$R/.venv/bin/python" scripts/build_common_seeds.py --n 100 --rng-seed 20260826 || \
    { say "B  ABORT: could not draw 100 common seeds (widen scan range)"; exit 1; }
  nseedset=$(ls -1 seed_sets/common__*.txt 2>/dev/null | wc -l)
fi
SEEDS_READY=0
[ "$nseedset" -ge 2 ] && { SEEDS_READY=1; echo "B  seed sets: ready"; } || echo "B  seed sets: not ready (gates F only, not E1)"
if [ "$SEEDS_READY" = 1 ]; then
  for t in $TASKS; do
    [ -f "seed_sets/shard__${t}__0of${SHARDS[$t]}.txt" ] || \
      { [ "$DRY_RUN" = "1" ] || python3 scripts/_make_shards.py "$t" "${SHARDS[$t]}" >/dev/null; }
  done
fi

## ---- Phase C gate: all 20 TASK_CONFIGS keys --------------------------------
nkeys=$("$R/.venv/bin/python" - <<'PY'
import json
c=json.load(open("XPolicyLab/policy/ACT/TASK_CONFIGS.json"))
want=[f"demo_clean-{t}-{e}-{a}" for t in ("place_burger_fries","press_stapler")
      for e in ("aloha_agilex","arx_x5","franka","piper","ur5") for a in ("joint","ee")]
print(sum(1 for k in want if k in c))
PY
)
echo "C  datasets registered: ${nkeys}/20"
[ "$nkeys" -lt 20 ] && { echo "   gate closed - waiting on conversion"; exit 0; }

## ---- Phase E1: single-task training ----------------------------------------
e1_done=0; e1_todo=()
for t in $TASKS; do for e in $EMB; do for a in $ACTS; do
  ck="demo_clean-$t-$e-$a-0"; name="tr_${t:0:2}${e:0:2}${a:0:1}"
  if [ -f "ckpts/ACT/$t/$ck/policy_last.ckpt" ]; then e1_done=$((e1_done+1))
  elif [ -n "$(inq "$name")" ]; then :
  else e1_todo+=("$name|$t|$e|$a"); fi
done; done; done
echo "E1 single-task training: $e1_done/20 done, ${#e1_todo[@]} to submit"
for spec in "${e1_todo[@]}"; do IFS='|' read -r n t e a <<< "$spec"
  sub "$n" sbatch/train_act.sh "$t" 0 demo_clean "$e" "$a"; done

## ---- Phase F1/F2: evals (need E1) ------------------------------------------
# Evals are gated per-checkpoint, not on all 20: each F1 run needs only its own model,
# and each F2 run only its source model. Waiting for the whole cohort would idle the
# earliest finishers for about an hour.
if [ "$SEEDS_READY" = 1 ]; then
  for t in $TASKS; do S=${SHARDS[$t]}
    for e in $EMB; do for a in $ACTS; do
      for i in $(seq 0 $((S-1))); do
        name="f1_${t:0:2}${e:0:2}${a:0:1}$i"
        [ -f "ckpts/ACT/$t/demo_clean-$t-$e-$a-0/policy_last.ckpt" ] || continue
        d="eval_result/$t/ACT/${CFG[$e]}/demo_clean-$t-$e-$a-0"
        ls -1 $d/*/_result.txt >/dev/null 2>&1 && continue
        [ -n "$(inq "$name")" ] && continue
        sub "$name" sbatch/eval_act.sh "$t" $((100/S)) 0 "$e" "${CFG[$e]}" "$a" \
            "$R/seed_sets/shard__${t}__${i}of${S}.txt"
      done; done; done
    # F2: transfer into piper. joint only from 14-dim sources (franka is 16-dim).
    for src in aloha_agilex arx_x5 franka ur5; do for a in $ACTS; do
      [ "$src" = franka ] && [ "$a" = joint ] && continue
      for i in $(seq 0 $((S-1))); do
        name="f2_${t:0:2}${src:0:2}${a:0:1}$i"
        [ -f "ckpts/ACT/$t/demo_clean-$t-$src-$a-0/policy_last.ckpt" ] || continue
        [ -n "$(inq "$name")" ] && continue
        ls -1 eval_result/$t/ACT/demo_clean_piper/demo_clean-$t-$src-$a-0/*/_result.txt >/dev/null 2>&1 && continue
        sub "$name" sbatch/eval_act.sh "$t" $((100/S)) 0 "$src" demo_clean_piper "$a" \
            "$R/seed_sets/shard__${t}__${i}of${S}.txt"
      done; done; done
  done
else
  echo "F1/F2 waiting on seed sets (ready=$SEEDS_READY)"
fi
## ---- Phase F3: cotrain-ee -> all 5 embodiments (needs E2) -------------------
# Gated on the cotrain checkpoint per task, so one task can start the moment its
# run lands rather than waiting for both.
#
# Shard bookkeeping: merge_xemb_results.py maps result dirs to shard indices by
# mtime order (scripts/merge_xemb_results.py:30,45). A partially-complete cell
# must therefore be topped up from the TAIL -- submit shard i only when
# i >= (result dirs already present). Resubmitting every shard would duplicate a
# finished one and silently shift that mapping, scoring shard 1's seeds against
# shard 0's outcomes.
if [ "$SEEDS_READY" = 1 ]; then
  for t in $TASKS; do S=${SHARDS[$t]}
    CK="demo_clean-${t}-cotrain4-ee-0"
    if [ ! -f "ckpts/ACT/$t/$CK/policy_last.ckpt" ]; then
      echo "F3 $t: waiting on E2 cotrain ckpt"; continue
    fi
    export CKPT_OVERRIDE="$CK"
    for e in $EMB; do
      # count FINISHED shards, not directories -- eval_policy creates its result
      # dir at startup, so a dir count marks a cell done the moment it launches
      # (and would leave a crashed shard permanently unresubmitted).
      have=$(ls -1 "eval_result/$t/ACT/${CFG[$e]}/$CK"/*/_result.txt 2>/dev/null | wc -l)
      [ "$have" -ge "$S" ] && continue
      for i in $(seq "$have" $((S-1))); do
        name="f3_${t:0:2}${e:0:2}$i"
        [ -n "$(inq "$name")" ] && continue
        sub "$name" sbatch/eval_act.sh "$t" $((100/S)) 0 "$e" "${CFG[$e]}" ee \
            "$R/seed_sets/shard__${t}__${i}of${S}.txt"
      done
    done
    unset CKPT_OVERRIDE
  done
fi
echo "done."
