#!/bin/bash
# Eval supervisor for the open_laptop / stack_bowls_two volume x diversity grids.
#
# For each of the 12 arms per task it evaluates the policy into HELD-OUT PIPER on the
# task's 100 pinned scenes, at BOTH ep2000 and ep6000. The epoch axis matters: on
# place_burger_fries transfer peaked at 44% @ep2000 and collapsed to 18% @ep6000, so
# reading only the final checkpoint understates the effect.
#
# ep2000 is reached by staging a sibling ckpt dir whose policy_last.ckpt symlinks the
# epoch-2000 weights -- detr/act_policy.py:140 loads that exact filename and nothing else.
#
# ENVCFG must be the TARGET embodiment (piper), never the arm name. eval_act.sh resolves
# it through get_robot_action_dim_info, which reads env_cfg/<name>.yml -- a file that
# exists only for the five real embodiments, so an arm name dies in the width guard with
# an empty policy dim. The arm is selected purely by CKPT_OVERRIDE; env_cfg_type never
# reaches the simulator (only --task_config does), so piper/piper is both legal and honest.
#
#   bash sbatch/_supervise_grid2.sh            report + submit what is ready
#   DRY_RUN=1 bash sbatch/_supervise_grid2.sh  report only
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
DRY_RUN="${DRY_RUN:-0}"
MAX_SUBMIT="${MAX_SUBMIT:-16}"
TASKS="${TASKS:-open_laptop stack_bowls_two}"
S=2                      # eval shards per cell (50 pinned scenes each)
EPOCHS="2000 6000"
# arm|bench|env_cfg_type(width key)|short
ARMS="scene40k1|demo_clean|scene40k1|s41 scene40k2|demo_clean|scene40k2|s42
gridk3T32|demo_clean|gridk3T32|g32 scene40k4|demo_clean|scene40k4|s44
gridk1T64|demo_clean|gridk1T64|g16 gridk2T64|demo_clean|gridk2T64|g26
gridk3T64|demo_clean|gridk3T64|g36 scene80k4|demo_clean|scene80k4|s84
aloha_agilex|demo_clean_n200|aloha_agilex|k16 gridk2T160|demo_clean|gridk2T160|g21
gridk3T160|demo_clean|gridk3T160|g31 cotrain4|demo_clean|cotrain4|ct4"

if [ "$DRY_RUN" != "1" ]; then
  exec 9>"$R/.grid2.lock"; flock -w 300 9 || { echo "another run holds the lock"; exit 0; }
fi
# Fail closed on an unreadable queue: empty squeue output is indistinguishable from an
# idle cluster, and every gate below would then resubmit live work.
Q=""; QOK=0
for _t in 1 2 3; do
  if Q="$(timeout 45 squeue -u "$USER" -h -o '%i|%j|%T|%P|%b' 2>/dev/null)"; then QOK=1; break; fi
  sleep 5
done
[ "$QOK" = 1 ] || { echo "ABORT: squeue unavailable"; exit 0; }
inq() { echo "$Q" | awk -F'|' -v n="^$1$" '$2 ~ n {print $1; exit}'; }


# ---- placement ---------------------------------------------------------------
# The rl2-lab QOS grants TWO separate GPU budgets: a40=8 AND l40s=8. Only the a40 half
# was ever used, leaving 8 non-preemptible GPUs idle on the faster hardware -- L40S
# measured ~1.6x A40 on this workload (1.47-1.73 vs 2.39-2.92 s/epoch, 4 nodes).
# Fill rl2-lab (non-preemptible) first, l40s ahead of a40; only then spill to overcap,
# alternating pools there so neither queue becomes the bottleneck.
# Walltime is requested at SUBMIT time, never by editing eval_act.sh/train_act.sh --
# those files are being read by running jobs, and bash re-reads a script mid-execution at
# a shifted byte offset if it changes (that hazard already killed two trainings here).
# Neither script sets --time, so both inherited the 2-day partition default while real
# runtimes measured 4:26 (eval) and 11:42 (train). A 48h request cannot be backfilled into
# the gaps between other users' jobs, which is why pending jobs sat at ReqNodeNotAvail
# with free GPUs on the node. Limits below are ~2x the observed maximum.
EVAL_TIME="${EVAL_TIME:-10:00:00}"
TRAIN_TIME="${TRAIN_TIME:-20:00:00}"
RR=0
# Cluster-wide free l40s, computed ONCE per run. Having QOS headroom is not the same as
# having a free GPU: the l40s pool is small (bishop is excluded by both job scripts, so
# only grom/dynamics/zhurong are usable) and is routinely saturated by other users.
# Preferring l40s blindly parks jobs in PENDING behind a full pool while a40 sits idle.
# Count free l40s from scontrol AllocTRES, which is authoritative. A squeue-based tally
# undercounts (it missed grom's free GPU and reported 0 while one was idle), because %b
# shows the request, not the node's actual allocation.
#
# Partition matters more than the GPU type here: partition rl2-lab carries QoS=rl2-lab
# (a40=8 AND l40s=8) and that cap is ACCOUNT-WIDE, shared with every other rl2-lab user --
# it is already exhausted by others, which is why l40s jobs there sat in QOSGrpGRES.
# Partition overcap carries QoS=overcap, whose only limit is MaxTRESPU gres/gpu=48 with NO
# per-type restriction, so overcap l40s is freely usable up to that per-user budget.
L40S_TARGET="${L40S_TARGET:-6}"   # standing queued+running claim on overcap l40s
L40S_NODES="${L40S_NODES:-grom dynamics zhurong}"   # bishop excluded by both job scripts
L40S_FREE=0
for _n in $L40S_NODES; do
  _l=$(timeout 20 scontrol show node "$_n" 2>/dev/null)
  echo "$_l" | grep -q 'State=\(DOWN\|DRAIN\)' && continue
  _c=$(echo "$_l" | grep -oP 'CfgTRES=\S*gres/gpu:l40s=\K[0-9]+')
  _a=$(echo "$_l" | grep -oP 'AllocTRES=\S*gres/gpu:l40s=\K[0-9]+')
  L40S_FREE=$((L40S_FREE + ${_c:-0} - ${_a:-0}))
done
[ "$L40S_FREE" -lt 0 ] && L40S_FREE=0
echo "  placement: l40s free=$L40S_FREE ($L40S_NODES)"
place() {   # -> "<partition> <gputype>"
  local a l
  a=$(echo "$Q" | awk -F'|' '$4=="rl2-lab" && $5 ~ /a40/' | wc -l)
  l=$(echo "$Q" | awk -F'|' '$4=="rl2-lab" && $5 ~ /l40s/' | wc -l)
  # Keep a STANDING claim of L40S_TARGET jobs on overcap l40s rather than only
  # submitting when a GPU is already idle. The l40s pool is small and contended, so a
  # free-count gate almost never fires and we would never touch L40S at all; a queued
  # job instead grabs capacity the moment it frees. Capped, so the bulk of throughput
  # still goes to the abundant a40 pool.
  local mine
  mine=$(echo "$Q" | awk -F'|' '$4=="overcap" && $5 ~ /l40s/' | wc -l)
  # rl2-lab l40s FIRST when the account budget has room: same fast hardware as overcap
  # l40s but NON-PREEMPTIBLE, and this study has repeatedly lost jobs to preemption.
  # The l40s=8 budget is shared across the whole rl2-lab account, so it empties and
  # refills as other users come and go -- check it live rather than assuming.
  local rl
  rl=$(timeout 20 squeue -h -A rl2-lab -p rl2-lab -t RUNNING,PENDING -o '%b' 2>/dev/null \
        | grep -oP 'l40s:\K[0-9]+' | awk '{s+=$1} END{print s+0}')
  if   [ "${rl:-8}" -lt 8 ] && [ "$L40S_FREE" -gt 0 ]; then L40S_FREE=$((L40S_FREE-1)); echo "rl2-lab l40s"
  elif [ "$mine" -lt "$L40S_TARGET" ]; then echo "overcap l40s"
  elif [ "$a" -lt 8 ]; then echo "rl2-lab a40"
  else echo "overcap a40"; fi
}

# ---- training watchdog -------------------------------------------------------
# overcap preemption does NOT reliably requeue: jobs vanish from squeue and the epoch
# counter simply stops advancing, which looks identical to "still training" unless you
# cross-check the queue. Three arms died this way before this block existed. Resume from
# the newest epoch checkpoint, carrying the optimiser state (.optim) so momentum is not
# silently reset -- ACT uses AdamW with no LR schedule, so a cold restart is a real
# perturbation, not a no-op.
declare -A AB=( [scene40k1]=s41 [scene40k2]=s42 [gridk3T32]=g32 [scene40k4]=s44
                [gridk1T64]=g16 [gridk2T64]=g26 [gridk3T64]=g36 [scene80k4]=s84
                [gridk2T160]=g21 [gridk3T160]=g31 [cotrain4]=ct4 [aloha_agilex]=k16 )
for T in $TASKS; do
  tp=$([ "$T" = open_laptop ] && echo ol || echo sb)
  for arm in "${!AB[@]}"; do
    bench=demo_clean; [ "$arm" = aloha_agilex ] && bench=demo_clean_n200
    d="$R/ckpts/ACT/$T/$bench-$T-$arm-ee-0"
    [ -d "$d" ] || continue
    [ -f "$d/policy_last.ckpt" ] && continue
    jn="tr_${tp}${AB[$arm]}"
    [ -n "$(inq "$jn")" ] && continue
    ep=$(ls -1 "$d"/policy_epoch_*.ckpt 2>/dev/null | sed 's/.*epoch_//;s/_.*//' | sort -n | tail -1)
    ck="$d/policy_epoch_${ep}_seed_0.ckpt"
    if [ -z "$ep" ] || [ ! -f "$ck" ] || [ ! -f "${ck%.ckpt}.optim" ]; then
      echo "  WATCHDOG $jn dead with no resumable checkpoint - needs a fresh run"; continue
    fi
    if [ "$DRY_RUN" = "1" ]; then echo "  DRY resume $jn from ep$ep"; else
      read -r PART GPU <<< "$(place)"; RR=$((RR+1))
      # START_EPOCH is MANDATORY alongside RESUME_CKPT. imitate_episodes.py:197 loops
      # range(start_epoch, num_epochs) and :236 names checkpoints policy_epoch_{epoch+1}.
      # Resuming without it restarts the counter at 0, so the run trains a further
      # num_epochs on top of already-trained weights AND overwrites policy_epoch_1000/2000
      # with over-trained models -- silently destroying the epoch-controlled comparison
      # (a file labelled ep2000 then holds e.g. 6000 epochs of training).
      sbatch --partition="$PART" --account=rl2-lab --qos=short --gpus-per-node="$GPU:1" \
        --time="$TRAIN_TIME" --job-name="$jn" --export=ALL,RESUME_CKPT="$ck",START_EPOCH="$ep" \
        sbatch/train_act.sh "$T" 0 "$bench" "$arm" ee >/dev/null 2>&1 \
        && echo "  WATCHDOG resumed $jn from ep$ep -> $PART/$GPU"
      Q="$Q"$'\n'"0|$jn|PENDING|$PART|gres/gpu:$GPU:1"
    fi
  done
done

sub=0
for T in $TASKS; do
  tp=$([ "$T" = open_laptop ] && echo ol || echo sb)
  SEEDS="$R/seed_sets/common__$T.txt"
  [ -f "$SEEDS" ] || { echo "$T: no pinned seed set"; continue; }
  for A in $ARMS; do
    IFS='|' read -r arm bench _env ab <<< "$A"   # _env: ckpt-dir component only, not passed to eval
    BASE="ckpts/ACT/$T/$bench-$T-$arm-ee-0"
    for EP in $EPOCHS; do
      if [ "$EP" = 6000 ]; then
        [ -f "$BASE/policy_last.ckpt" ] || continue
        CK="$bench-$T-$arm-ee-0"
      else
        SRC="$BASE/policy_epoch_${EP}_seed_0.ckpt"
        [ -f "$SRC" ] || continue
        CK="$bench-$T-${arm}ep${EP}-ee-0"
        D="ckpts/ACT/$T/$CK"
        if [ ! -L "$D/policy_last.ckpt" ]; then
          [ "$DRY_RUN" = "1" ] || { mkdir -p "$D"
            ln -sfn "$R/$SRC" "$D/policy_last.ckpt"
            ln -sfn "$R/$BASE/dataset_stats.pkl" "$D/dataset_stats.pkl"; }
        fi
      fi
      # Count FINISHED shards, not directories: eval_policy creates its result dir at
      # startup, so a dir count marks a cell done the moment it launches. Top up from
      # the TAIL -- merge_xemb_results.py maps dirs to shards by mtime order, so
      # resubmitting an already-finished shard would shift that mapping and score one
      # shard's seeds against another's outcomes.
      # Decide per-SHARD, not by counting finished results. A count cannot say WHICH
      # shard finished: if shard 1 completes while shard 0 dies, have=1 and the old code
      # resubmitted index 1 -- re-running a finished shard while shard 0 stayed missing.
      # That produced a cell with three shard-1 runs and no shard 0, which the mtime-based
      # reader then averaged as if it were a complete 100-scene pair.
      # A shard is done iff its own job log ends with the harness's "Final success rate".
      for i in $(seq 0 $((S-1))); do
        name="ev${tp}${ab}${EP:0:1}$i"
        # A shard counts as done only if a log FINISHED IT *and* that log is admissible.
        # For arms quarantined over START_EPOCH contamination, pre-quarantine logs record
        # runs against an over-trained checkpoint: the grid reader discards them, so the
        # supervisor must not treat them as done or the shard is never re-run and the cell
        # stays permanently half-measured.
        qcut=0
        # EXACT task+arm pairs only. Matching on arm alone also caught
        # open_laptop/cotrain4, which was never quarantined -- that would have thrown
        # away a valid completed cell and re-run it for nothing.
        case "${T}|${arm}" in
          stack_bowls_two\|cotrain4|open_laptop\|gridk1T64|open_laptop\|aloha_agilex) qcut=1788331740 ;;
        esac
        if [ "$qcut" -gt 0 ]; then
          fin=$(find sbatch_logs -maxdepth 1 -name "act_eval_${name}_*.out" -newermt "@$qcut" \
                 -exec grep -l "Final success rate" {} + 2>/dev/null | head -1)
        else
          fin=$(grep -l "Final success rate" sbatch_logs/act_eval_${name}_*.out 2>/dev/null | head -1)
        fi
        [ -n "$fin" ] && continue
        [ -n "$(inq "$name")" ] && continue
        [ "$sub" -ge "$MAX_SUBMIT" ] && continue
        read -r PART GPU <<< "$(place)"; RR=$((RR+1))
        if [ "$DRY_RUN" = "1" ]; then
          echo "  DRY $name  $PART/$GPU  CKPT=$CK shard=$i"
        else
          sbatch --partition="$PART" --account=rl2-lab --qos=short \
            --gpus-per-node="$GPU:1" --time="$EVAL_TIME" --job-name="$name" \
            --export=ALL,BENCH="$bench",CKPT_OVERRIDE="$CK" \
            sbatch/eval_act.sh "$T" $((100/S)) 0 piper demo_clean_piper ee "$SEEDS" >/dev/null 2>&1 \
            && echo "  + $name -> $PART/$GPU (ep$EP shard$i)"
          Q="$Q"$'\n'"0|$name|PENDING|$PART|gres/gpu:$GPU:1"
        fi
        sub=$((sub+1))
      done
    done
  done
done
echo "submitted $sub eval jobs (cap $MAX_SUBMIT)"
