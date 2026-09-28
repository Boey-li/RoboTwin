#!/bin/bash
# Supervisor for the cross-embodiment CONTROLS (es_/cb_/cc_/ld_).
# Fails closed on an unreadable queue: an empty squeue result is indistinguishable from
# a real outage, and every gate below would then read "nothing running".
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
Q=""; QOK=0
for _t in 1 2 3; do
  if Q="$(timeout 45 squeue -u "$USER" -h -o '%i|%j|%T|%P|%R' 2>/dev/null)"; then QOK=1; break; fi
  sleep 5
done
[ "$QOK" = 1 ] || { echo "ABORT: squeue unavailable"; exit 0; }
nq(){ echo "$Q" | grep -c "|$1"; }
run(){ echo "$Q" | awk -F'|' -v p="^$1" '$2~p && $3=="RUNNING"' | wc -l; }
pend(){ echo "$Q" | awk -F'|' -v p="^$1" '$2~p && $3!="RUNNING"' | wc -l; }

echo "=== $(date '+%F %H:%M')  controls ==="
printf "  %-34s %2d run  %2d pend\n" "es_ epoch sweep (single-src)" "$(run es_)" "$(pend es_)"
printf "  %-34s %2d run  %2d pend\n" "cb_ Control B (diversity)"    "$(run cb_)" "$(pend cb_)"
printf "  %-34s %2d run  %2d pend\n" "cc_ Control C (volume)"       "$(run cc_)" "$(pend cc_)"
printf "  %-34s %2d run  %2d pend\n" "ld_ dataloader sweep"         "$(run ld_)" "$(pend ld_)"

# --- epoch sweep completion (32 shards -> 16 cells) --------------------------
done_es=0
for t in place_burger_fries press_stapler; do
  for e in aloha_agilex arx_x5 franka ur5; do
    for ep in 1000 2000; do
      n=$(ls -1 "eval_result/$t/ACT/demo_clean_piper/demo_clean-$t-${e}ep${ep}-ee-0"/*/_result.txt 2>/dev/null | wc -l)
      [ "$n" -ge 2 ] && done_es=$((done_es+1))
    done
  done
done
echo "  es_ cells complete: $done_es/16"

# --- control trainings: report epoch progress from the live logs -------------
for pre in cb cc; do
  for f in $(ls -t sbatch_logs/act_train_${pre}_*.err 2>/dev/null | head -8); do
    j=$(basename "$f" | sed 's/.*_\([0-9]*\)\.err/\1/')
    tag=$(basename "$f" | sed 's/act_train_//; s/_[0-9]*\.err//')
    last=$(tail -c 300 "$f" 2>/dev/null | tr '\r' '\n' | grep -o '[0-9]*/6000' | tail -1)
    ck=$(ls ckpts/ACT/*/*"$(echo $tag|sed 's/^c[bc]_[a-z]*_//')"*/policy_last.ckpt 2>/dev/null | wc -l)
    [ -n "$last" ] && printf "    %-24s %-10s %s\n" "$tag" "${last:-?}" "$([ "$ck" -gt 0 ] && echo '[done]')"
  done
done

# --- requeue anything that died on overcap without finishing ----------------
gone=$(echo "$Q" | grep -c PREEMPT)
[ "$gone" -gt 0 ] && echo "  !! $gone job(s) show PREEMPT state - inspect before requeue"
echo "done."
