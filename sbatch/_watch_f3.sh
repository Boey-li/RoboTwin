#!/bin/bash
# One-shot watcher for Phase F3 (20 eval shards -> 10 cells). Exits on EITHER
# outcome so a stalled batch is never mistaken for progress:
#   DONE     all 10 cells have their 2 result dirs
#   STALLED  no f3_ job queued/running AND cells still incomplete
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
declare -A CFG=( [aloha_agilex]=demo_clean [arx_x5]=demo_clean_arx [franka]=demo_clean_franka \
                 [piper]=demo_clean_piper [ur5]=demo_clean_ur5 )
while true; do
  done_cells=0
  for t in place_burger_fries press_stapler; do
    for e in aloha_agilex arx_x5 franka piper ur5; do
      # count FINISHED shards: the eval creates its timestamped dir at startup,
      # so counting dirs reports "done" the moment the jobs launch.
      n=$(ls -1 "eval_result/$t/ACT/${CFG[$e]}/demo_clean-$t-cotrain4-ee-0"/*/_result.txt 2>/dev/null | wc -l)
      [ "$n" -ge 2 ] && done_cells=$((done_cells+1))
    done
  done
  if [ "$done_cells" -ge 10 ]; then echo "F3 DONE: 10/10 cells complete"; exit 0; fi
  # only trust a queue probe that actually succeeded
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    if ! grep -q '^f3_' <<<"$q"; then
      echo "F3 STALLED: no f3_ jobs queued but only ${done_cells}/10 cells complete"; exit 1
    fi
  fi
  sleep 300
done
