#!/bin/bash
# Fires when the single-source epoch sweep completes (16 cells x 2 shards), or stalls.
# Counts _result.txt, never directories: eval_policy creates its result dir at STARTUP,
# so a directory count reports "done" the moment the jobs launch.
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
while true; do
  d=0
  for t in place_burger_fries press_stapler; do
    for e in aloha_agilex arx_x5 franka ur5; do
      for ep in 1000 2000; do
        n=$(ls -1 "eval_result/$t/ACT/demo_clean_piper/demo_clean-$t-${e}ep${ep}-ee-0"/*/_result.txt 2>/dev/null | wc -l)
        [ "$n" -ge 2 ] && d=$((d+1))
      done
    done
  done
  [ "$d" -ge 16 ] && { echo "ES DONE: 16/16 epoch-sweep cells complete"; exit 0; }
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    grep -q '^es_' <<<"$q" || { echo "ES STALLED: no es_ jobs queued but only $d/16 cells complete"; exit 1; }
  fi
  sleep 300
done
