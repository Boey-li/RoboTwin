#!/bin/bash
# Fires when both single-robot N=160 -> piper cells reach n=100, or when the jobs vanish early.
# Counts _result.txt for completion (eval creates its result dir at STARTUP, so a dir count
# would report done at launch) but reports trial counts from tagged mp4s.
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
T=place_burger_fries
A=demo_clean_n200-place_burger_fries-aloha_agilex-ee-0
B=demo_clean_n200-place_burger_fries-aloha_agilexep2000-ee-0
while true; do
  d=0
  for tag in "$A" "$B"; do
    n=$(ls -1 "eval_result/$T/ACT/demo_clean_piper/$tag"/*/_result.txt 2>/dev/null | wc -l)
    [ "$n" -ge 2 ] && d=$((d+1))
  done
  [ "$d" -ge 2 ] && { echo "N200 EVAL DONE: both cells complete"; exit 0; }
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    grep -q '^n200e_' <<<"$q" || { echo "N200 EVAL STALLED: no n200e_ jobs but only $d/2 cells complete"; exit 1; }
  fi
  sleep 300
done
