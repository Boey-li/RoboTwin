#!/bin/bash
# Fires when an N=200 collection finishes for <task>, or when the job vanishes early.
# Counts CANONICAL episode_%07d.hdf5 only -- collect_data.py:243 writes that scheme, while
# the pre-staged 50 use the older episodeN.hdf5 naming. Counting both fired this early once.
T="${1:?usage: _watch_collect2.sh <task>}"
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
O="data/demo_clean_n200/$T/aloha_agilex"
while true; do
  n=$(ls "$O/data" 2>/dev/null | grep -c '^episode_[0-9]*\.hdf5$')
  [ "$n" -ge 200 ] && { echo "COLLECT DONE ($T): $n canonical episodes"; exit 0; }
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    grep -q '^col_' <<<"$q" || { echo "COLLECT STALLED ($T): job gone at $n/200"; exit 1; }
  fi
  sleep 240
done
