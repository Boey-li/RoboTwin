#!/bin/bash
# Fires when the N=200 collection has written 200 episodes in the CANONICAL naming
# (collect_data.py:243 writes episode_%07d.hdf5). The 50 pre-staged files use the older
# episodeN.hdf5 scheme and must not be counted -- doing so fired this watcher early once.
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
O=data/demo_clean_n200/place_burger_fries/aloha_agilex
while true; do
  n=$(ls "$O/data" 2>/dev/null | grep -c '^episode_[0-9]*\.hdf5$')
  [ "$n" -ge 200 ] && { echo "COLLECT DONE: $n canonical episodes"; exit 0; }
  if q=$(timeout 45 squeue -u "$USER" -h -o '%j' 2>/dev/null); then
    grep -q '^col_' <<<"$q" || { echo "COLLECT STALLED: job gone at $n/200"; exit 1; }
  fi
  sleep 180
done
