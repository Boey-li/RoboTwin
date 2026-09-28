#!/bin/bash
# Status of the pinned-seed rerun: scans -> seed draw -> 8 evals.
# Read-only. Exits 0 while work remains, 3 once all 8 results are on disk.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1

RUNS=(
  "adjust_bottle|aloha_agilex|demo_clean|joint|rp_ab_al_j"
  "adjust_bottle|aloha_agilex|demo_clean|ee|rp_ab_al_e"
  "adjust_bottle|arx_x5|demo_clean_arx|joint|rp_ab_ax_j"
  "adjust_bottle|arx_x5|demo_clean_arx|ee|rp_ab_ax_e"
  "grab_roller|aloha_agilex|demo_clean|joint|rp_gr_al_j"
  "grab_roller|aloha_agilex|demo_clean|ee|rp_gr_al_e"
  "grab_roller|arx_x5|demo_clean_arx|joint|rp_gr_ax_j"
  "grab_roller|arx_x5|demo_clean_arx|ee|rp_gr_ax_e"
)
MARKER="$R/seed_sets/.rerun_marker"
Q="$(squeue -u "$USER" -h -o '%i|%j|%T|%P|%R' 2>/dev/null)"
echo "=== $(date '+%F %H:%M') ==="

nscan=$(echo "$Q" | grep -c '|sc2\?_' || true)
if [ "$nscan" -gt 0 ]; then echo "scans: $nscan still running"; else echo "scans: done"; fi
ps -u "$USER" -o cmd 2>/dev/null | grep -q "[_]rerun_pinned.sh" \
  && echo "pipeline: alive" || echo "pipeline: not running"
for t in adjust_bottle grab_roller; do
  f="seed_sets/common__$t.txt"
  if [ -f "$f" ]; then
    echo "seed set $t: $(grep -vc '^#' "$f") seeds  [$(grep -v '^#' "$f" | head -1)..$(grep -v '^#' "$f" | tail -1)]"
  else
    echo "seed set $t: not drawn yet"
  fi
done

done_n=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r TASK ENVCFG TASKCFG ACTION NAME <<< "$spec"
  CKPT="demo_clean-${TASK}-${ENVCFG}-${ACTION}-0"
  # A pinned run is one whose log records a --seed_list_file; match on job name instead,
  # which is unique to this rerun, then take its newest result dir.
  jid=$(echo "$Q" | awk -F'|' -v n="^$NAME$" '$2 ~ n {print $1"|"$3"|"$4"|"$5; exit}')
  # Distinguish rerun output from the original grid by age: MARKER was touched when
  # the rerun was set up, and every earlier result dir has a fixed, older mtime.
  newest=$(ls -td eval_result/"$TASK"/ACT/"$TASKCFG"/"$CKPT"/*/ 2>/dev/null | head -1)
  if [ -n "$jid" ]; then
    # Only count from a dir this rerun created; otherwise we report the old run's videos.
    if [ -n "$newest" ] && [ "$newest" -nt "$MARKER" ]; then
      s=$(ls -1 "$newest"/*success*.mp4 2>/dev/null | wc -l)
      f=$(ls -1 "$newest"/*failure*.mp4 2>/dev/null | wc -l)
      prog=$(printf "%3d/100  %dS/%dF" "$((s+f))" "$s" "$f")
    else
      prog="starting"
    fi
    printf "  %-11s %-34s %s\n" "$NAME" "$jid" "$prog"
  elif [ -f "$newest/_result.txt" ] && [ "$newest" -nt "$MARKER" ]; then
    printf "  %-11s DONE -> %s\n" "$NAME" "$(tail -1 "$newest/_result.txt")"
    done_n=$((done_n+1))
  else
    printf "  %-11s not submitted yet\n" "$NAME"
  fi
done
echo "pinned results complete: $done_n/8"
[ "$done_n" -ge 8 ] && exit 3
exit 0
