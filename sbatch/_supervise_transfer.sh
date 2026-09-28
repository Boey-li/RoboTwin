#!/bin/bash
# Status of the cross-embodiment transfer runs: a checkpoint trained on one robot,
# rolled out on the other, on the same pinned scene set as the matched grid.
#
#   ckpt robot  <- env_cfg_type (also selects action dims; both are 14)
#   sim  robot  <- task_config
# Exits 3 once all 4 have results.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1

# task | ckpt env_cfg | sim task_config | label | in-distribution baseline
# baseline = the SIM robot's own policy in the SAME action space, same 100 scenes
RUNS=(
  "adjust_bottle|aloha_agilex|demo_clean_arx|joint|xf_ab_al2ax|aloha->ARX |90.0"
  "adjust_bottle|arx_x5|demo_clean|joint|xf_ab_ax2al|ARX->aloha|85.0"
  "grab_roller|aloha_agilex|demo_clean_arx|joint|xf_gr_al2ax|aloha->ARX |88.0"
  "grab_roller|arx_x5|demo_clean|joint|xf_gr_ax2al|ARX->aloha|89.0"
  "adjust_bottle|aloha_agilex|demo_clean_arx|ee|xe_ab_al2ax|aloha->ARX |81.0"
  "adjust_bottle|arx_x5|demo_clean|ee|xe_ab_ax2al|ARX->aloha|81.0"
  "grab_roller|aloha_agilex|demo_clean_arx|ee|xe_gr_al2ax|aloha->ARX |84.0"
  "grab_roller|arx_x5|demo_clean|ee|xe_gr_ax2al|ARX->aloha|86.0"
)
Q="$(squeue -u "$USER" -h -o '%i|%j|%T|%P|%R' 2>/dev/null)"
echo "=== $(date '+%F %H:%M')  cross-embodiment transfer ==="
echo "   baseline = the SIM robot's own policy on these same 100 scenes"
done_n=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r TASK ENVCFG TASKCFG ACT NAME LABEL BASE <<< "$spec"
  CKPT="demo_clean-${TASK}-${ENVCFG}-${ACT}-0"
  d=$(ls -td eval_result/"$TASK"/ACT/"$TASKCFG"/"$CKPT"/*/ 2>/dev/null | head -1)
  jid=$(echo "$Q" | awk -F'|' -v n="^$NAME$" '$2 ~ n {print $3"/"$4}')
  if [ -n "$d" ] && [ -f "$d/_result.txt" ] && [ -z "$jid" ]; then
    r=$(tail -1 "$d/_result.txt")
    printf "  %-14s %-5s %-11s DONE -> %-6s  (own %s policy: %s%%)\n" "$TASK" "$ACT" "$LABEL" "$r" "$ACT" "$BASE"
    done_n=$((done_n+1))
  elif [ -n "$jid" ]; then
    s=$(ls -1 "$d"/*success*.mp4 2>/dev/null | wc -l); f=$(ls -1 "$d"/*failure*.mp4 2>/dev/null | wc -l)
    printf "  %-14s %-5s %-11s %-16s %3d/100  %dS/%dF\n" "$TASK" "$ACT" "$LABEL" "$jid" "$((s+f))" "$s" "$f"
  else
    printf "  %-14s %-5s %-11s not started\n" "$TASK" "$ACT" "$LABEL"
  fi
done
echo "transfer results complete: $done_n/8"
[ "$done_n" -ge 8 ] && exit 3
exit 0
