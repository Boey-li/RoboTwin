#!/bin/bash
# Phase C: joint conversion (raw -> interchange -> ACT/joint), then ee (repack only).
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"; cd "$R" || exit 1
L="$R/sbatch_logs/convert_chain.log"
echo "[$(date)] waiting for joint conversion" >> "$L"
while pgrep -u "$USER" -f "_convert_all.sh" >/dev/null; do sleep 60; done
echo "[$(date)] joint done -> starting ee" >> "$L"
bash scripts/_convert_ee.sh >> "$R/sbatch_logs/convert_ee_xemb.log" 2>&1
echo "[$(date)] ee done rc=$?" >> "$L"
"$R/.venv/bin/python" - >> "$L" 2>&1 <<'PY'
import json
c=json.load(open("XPolicyLab/policy/ACT/TASK_CONFIGS.json"))
want=[f"demo_clean-{t}-{e}-{a}" for t in ("open_microwave","press_stapler")
      for e in ("aloha_agilex","arx_x5","franka","piper","ur5") for a in ("joint","ee")]
have=[k for k in want if k in c]
print(f"TASK_CONFIGS: {len(have)}/20 registered")
missing=[k for k in want if k not in c]
if missing: print("MISSING:", missing)
PY
