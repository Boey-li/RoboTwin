#!/bin/bash
set -uo pipefail
R=/coc/flash7/bli678/Projects/RoboTwin; cd "$R" || exit 1
bash scripts/_convert_all.sh >> sbatch_logs/convert_joint_xemb.log 2>&1
bash scripts/_convert_ee.sh  >> sbatch_logs/convert_ee_xemb.log 2>&1
"$R/.venv/bin/python" - >> sbatch_logs/convert_chain.log 2>&1 <<'PY'
import json
c=json.load(open("XPolicyLab/policy/ACT/TASK_CONFIGS.json"))
want=[f"demo_clean-{t}-{e}-{a}" for t in ("place_burger_fries","press_stapler")
      for e in ("aloha_agilex","arx_x5","franka","piper","ur5") for a in ("joint","ee")]
print(f"TASK_CONFIGS: {sum(1 for k in want if k in c)}/20")
PY
