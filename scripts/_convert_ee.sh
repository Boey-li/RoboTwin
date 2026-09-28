#!/bin/bash
# Build the ACT ee-action datasets from the EXISTING interchange HDF5s.
#
# No raw re-conversion is needed: process_data_xpolicylab.py already writes
# left/right_ee_poses (7) and left/right_ee_joint_states (1) into every episode,
# so the ee dataset is a repack of data we have. Only the ACT stage runs here.
#
# Packed ee vector = [left_ee_pose(7), left_grip(1), right_ee_pose(7), right_grip(1)] = 16
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS="${TASKS:-place_burger_fries press_stapler}"
ENVS="${ENVS:-aloha_agilex arx_x5 franka piper ur5}"

cd "$R/XPolicyLab/policy/ACT" || exit 1   # CWD-sensitive: relative ../../../data + ./TASK_CONFIGS.json
source "$R/.venv/bin/activate" || exit 1
for ENV in $ENVS; do
  for T in $TASKS; do
    [ -d "$R/data/demo_clean/$T/$ENV/data" ] || { echo "SKIP $ENV/$T (no interchange data)"; continue; }
    echo "=== [$(date +%H:%M:%S)] [$ENV/$T] interchange -> ACT (ee) ==="
    bash process_data.sh demo_clean "$T" "$ENV" ee || exit 1
  done
done
echo "=== registered datasets ==="
python -c "import json;[print(' ',k) for k in sorted(json.load(open('TASK_CONFIGS.json')))]"
