#!/bin/bash
# Regenerate every interchange + ACT dataset from the raw symlinks.
#
# Layout (raw and generated trees are deliberately identical in shape):
#   data/raw_data/<task>/<env_cfg_type>                       symlinks -> scratch
#   data/demo_clean/<task>/<env_cfg_type>/data/episode_%07d.hdf5   interchange, generated
#   XPolicyLab/policy/ACT/processed_data/demo_clean/<task>/<env>-joint/   ACT native, generated
#
# Why the temp shim below: process_data_xpolicylab.py:401 computes its input as
#   input_root / <task_name> / <task_config>
# and its output (line 419) as
#   output_root / <task_config> / <task_name> / <env_cfg_type>
# One shared output name ("demo_clean") therefore forces the embodiment ABOVE the
# task on input - the opposite nesting from data/raw_data. Rather than keep a second
# permanent tree just to satisfy that, we build the <env>/<task>/demo_clean view in a
# temp dir (symlinks only, nothing copied) and throw it away afterwards.
set -uo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$R/.venv/bin/python"
TASKS="${TASKS:-place_burger_fries press_stapler}"
ENVS="${ENVS:-aloha_agilex arx_x5 franka piper ur5}"
ACTIONS="${ACTIONS:-joint}"   # grid arms are trained on ee; joint kept as the default

SHIM="$(mktemp -d /tmp/rt_convert_shim.XXXXXX)"
cleanup() { rm -rf "$SHIM"; }
trap cleanup EXIT

for ENV in $ENVS; do
  for T in $TASKS; do
    src="$R/data/raw_data/$T/$ENV"
    [ -e "$src/data" ] || { echo "SKIP $ENV/$T (no $src/data)"; continue; }
    mkdir -p "$SHIM/$ENV/$T"
    ln -sfn "$src" "$SHIM/$ENV/$T/demo_clean"
  done
done

for ENV in $ENVS; do
  for T in $TASKS; do
    [ -e "$SHIM/$ENV/$T/demo_clean" ] || continue
    echo "=== [$ENV/$T] raw -> interchange ==="
    "$PY" "$R/scripts/process_data_xpolicylab.py" "$T" demo_clean 50 \
        --env-cfg-type "$ENV" \
        --input-root "$SHIM/$ENV" \
        --output-root "$R/data" \
        --overwrite || exit 1
  done
done

cd "$R/XPolicyLab/policy/ACT" || exit 1   # CWD-sensitive: relative ../../../data + ./TASK_CONFIGS.json
source "$R/.venv/bin/activate" || exit 1
for ENV in $ENVS; do
  for T in $TASKS; do
    [ -d "$R/data/demo_clean/$T/$ENV/data" ] || continue
    for A in $ACTIONS; do
      echo "=== [$ENV/$T] interchange -> ACT ($A) ==="
      bash process_data.sh demo_clean "$T" "$ENV" "$A" || exit 1
    done
  done
done

echo "=== registered datasets ==="
python -c "import json;[print(' ',k) for k in sorted(json.load(open('TASK_CONFIGS.json')))]"
