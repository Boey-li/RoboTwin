#!/bin/bash
#SBATCH --job-name=convert_collected
#SBATCH --output=sbatch_logs/convert_%x_%j.out
#SBATCH --error=sbatch_logs/convert_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# ACT-stage conversion for FRESHLY COLLECTED data (scripts/collect_data.py already
# writes the interchange format, so the raw->interchange stage is not needed).
#
# A real script with a bash shebang, not `sbatch --wrap`: --wrap runs the body under
# /bin/sh, where `source` does not exist. That broke an && chain silently once and the
# trailing success echo still fired, reporting ALL_OK for work that never ran.
#
#   sbatch sbatch/convert_collected.sh "open_laptop stack_bowls_two"
set -uo pipefail
R=/coc/flash7/bli678/Projects/RoboTwin
TASKS="${1:?usage: sbatch sbatch/convert_collected.sh \"<tasks>\"}"
# bench:env_cfg_type pairs. The dir alias (arx_x5 -> arx_x5_arx_x5) must already exist:
# collect_data.py names dual-arm output by joining both arm names.
PAIRS="${2:-demo_clean_n200:aloha_agilex demo_clean_n100_arx:arx_x5 demo_clean_n70_franka:franka}"
cd "$R/XPolicyLab/policy/ACT" || exit 1
source "$R/.venv/bin/activate" || exit 1
fail=0
for T in $TASKS; do
  for P in $PAIRS; do
    B="${P%%:*}"; E="${P##*:}"
    if [ ! -d "$R/data/$B/$T/$E/data" ]; then
      echo "SKIP $B/$T/$E (not collected)"; continue
    fi
    echo "=== $B / $T / $E -> ACT ee ==="
    bash process_data.sh "$B" "$T" "$E" ee || { echo "FAIL $B/$T/$E"; fail=1; }
  done
done
# Verify the artifacts exist rather than trusting exit codes.
for T in $TASKS; do
  for P in $PAIRS; do
    B="${P%%:*}"; E="${P##*:}"
    [ -d "$R/data/$B/$T/$E/data" ] || continue
    n=$(ls -1 "processed_data/$B/$T/$E-ee/"episode_*.hdf5 2>/dev/null | wc -l)
    echo "  check $B/$T/$E-ee -> $n episodes"
    [ "$n" -gt 0 ] || fail=1
  done
done
[ "$fail" = 0 ] && echo ALL_OK || echo "INCOMPLETE"
exit $fail
