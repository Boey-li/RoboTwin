#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output=sbatch_logs/convert_%x_%j.out
#SBATCH --error=sbatch_logs/convert_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# raw -> interchange -> ACT for a set of tasks. CPU-only (no SAPIEN rollout), but
# image decode over 50 eps x 5 embodiments is slow enough to not belong on the login node.
#   sbatch sbatch/convert.sh "open_laptop stack_bowls_two" ee
set -uo pipefail
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
export TASKS="${1:?usage: sbatch sbatch/convert.sh \"<tasks>\" [actions]}"
export ACTIONS="${2:-ee}"
echo "[$(date)] node=$SLURM_NODELIST TASKS=$TASKS ACTIONS=$ACTIONS"
bash scripts/_convert_all.sh
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
