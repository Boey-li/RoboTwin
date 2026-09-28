#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --output=sbatch_logs/collect_%x_%j.out
#SBATCH --error=sbatch_logs/collect_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# Collect scripted-expert demos for one task/config. Needs a GPU: setup_demo builds cameras.
#
# scripts/collect_data.py is RESUMABLE -- if seed.txt exists in the output dir it reads it,
# sets suc_num=len(seeds) and epid=max(seed)+1, and appends until episode_num is reached.
# The output dir here is pre-seeded with the existing 50 demos, so the resulting 200-episode
# set is a strict SUPERSET of the data every published result already uses.
set -uo pipefail
TASK="${1:?usage: sbatch sbatch/collect_demos.sh <task> <task_config>}"
CFG="${2:?}"
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1
echo "[$(date)] node=$SLURM_NODELIST task=$TASK cfg=$CFG"
OUT="data/$CFG/$TASK/aloha_agilex"
echo "[before] episodes=$(ls $OUT/data 2>/dev/null | wc -l) seeds=$(wc -w < $OUT/seed.txt 2>/dev/null)"
PYTHONWARNINGS=ignore::UserWarning python scripts/collect_data.py "$TASK" "$CFG"
rc=$?
echo "[after]  episodes=$(ls $OUT/data 2>/dev/null | wc -l) seeds=$(wc -w < $OUT/seed.txt 2>/dev/null)"
echo "[$(date)] EXIT=$rc"
exit $rc
