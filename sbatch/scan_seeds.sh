#!/bin/bash
#SBATCH --job-name=seed_scan
#SBATCH --output=sbatch_logs/seed_scan_%x_%j.out
#SBATCH --error=sbatch_logs/seed_scan_%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop,spd-13,kitt"
#
# Scan which eval seeds the scripted expert can solve, for one task+embodiment.
#   sbatch --job-name=sc_ab_al sbatch/scan_seeds.sh adjust_bottle demo_clean 100000 260
#
# Needs a GPU: setup_demo builds the cameras even though no video is written.

TASK="${1:?usage: sbatch sbatch/scan_seeds.sh <task> <task_config> [start] [count]}"
TASKCFG="${2:?}"
START="${3:-100000}"
COUNT="${4:-260}"

R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
mkdir -p sbatch_logs seed_sets
source "${ROBOTWIN_VENV:-$R/.venv}/bin/activate" || exit 1

export PYOPENGL_PLATFORM=egl DISPLAY="" PYTHONWARNINGS=ignore::UserWarning
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rt_scan_${SLURM_JOB_ID:-$$}.XXXXXX")"
export TMPDIR="$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

OUT="$R/seed_sets/${TASK}__${TASKCFG}__${START}.txt"   # range-suffixed: extension scans must not clobber earlier ones
echo "[$(date)] node=$SLURM_NODELIST part=$SLURM_JOB_PARTITION task=$TASK cfg=$TASKCFG seeds=${START}+${COUNT} -> $OUT"

python3 scripts/scan_expert_seeds.py \
    --task_name "$TASK" --task_config "$TASKCFG" \
    --start "$START" --count "$COUNT" --out "$OUT"
rc=$?
echo "[$(date)] EXIT=$rc  pass=$(grep -c PASS "$OUT" 2>/dev/null) / $(wc -l < "$OUT" 2>/dev/null)"
exit $rc
