#!/bin/bash
#SBATCH --job-name=cap_all
#SBATCH --output=sbatch_logs/cap_all_%j.out
#SBATCH --error=sbatch_logs/cap_all_%j.err
#SBATCH --partition="overcap"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop,spd-13,kitt"
#
# Render the front (observer) view of every task at its home pose, aloha-agilex.
#   sbatch --export=ALL,SHARD=0,NSHARD=5 sbatch/capture_all_tasks.sh
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1
export PYOPENGL_PLATFORM=egl DISPLAY="" PYTHONWARNINGS=ignore::UserWarning
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rt_capall_${SLURM_JOB_ID:-$$}.XXXXXX")"; export TMPDIR="$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT
mkdir -p "$R/figs/tasks"
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
i=0
for T in $(ls -1 /coc/flash7/scratch/robotwin_shared/dataset); do
  if [ $((i % NSHARD)) -eq "$SHARD" ]; then
    if [ -f "$R/figs/tasks/${T}_home.png" ]; then echo "skip $T (exists)"; else
      echo "=== [$(date +%H:%M:%S)] $T ==="
      timeout 300 python3 scripts/capture_third_view.py --task_name "$T" \
        --task_config demo_clean --seed 100000 --skip-expert \
        --out_prefix "$R/figs/tasks/${T}" || echo "  FAILED: $T"
    fi
  fi
  i=$((i+1))
done
echo "[$(date)] shard $SHARD done: $(ls -1 $R/figs/tasks/*_home.png 2>/dev/null | wc -l) images total"
