#!/bin/bash
#SBATCH --job-name=capture_tv
#SBATCH --output=sbatch_logs/capture_tv_%j.out
#SBATCH --error=sbatch_logs/capture_tv_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop,spd-13,kitt"
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1
export PYOPENGL_PLATFORM=egl DISPLAY="" PYTHONWARNINGS=ignore::UserWarning
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rt_cap_${SLURM_JOB_ID:-$$}.XXXXXX")"; export TMPDIR="$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT
mkdir -p "$R/figs"
SEED="${SEED:-100008}"
for spec in "demo_clean|aloha" "demo_clean_arx|arx"; do
  IFS='|' read -r CFG TAG <<< "$spec"
  python3 scripts/capture_third_view.py --task_name "${TASK:-adjust_bottle}" \
      --task_config "$CFG" --seed "$SEED" --out_prefix "$R/figs/${TAG}_${TASK:-adjust_bottle}"
done
ls -la "$R/figs"
