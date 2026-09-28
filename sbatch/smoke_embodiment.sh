#!/bin/bash
#SBATCH --job-name=smoke_emb
#SBATCH --output=sbatch_logs/smoke_emb_%x_%j.out
#SBATCH --error=sbatch_logs/smoke_emb_%x_%j.err
#SBATCH --partition="overcap"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#
# Phase A gate: prove each embodiment's task_config actually builds a scene.
# Checkpoint-free - capture_third_view.py sets up the sim with no policy, so this
# validates: task_config -> embodiment resolution, URDF load, curobo path resolution
# (which point into a DIFFERENT checkout and must still resolve), and camera setup.
#   sbatch --export=ALL,TASKCFG=demo_clean_piper,TAG=piper sbatch/smoke_embodiment.sh
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1
export PYOPENGL_PLATFORM=egl DISPLAY="" PYTHONWARNINGS=ignore::UserWarning
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rt_smoke_${SLURM_JOB_ID:-$$}.XXXXXX")"; export TMPDIR="$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT
mkdir -p "$R/figs/smoke"
echo "[$(date)] node=$SLURM_NODELIST task=${TASK:-open_microwave} cfg=${TASKCFG} tag=${TAG}"
python3 scripts/capture_third_view.py --task_name "${TASK:-open_microwave}" \
  --task_config "${TASKCFG}" --seed 100000 --skip-expert \
  --out_prefix "$R/figs/smoke/${TAG}"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
