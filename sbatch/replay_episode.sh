#!/bin/bash
#SBATCH --job-name=replay
#SBATCH --output=sbatch_logs/replay_%x_%j.out
#SBATCH --error=sbatch_logs/replay_%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop,spd-13,kitt"
#
# Open-loop replay of one recorded episode in the RoboTwin eval env.
# Needs a GPU: setup_scene() uses the "rt" shader, and setup_demo() builds cameras.
#
#   sbatch sbatch/replay_episode.sh data/demo_clean/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5
set -uo pipefail
EP="${1:-data/demo_clean/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5}"
shift || true
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1
echo "[$(date)] node=${SLURM_NODELIST:-local} episode=$EP"
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONWARNINGS=ignore::UserWarning python scripts/replay_episode_eval.py --episode-file "$EP" "$@"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
