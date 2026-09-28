#!/bin/bash
#SBATCH --job-name=act_train
#SBATCH --output=sbatch_logs/act_train_%x_%j.out
#SBATCH --error=sbatch_logs/act_train_%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"
#
# ACT training for one RoboTwin task / embodiment / action space.
#   sbatch --job-name=act_adjbot sbatch/train_act.sh adjust_bottle
#   sbatch --job-name=ee_adjbot  sbatch/train_act.sh adjust_bottle 0 demo_clean aloha_agilex ee
#
# action_type selects the packed vector: joint -> 14 dims
# [l_arm(6), l_grip(1), r_arm(6), r_grip(1)]; ee -> 16 dims
# [l_ee_pose(7), l_grip(1), r_ee_pose(7), r_grip(1)]. train.sh derives
# ACT_ACTION_DIM from it via utils/get_action_dim.sh, so nothing else changes.
#
# train.sh args: <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Writes checkpoints/<bench>-<task>-<env_cfg_type>-<action_type>-<seed>/

TASK="${1:?usage: sbatch sbatch/train_act.sh <task> [seed] [bench] [env_cfg_type] [action_type]}"
SEED="${2:-0}"
BENCH="${3:-demo_clean}"
ENVCFG="${4:-aloha_agilex}"
ACTION="${5:-joint}"

R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
mkdir -p sbatch_logs

# uv venv, not conda: train.sh calls a bare `python3`, which an activated venv satisfies.
source "${ROBOTWIN_VENV:-$R/.venv}/bin/activate" || exit 1

NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "[$(date)] node=$SLURM_NODELIST gpus=$SLURM_GPUS task=$TASK bench=$BENCH env=$ENVCFG action=$ACTION seed=$SEED"
echo "[$(date)] python=$(which python3)  torch=$(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$R/XPolicyLab/policy/ACT" || exit 1
bash train.sh "$BENCH" "$TASK" "$ENVCFG" "$ACTION" "$SEED" 0
rc=$?
echo "[$(date)] EXIT=$rc"
ls -la "checkpoints/${BENCH}-${TASK}-${ENVCFG}-${ACTION}-${SEED}/" 2>/dev/null
exit $rc
