#!/bin/bash
#SBATCH --output=sbatch_logs/bench_%x_%j.out
#SBATCH --error=sbatch_logs/bench_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# A40 vs L40S benchmark for ACT training.
#
# Runs a fixed number of epochs and reports s/it alongside GPU utilisation, so a slow
# result can be attributed: low GPU util => CPU/dataloader bound (utils.py uses
# num_workers=1, prefetch_factor=1), high util => genuinely GPU bound.
#
# Writes to a scratch ckpt dir under $TMPDIR so it cannot touch real checkpoints.
set -uo pipefail
EPOCHS="${1:-150}"
TASK="${2:-place_burger_fries}"
ENVCFG="${3:-aloha_agilex}"
R=/coc/flash7/bli678/Projects/RoboTwin
cd "$R" || exit 1
source "$R/.venv/bin/activate" || exit 1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "[bench] node=$SLURM_NODELIST part=$SLURM_JOB_PARTITION gpu=$GPU"
echo "[bench] cpu=$CPU  cpus_alloc=$SLURM_CPUS_PER_TASK  epochs=$EPOCHS"
echo "[bench] loader: workers=${ACT_NUM_WORKERS:-1} prefetch=${ACT_PREFETCH:-1} persistent=${ACT_PERSISTENT:-0}"

# sample GPU utilisation in the background
# -i restricts sampling to THIS job's GPU. Without it nvidia-smi emits one line per
# GPU (8 on these nodes) and the mean silently blends in other users' jobs -- which
# is how an earlier run produced a bogus '~50% utilisation, dataloader-starved'.
( while true; do nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits; sleep 2; done ) > "$TMPDIR/util.csv" 2>/dev/null &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

CKDIR="$TMPDIR/bench_ckpt"; mkdir -p "$CKDIR"
export ACT_ACTION_DIM=$(bash "$R/XPolicyLab/utils/get_action_dim.sh" "$R" "$ENVCFG" ee)
cd "$R/XPolicyLab/policy/ACT" || exit 1

START=$(date +%s)
python3 imitate_episodes.py \
  --bench_name demo_clean --task_name "$TASK" \
  --ckpt_setting "demo_clean-${TASK}-${ENVCFG}-ee" \
  --ckpt_dir "$CKDIR" --policy_class ACT --kl_weight 10 --chunk_size 50 \
  --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
  --num_epochs "$EPOCHS" --lr 1e-5 --save_freq 100000 --seed 99 2>&1 | tail -3
END=$(date +%s)

kill $SAMP 2>/dev/null
ELAPSED=$((END-START))
UTIL=$(awk -F, '{g+=$1; n++} END{if(n)printf "%.1f", g/n}' "$TMPDIR/util.csv")
MAXU=$(awk -F, 'BEGIN{m=0}{if($1>m)m=$1}END{print m}' "$TMPDIR/util.csv")
echo "[RESULT] node=$SLURM_NODELIST gpu=$GPU elapsed=${ELAPSED}s epochs=$EPOCHS" \
     "s_per_epoch=$(echo "scale=3;$ELAPSED/$EPOCHS"|bc) gpu_util_mean=${UTIL}% gpu_util_max=${MAXU}%"
rm -rf "$CKDIR"
