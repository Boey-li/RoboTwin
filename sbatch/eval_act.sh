#!/bin/bash
#SBATCH --job-name=act_eval
#SBATCH --output=sbatch_logs/act_eval_%x_%j.out
#SBATCH --error=sbatch_logs/act_eval_%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop,spd-13,kitt"
#
# ACT clean2clean rollout for one task/embodiment.
#   sbatch --job-name=ev_adjbot sbatch/eval_act.sh adjust_bottle 100 0 aloha_agilex demo_clean
#   sbatch --job-name=ev_arxbot sbatch/eval_act.sh adjust_bottle 100 0 arx_x5 demo_clean_arx
#   sbatch --job-name=ee_adjbot sbatch/eval_act.sh adjust_bottle 100 0 aloha_agilex demo_clean ee
#   sbatch --job-name=sm_adjbot sbatch/eval_act.sh adjust_bottle 2      # smoke test
#
# The simulator reads the embodiment from the TASK CONFIG, not from a flag, so
# an ARX rollout needs BOTH env_cfg_type=arx_x5 (selects the checkpoint and the
# XPolicyLab env-cfg branch) and task_config=demo_clean_arx (embodiment: [ARX-X5]).
#
# eval.sh takes 10 args:
#   bench task ckpt env_cfg action seed policy_gpu env_gpu policy_env eval_env
# Our patched scripts accept a venv PATH (contains '/') in place of a conda name.
# test_num / task_config are injected through ROBOTWIN_EVAL_ARGS_FILE, which
# scripts/eval_policy.sh appends after "$@" - later argv wins in argparse, so no
# source edits are needed.
#
# To run off rl2-lab when its a40 pool is saturated, override at submit time:
#   sbatch --partition=overcap --account=overcap --qos=... sbatch/eval_act.sh ...

TASK="${1:?usage: sbatch sbatch/eval_act.sh <task> [test_num] [seed] [env_cfg_type] [task_config] [action_type]}"
TEST_NUM="${2:-100}"
SEED="${3:-0}"
ENVCFG="${4:-aloha_agilex}"
TASKCFG="${5:-demo_clean}"
ACTION="${6:-joint}"
SEEDFILE="${7:-}"           # optional: pin the scene set (see scripts/scan_expert_seeds.py)
BENCH="${BENCH:-demo_clean}"          # training bench name; checkpoints are keyed on it


R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
VENV="${ROBOTWIN_VENV:-$R/.venv}"
# CKPT_OVERRIDE lets an epoch-sweep point at a sibling dir holding one epoch's
# weights as policy_last.ckpt (detr/act_policy.py:140 loads that exact filename).
CKPT="${CKPT_OVERRIDE:-${BENCH}-${TASK}-${ENVCFG}-${ACTION}-${SEED}}"

# GUARD: eval_policy_xpolicylab.py parses --env_cfg_type but never consumes it - the
# simulated robot comes ONLY from --task_config. Cross-embodiment transfer relies on
# that (policy from ENVCFG, sim from TASKCFG), so they are ALLOWED to differ.
#
# What is NOT safe is a WIDTH mismatch: envs/_base_task.py:1504 slices the action by the
# SIMULATOR's arm width, so a 16-dim franka action into a 14-dim sim mis-slices into
# plausible garbage instead of raising. Require the two widths to agree, and fail closed
# on any task_config we do not recognise.
case "${TASKCFG}" in
  demo_clean)        SIM_ENV=aloha_agilex ;;
  demo_clean_arx)    SIM_ENV=arx_x5 ;;
  demo_clean_franka) SIM_ENV=franka ;;
  demo_clean_piper)  SIM_ENV=piper ;;
  demo_clean_ur5)    SIM_ENV=ur5 ;;
  *) echo "[FATAL] unknown task_config '${TASKCFG}'" >&2; exit 1 ;;
esac
_dim() { PYTHONPATH="$R" "$VENV/bin/python" -c \
  "from XPolicyLab.utils.process_data import get_action_dim;print(get_action_dim('$1','$2'))"; }
POL_DIM=$(_dim "$ENVCFG" "$ACTION"); SIM_DIM=$(_dim "$SIM_ENV" "$ACTION")
if [ "$POL_DIM" != "$SIM_DIM" ]; then
  echo "[FATAL] action width mismatch: policy ${ENVCFG}/${ACTION}=${POL_DIM} but sim ${SIM_ENV}/${ACTION}=${SIM_DIM}." >&2
  echo "        _base_task.py slices by the sim width, so this would silently mis-slice." >&2
  exit 1
fi
[ "$ENVCFG" != "$SIM_ENV" ] && echo "[transfer] policy=${ENVCFG} -> sim=${SIM_ENV} (both ${POL_DIM}-dim)"

cd "$R" || exit 1
mkdir -p sbatch_logs
source "$VENV/bin/activate" || exit 1

export PYOPENGL_PLATFORM=egl DISPLAY="" PYTHONWARNINGS=ignore::UserWarning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# mplib leaks ~12k empty pymp-* dirs per 100-trial rollout into TMPDIR. Give the
# job its own scratch dir so the mess is confined and removed on exit instead of
# accumulating under the repo / /tmp.
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rt_eval_${SLURM_JOB_ID:-$$}.XXXXXX")"
export TMPDIR="$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

ARGS_FILE="$JOB_TMP/eval_args"
printf -- '--test_num\n%s\n--task_config\n%s\n' "$TEST_NUM" "$TASKCFG" > "$ARGS_FILE"
if [ -n "$SEEDFILE" ]; then
  [ -f "$SEEDFILE" ] || { echo "[FATAL] seed list not found: $SEEDFILE"; exit 1; }
  printf -- '--seed_list_file\n%s\n' "$SEEDFILE" >> "$ARGS_FILE"
fi
export ROBOTWIN_EVAL_ARGS_FILE="$ARGS_FILE"

echo "[$(date)] node=$SLURM_NODELIST part=$SLURM_JOB_PARTITION task=$TASK env=$ENVCFG task_cfg=$TASKCFG action=$ACTION trials=$TEST_NUM seeds=${SEEDFILE:-auto} ckpt=$CKPT"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -c "import curobo" 2>/dev/null && echo "curobo: available" || echo "curobo: NOT available (planner falls back)"

# Checkpoints are grouped by task (ckpts/ACT/<task>/<run>), so resolve one level down.
CKPT_DIR=$(compgen -G "$R/ckpts/ACT/*/$CKPT" | head -1)
[ -z "$CKPT_DIR" ] && [ -d "$R/ckpts/ACT/$CKPT" ] && CKPT_DIR="$R/ckpts/ACT/$CKPT"
if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/policy_last.ckpt" ]; then
  echo "[FATAL] no usable checkpoint for $CKPT (looked in $R/ckpts/ACT/*/$CKPT)"; exit 1
fi
ls -la "$CKPT_DIR"

cd "$R/XPolicyLab/policy/ACT" || exit 1
bash eval.sh "$BENCH" "$TASK" "$CKPT" "$ENVCFG" "$ACTION" "$SEED" 0 0 "$VENV" "$VENV"
rc=$?
echo "[$(date)] EXIT=$rc"

echo "=== result files ==="
find "$R/eval_result/$TASK" -name "_result.txt" -newermt "-6 hours" 2>/dev/null | while read f; do
  echo "  $f -> $(tail -1 "$f")"
done
exit $rc
