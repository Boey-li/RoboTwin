#!/bin/bash
#SBATCH --job-name=curobo_build
#SBATCH --output=sbatch_logs/curobo_build_%j.out
#SBATCH --error=sbatch_logs/curobo_build_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"
#
# curobo must be compiled ON a GPU node: its setup.py probes the local device to
# pick CUDA archs, and on the login node (no nvcc, no GPU) that raises
# IndexError: list index out of range.

R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
cd "$R" || exit 1
mkdir -p sbatch_logs
source "${ROBOTWIN_VENV:-$R/.venv}/bin/activate" || exit 1

UV=/coc/flash7/bli678/miniconda3/bin/uv
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.6"          # A40 = sm_86
# System nvcc on these nodes is CUDA 13.0, but torch is cu121; torch's
# _check_cuda_version rejects the major mismatch. Use the pinned 12.1 toolkit.
export CUDA_HOME="${CUDA_HOME:-$R/.cuda121}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# CUDA 12.1 refuses gcc > 12; the nodes ship a newer default compiler.
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

echo "[$(date)] node=$SLURM_NODELIST"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "nvcc: $(which nvcc || echo MISSING)"
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

[ -d "$R/envs/curobo" ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$R/envs/curobo"

echo "[$(date)] building curobo"
"$UV" pip install --python "$R/.venv/bin/python" -e "$R/envs/curobo" --no-build-isolation
rc=$?
echo "[$(date)] build rc=$rc"

python3 -c "import curobo; print('curobo import OK')" || rc=1

# While we hold a GPU, also verify SAPIEN can acquire a render device -
# the other unverified prerequisite for rollout.
export PYOPENGL_PLATFORM=egl DISPLAY=""
python3 - <<'PY'
try:
    import sapien
    print("sapien", sapien.__version__)
    import sapien.core as sc
    print("SAPIEN render probe: import OK")
except Exception as e:
    print("SAPIEN probe FAILED:", type(e).__name__, e)
PY

echo "[$(date)] EXIT=$rc"
exit $rc
