#!/bin/bash
# Build the RoboTwin runtime as a uv venv at RoboTwin/.venv.
#
# Mirrors scripts/_install.sh, but with uv instead of conda pip, and split into
# stages so a failure in the heavy source builds (pytorch3d, curobo) does not
# lose the fast base install. Re-runnable: each stage is idempotent.
set -uo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$R/.venv"
PY="$VENV/bin/python"
UV=/coc/flash7/bli678/miniconda3/bin/uv
export UV_LINK_MODE=copy          # NFS-safe

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "STAGE 1: create venv (python 3.10)"
"$UV" venv --python 3.10 "$VENV" || exit 1

log "STAGE 2: base requirements"
"$UV" pip install --python "$PY" -r "$R/scripts/requirements.txt" || exit 1

log "STAGE 3: XPolicyLab (editable)"
"$UV" pip install --python "$PY" -e "$R/XPolicyLab" || exit 1

log "STAGE 4: sapien + mplib source patches"
SAPIEN_LOCATION="$("$PY" -m pip show sapien 2>/dev/null | grep 'Location' | awk '{print $2}')/sapien"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
if [ -f "$URDF_LOADER" ]; then
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
    log "  patched $URDF_LOADER"
else
    log "  WARN: urdf_loader.py not found at $URDF_LOADER"
fi
MPLIB_LOCATION="$("$PY" -m pip show mplib 2>/dev/null | grep 'Location' | awk '{print $2}')/mplib"
PLANNER="$MPLIB_LOCATION/planner.py"
if [ -f "$PLANNER" ]; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
    log "  patched $PLANNER"
else
    log "  WARN: planner.py not found at $PLANNER"
fi

log "STAGE 5: pytorch3d (source build, slow)"
"$UV" pip install --python "$PY" "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation || log "  WARN: pytorch3d failed (not required for joint-space ACT)"

log "STAGE 6: curobo v0.7.8 (source build, slow)"
if [ ! -d "$R/envs/curobo" ]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$R/envs/curobo" || \
        log "  WARN: curobo clone failed"
fi
if [ -d "$R/envs/curobo" ]; then
    "$UV" pip install --python "$PY" -e "$R/envs/curobo" --no-build-isolation || \
        log "  WARN: curobo build failed"
    "$UV" pip install --python "$PY" warp-lang==1.12.0 setuptools==69.5.1 || true
fi

log "DONE"
"$PY" - <<'PY'
import importlib
for m in ("torch","numpy","sapien","mplib","h5py","websockets","XPolicyLab","curobo","pytorch3d"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} OK  {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  {m:12s} --  {type(e).__name__}")
PY
