#!/bin/bash
# Expose one training epoch as an evaluable checkpoint dir.
#
# detr/act_policy.py:140 hard-loads "policy_last.ckpt" by name, so an intermediate
# epoch cannot be evaluated in place. This builds <run>-ep<N>/ containing symlinks:
# policy_last.ckpt -> ../<run>/policy_epoch_<N>_seed_0.ckpt, plus the run's
# dataset_stats.pkl (normalisation must stay the run's own, not be recomputed).
# Symlinks only - no 335 MB copies.
set -uo pipefail
R="${ROBOTWIN_ROOT:-/coc/flash7/bli678/Projects/RoboTwin}"
RUN="${1:?usage: $0 <ckpt_run_name> <epoch> [epoch...]}"; shift
SRC="$R/ckpts/ACT/$RUN"
[ -d "$SRC" ] || { echo "no such run: $SRC"; exit 1; }
for EP in "$@"; do
  W="$SRC/policy_epoch_${EP}_seed_0.ckpt"
  [ -f "$W" ] || { echo "  MISSING epoch $EP for $RUN"; continue; }
  D="$R/ckpts/ACT/${RUN}-ep${EP}"
  mkdir -p "$D"
  ln -sfn "../$RUN/policy_epoch_${EP}_seed_0.ckpt" "$D/policy_last.ckpt"
  ln -sfn "../$RUN/dataset_stats.pkl" "$D/dataset_stats.pkl"
  echo "  ${RUN}-ep${EP}  ->  epoch $EP"
done
