"""Report which eval seeds the scripted expert can solve, for one task+embodiment.

Why this exists: eval_policy_xpolicylab.py:786-807 accepts a candidate seed as a
trial only if the scripted expert solves that scene. Expert feasibility depends on
the robot, so two embodiments starting from the same seed evaluate on *different*
scene sets - measured overlap was only 64-71/100. Comparing success rates across
embodiments is therefore not apples-to-apples.

This scans a seed range and prints one "<seed> PASS|FAIL <reason>" line per seed.
Intersecting the PASS sets across embodiments gives a common scene set that every
run can be evaluated on. No policy server is involved - the expert drives.

    python scripts/scan_expert_seeds.py --task_name adjust_bottle \
        --task_config demo_clean --start 100000 --count 260 --out seeds_aloha.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_policy_xpolicylab import class_decorator, load_task_args  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--start", type=int, default=100000)
    ap.add_argument("--count", type=int, default=260)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    usr_args = {
        "task_name": a.task_name,
        "task_config": a.task_config,
        "policy_name": "expert_scan",
        "seed": 0,
        "ckpt_setting": "scan",
    }
    args, embodiment_name = load_task_args(usr_args)
    args["eval_video_log"] = False      # no policy, no video - just feasibility
    args["render_freq"] = 0
    args["eval_mode"] = True
    task_env = class_decorator(args["task_name"])

    print(f"[scan] task={a.task_name} config={a.task_config} embodiment={embodiment_name} "
          f"seeds {a.start}..{a.start + a.count - 1}", flush=True)

    n_pass = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for i in range(a.count):
            seed = a.start + i
            try:
                task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
                task_env.play_once()
                ok = bool(task_env.plan_success and task_env.check_success())
                task_env.close_env()
                reason = "" if ok else "expert_failed"
            except UnStableError:
                task_env.close_env()
                ok, reason = False, "unstable"
            except Exception as e:                      # noqa: BLE001 - mirror eval's own catch-all
                task_env.close_env()
                ok, reason = False, f"{type(e).__name__}"
            n_pass += ok
            fh.write(f"{seed} {'PASS' if ok else 'FAIL'} {reason}\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"[scan] {i+1}/{a.count} scanned, {n_pass} pass", flush=True)

    print(f"[scan] done: {n_pass}/{a.count} pass -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
