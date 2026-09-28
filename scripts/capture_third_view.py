"""Render the observer (third-person) camera for one task+embodiment.

The official ARX raw data was collected with third_view disabled, so there is no
stored third-person frame for it - envs/camera/camera.py:220 builds the observer
camera regardless, and get_observer_rgb() can be called at any time. This grabs it
at the home pose and again after the scripted expert finishes, so the two
embodiments can be compared on the same scene.

    python scripts/capture_third_view.py --task_name adjust_bottle \
        --task_config demo_clean --seed 100008 --out_prefix /tmp/aloha
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_policy_xpolicylab import class_decorator, load_task_args  # noqa: E402


def save(img, path):
    cv2.imwrite(path, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    print(f"  wrote {path} {np.asarray(img).shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--seed", type=int, default=100008)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--max_tries", type=int, default=25)
    ap.add_argument("--skip-expert", action="store_true",
                    help="only capture the home pose; avoids play_once(), which needs "
                         "curobo (compiled for sm_86 and unusable on other archs)")
    a = ap.parse_args()

    usr_args = {"task_name": a.task_name, "task_config": a.task_config,
                "policy_name": "capture", "seed": 0, "ckpt_setting": "capture"}
    args, embodiment_name = load_task_args(usr_args)
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args["eval_mode"] = True
    print(f"[capture] {a.task_name} / {a.task_config} / {embodiment_name} seed={a.seed}", flush=True)

    task_env = class_decorator(args["task_name"])

    # Not every seed yields a stable scene - _base_task raises UnStableError when an
    # object is still settling. Walk forward until one builds; purely cosmetic which
    # seed we land on, since this frame only illustrates the task.
    for k in range(a.seed, a.seed + a.max_tries):
        try:
            task_env.setup_demo(now_ep_num=0, seed=k, is_test=True, **args)
            if k != a.seed:
                print(f"  (seed {a.seed} unusable; used {k})", flush=True)
            break
        except Exception as e:                              # noqa: BLE001
            print(f"  seed {k}: {type(e).__name__}", flush=True)
            try:
                task_env.close_env()
            except Exception:
                pass
    else:
        print(f"  GIVING UP after {a.max_tries} seeds")
        return
    save(task_env.cameras.get_observer_rgb(), f"{a.out_prefix}_home.png")
    if a.skip_expert:
        task_env.close_env()
        return
    try:
        task_env.play_once()
        save(task_env.cameras.get_observer_rgb(), f"{a.out_prefix}_end.png")
    except Exception as e:                                  # noqa: BLE001
        print(f"  expert rollout failed ({type(e).__name__}); home frame still saved")
    task_env.close_env()


if __name__ == "__main__":
    main()
