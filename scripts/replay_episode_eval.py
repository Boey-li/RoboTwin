#!/usr/bin/env python3
"""Open-loop replay of one recorded episode inside the RoboTwin *eval* environment.

This answers "can the recorded trajectory finish the task under eval conditions?"
without any policy server: it rebuilds the eval scene from the episode's collection
seed, then feeds the episode's recorded joint targets straight into
``Base_Task.take_action`` -- the exact same entry point the policy rollout in
``scripts/eval_policy_xpolicylab.py`` uses -- and reports ``check_success()``.

Layout assumptions (XPolicyLab trajectory v1.0, as produced by
``scripts/process_data_xpolicylab.py``):

    data/<task_config>/<task_name>/<env_cfg_type>/
        seed.txt                      # collection seed per episode index
        data/episode_XXXXXXX.hdf5     # action/joint_states = [L arm(6), L grip, R arm(6), R grip]

``action/joint_states`` comes from RoboTwin's ``joint_action/vector`` with
``next_state_as_action`` alignment, i.e. action[i] is the joint state recorded at
i+1. That is precisely the absolute-qpos target ``take_action(..., 'qpos')``
expects, so replay is a straight feed-through.

Examples
--------
    python scripts/replay_episode_eval.py \
        --episode-file data/demo_clean/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5

    python scripts/replay_episode_eval.py --episode-file ... --no-video --hold 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
for _path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "scripts", ROBOTWIN_ROOT / "description" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_policy_xpolicylab import (  # noqa: E402
    class_decorator,
    get_camera_config,
    is_episode_end,
    load_task_args,
)

DEFAULT_EPISODE = (
    ROBOTWIN_ROOT / "data/demo_clean/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5"
)


# --------------------------------------------------------------------------- #
# episode file helpers
# --------------------------------------------------------------------------- #
def infer_dataset_layout(episode_file: Path) -> dict[str, Any]:
    """data/<task_config>/<task_name>/<env_cfg_type>/data/episode_XXXXXXX.hdf5."""
    episode_dir = episode_file.parent                 # .../<env_cfg_type>/data
    embodiment_dir = episode_dir.parent               # .../<env_cfg_type>
    task_dir = embodiment_dir.parent                  # .../<task_name>
    config_dir = task_dir.parent                      # .../<task_config>

    match = re.search(r"(\d+)", episode_file.stem)
    if match is None:
        raise ValueError(f"cannot parse an episode index out of {episode_file.name!r}")

    return {
        "episode_index": int(match.group(1)),
        "env_cfg_type": embodiment_dir.name,
        "task_name": task_dir.name,
        "task_config": config_dir.name,
        "seed_file": embodiment_dir / "seed.txt",
    }


def read_seed(seed_file: Path, episode_index: int) -> int:
    """collect_data.py replays seed_list[episode_idx], so index == episode index."""
    if not seed_file.is_file():
        raise FileNotFoundError(
            f"{seed_file} not found; pass --seed explicitly to pick the scene"
        )
    seeds = [int(token) for token in seed_file.read_text().split()]
    if episode_index >= len(seeds):
        raise IndexError(
            f"{seed_file} lists {len(seeds)} seeds, episode index {episode_index} is out of range"
        )
    return seeds[episode_index]


def load_episode(episode_file: Path, action_key: str) -> dict[str, Any]:
    with h5py.File(episode_file, "r") as handle:
        if action_key not in handle:
            raise KeyError(f"{episode_file} has no dataset {action_key!r}")
        actions = np.asarray(handle[action_key][()], dtype=np.float64)
        states = (
            np.asarray(handle["state/joint_states"][()], dtype=np.float64)
            if "state/joint_states" in handle
            else None
        )
        instruction = handle["instruction"][()] if "instruction" in handle else b""
        frequency = (
            int(handle["additional_info/frequency"][()])
            if "additional_info/frequency" in handle
            else None
        )
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    if actions.ndim != 2:
        raise ValueError(f"{action_key} must be 2-D, got shape {actions.shape}")
    return {
        "actions": actions,
        "states": states,
        "instruction": str(instruction),
        "frequency": frequency,
    }


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
def current_qpos(task_env) -> np.ndarray:
    return np.asarray(
        task_env.robot.get_left_arm_jointState() + task_env.robot.get_right_arm_jointState(),
        dtype=np.float64,
    )


def scene_fingerprint(task_env) -> dict[str, Any]:
    """Whatever per-seed scene randomisation the task exposes, for seed verification."""
    fingerprint: dict[str, Any] = {}
    for attribute in ("qpose_tag", "model_id", "arm_tag", "object_id"):
        if hasattr(task_env, attribute):
            value = getattr(task_env, attribute)
            fingerprint[attribute] = value.item() if hasattr(value, "item") else str(value)
    return fingerprint


def replay(
    task_env,
    args: dict[str, Any],
    *,
    seed: int,
    episode: dict[str, Any],
    hold: int,
    video_size: str | None,
    save_dir: Path,
) -> dict[str, Any]:
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)

    instruction = episode["instruction"]
    task_env.set_instruction(instruction=instruction)

    fingerprint = scene_fingerprint(task_env)
    if fingerprint:
        print(f"\033[36m[scene]\033[0m seed={seed} {fingerprint}")

    ffmpeg = None
    if task_env.eval_video_path is not None:
        video_path = Path(task_env.eval_video_path) / "replay.mp4"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", video_size, "-framerate", "10", "-i", "-",
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
        task_env._set_eval_video_ffmpeg(ffmpeg)

    actions = episode["actions"]
    action_dim = actions.shape[1]
    expected_dim = len(task_env.robot.get_left_arm_jointState()) + len(
        task_env.robot.get_right_arm_jointState()
    )
    if action_dim != expected_dim:
        raise ValueError(
            f"recorded action width {action_dim} != simulator width {expected_dim}; "
            "envs/_base_task.py slices by the SIMULATOR width and would silently mis-slice"
        )

    schedule = list(actions)
    if hold > 0:
        schedule.extend([actions[-1]] * hold)

    achieved: list[np.ndarray] = []
    success_step: int | None = None
    error: str | None = None

    try:
        for step, action in enumerate(schedule):
            if is_episode_end(task_env):
                break
            # take_action() streams self.now_obs into ffmpeg, so refresh it first --
            # this mirrors the get_obs() call in the eval rollout loop.
            if task_env.eval_video_path is not None:
                task_env.get_obs()
            task_env.take_action(np.asarray(action, dtype=np.float64), action_type="qpos")
            achieved.append(current_qpos(task_env))
            if task_env.eval_success and success_step is None:
                success_step = step + 1
                break
    except Exception:
        error = traceback.format_exc()
        print("\033[91m[replay] error during rollout:\033[0m")
        print(error)

    # eval_success is only latched inside take_action's inner loop; ask the task
    # directly as well so a success that lands on the final settle still counts.
    final_check = bool(task_env.check_success())
    success = bool(task_env.eval_success) or final_check

    if ffmpeg is not None:
        task_env._del_eval_video_ffmpeg()

    result: dict[str, Any] = {
        "task_name": args["task_name"],
        "task_config": args["task_config"],
        "seed": seed,
        "instruction": instruction,
        "scene": fingerprint,
        "recorded_steps": int(actions.shape[0]),
        "hold_steps": hold,
        "executed_steps": int(task_env.take_action_cnt),
        "step_limit": int(task_env.step_lim) if task_env.step_lim else None,
        "eval_success_flag": bool(task_env.eval_success),
        "check_success_final": final_check,
        "success": success,
        "success_step": success_step,
        "error": error,
    }

    states = episode["states"]
    if achieved and states is not None:
        achieved_arr = np.asarray(achieved)
        # take_action(i) drives toward action[i] == state[i+1]; compare like for like.
        reference = states[1 : 1 + len(achieved_arr)]
        n = min(len(reference), len(achieved_arr))
        if n:
            residual = achieved_arr[:n] - reference[:n]
            result["tracking_rms_rad"] = float(np.sqrt(np.mean(residual**2)))
            result["tracking_max_abs_rad"] = float(np.max(np.abs(residual)))

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "replay_result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-file", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--task-name", default=None, help="default: inferred from the episode path")
    parser.add_argument("--task-config", default=None, help="default: inferred from the episode path")
    parser.add_argument("--seed", type=int, default=None, help="default: seed.txt[episode index]")
    parser.add_argument("--action-key", default="action/joint_states")
    parser.add_argument("--hold", type=int, default=15, help="extra steps repeating the last action so the scene settles")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=None)
    parsed = parser.parse_args()

    episode_file = parsed.episode_file.resolve()
    if not episode_file.is_file():
        parser.error(f"episode file not found: {episode_file}")

    layout = infer_dataset_layout(episode_file)
    task_name = parsed.task_name or layout["task_name"]
    task_config = parsed.task_config or layout["task_config"]
    seed = parsed.seed if parsed.seed is not None else read_seed(layout["seed_file"], layout["episode_index"])

    episode = load_episode(episode_file, parsed.action_key)

    print(f"\033[34m[replay]\033[0m episode={episode_file}")
    print(f"\033[34m[replay]\033[0m task={task_name} config={task_config} env_cfg={layout['env_cfg_type']}")
    print(f"\033[34m[replay]\033[0m seed={seed} (episode index {layout['episode_index']})")
    print(f"\033[34m[replay]\033[0m actions={episode['actions'].shape} freq={episode['frequency']}")
    print(f"\033[34m[replay]\033[0m instruction={episode['instruction']!r}")

    usr_args = {
        "task_name": task_name,
        "task_config": task_config,
        "policy_name": "replay",
        "ckpt_setting": "recorded",
    }
    args, embodiment_name = load_task_args(usr_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    print(f"\033[34m[replay]\033[0m embodiment={embodiment_name}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    save_dir = parsed.save_dir or (
        ROBOTWIN_ROOT / "eval_result" / task_name / "replay" / task_config
        / f"episode{layout['episode_index']:07d}" / current_time
    )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    want_video = bool(args.get("eval_video_log")) and not parsed.no_video
    if want_video:
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = f"{camera_config['w']}x{camera_config['h']}"
        args["eval_video_save_dir"] = save_dir

    task_env = class_decorator(task_name)
    try:
        result = replay(
            task_env,
            args,
            seed=seed,
            episode=episode,
            hold=max(0, parsed.hold),
            video_size=video_size,
            save_dir=save_dir,
        )
    finally:
        try:
            task_env.close_env(clear_cache=True)
        except Exception:
            pass

    print("\n================ Replay result ================")
    for key, value in result.items():
        if key == "error":
            continue
        print(f"{key}: {value}")
    print(f"saved: {save_dir / 'replay_result.json'}")
    if result["success"]:
        print("\033[92mSUCCESS - the recorded trajectory finishes the task in the eval env.\033[0m")
    else:
        print("\033[91mFAIL - the recorded trajectory did not satisfy check_success() in the eval env.\033[0m")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    raise SystemExit(main())
