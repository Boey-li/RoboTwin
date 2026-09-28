"""Register the 12 volume x diversity grid arms for one or more tasks.

Grid axes: T = total TRAIN episodes (32/64/160), k = number of source embodiments.
ACT holds out 20% for validation, so a per-robot demo count d contributes int(0.8*d)
train episodes and T = k * int(0.8*d). k=3 cannot hit 32/64/160 exactly under that
constraint, so its rungs are 31/63/159 (1-3% low) -- the same rounding the published
burger and press grids use.

The T=32 row is scene-controlled via results/controlB_assignment.json: all four k
values draw from ONE shared 40-scene set, so scene coverage cannot drift with k.
Those arms carry explicit episode-index LISTS; every other arm carries a count.

Writes only keys it can fully satisfy -- an arm whose collection has not finished is
skipped and reported, never registered pointing at short data.

    python scripts/register_grid_arms.py --tasks open_laptop stack_bowls_two
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / "XPolicyLab/policy/ACT"
CFGS = ACT / "TASK_CONFIGS.json"
ASSIGN = ROOT / "results/controlB_assignment.json"
CAMS = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
EPLEN = 5000


def pd(bench: str, task: str, emb: str) -> str:
    return f"processed_data/{bench}/{task}/{emb}-ee"


def count_arms(task):
    """arm -> [(bench, emb, n_episodes_used, n_required_on_disk)]"""
    C = "demo_clean"
    return {
        "gridk1T64":  [("demo_clean_n200", "aloha_agilex", 80, 80)],
        "gridk2T64":  [(C, "aloha_agilex", 40, 40), (C, "arx_x5", 40, 40)],
        "gridk3T64":  [(C, "aloha_agilex", 27, 27), (C, "arx_x5", 27, 27), (C, "franka", 27, 27)],
        "scene80k4":  [(C, e, 20, 20) for e in ("aloha_agilex", "arx_x5", "franka", "ur5")],
        "cotrain4":   [(C, e, 50, 50) for e in ("aloha_agilex", "arx_x5", "franka", "ur5")],
        "gridk2T160": [("demo_clean_n200", "aloha_agilex", 100, 100),
                       ("demo_clean_n100_arx", "arx_x5", 100, 100)],
        "gridk3T160": [("demo_clean_n200", "aloha_agilex", 67, 67),
                       ("demo_clean_n100_arx", "arx_x5", 67, 67),
                       ("demo_clean_n70_franka", "franka", 67, 67)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    a = ap.parse_args()

    cfgs = json.loads(CFGS.read_text())
    assign = json.loads(ASSIGN.read_text())
    added, skipped = [], []

    for task in a.tasks:
        # --- T=32 row: explicit index lists from the shared 40-scene set ---
        for k, arm in ((1, "scene40k1"), (2, "scene40k2"), (3, "gridk3T32"), (4, "scene40k4")):
            part = assign[f"{task}|k{k}"]
            embs = [e for e in ("aloha_agilex", "arx_x5", "franka", "ur5") if e in part]
            cfgs[f"demo_clean-{task}-{arm}-ee"] = {
                "dataset_dir": [pd("demo_clean", task, e) for e in embs],
                "num_episodes": [sorted(part[e]) for e in embs],
                "episode_len": EPLEN, "camera_names": CAMS,
            }
            added.append(f"{task}/{arm} ({'+'.join(str(len(part[e])) for e in embs)})")

        # --- everything else: counts, gated on the data actually existing ---
        for arm, spec in count_arms(task).items():
            ok = True
            for bench, emb, _, need in spec:
                have = len(list((ACT / pd(bench, task, emb)).glob("episode_*.hdf5")))
                if have < need:
                    skipped.append(f"{task}/{arm} (needs {need} from {bench}/{emb}, has {have})")
                    ok = False
                    break
            if not ok:
                continue
            cfgs[f"demo_clean-{task}-{arm}-ee"] = {
                "dataset_dir": [pd(b, task, e) for b, e, _, _ in spec],
                "num_episodes": [n for _, _, n, _ in spec],
                "episode_len": EPLEN, "camera_names": CAMS,
            }
            added.append(f"{task}/{arm} ({'+'.join(str(n) for _, _, n, _ in spec)})")

        # --- the k=1 T=160 arm keeps its own bench prefix, matching burger/press ---
        need = 200
        have = len(list((ACT / pd("demo_clean_n200", task, "aloha_agilex")).glob("episode_*.hdf5")))
        if have >= need:
            cfgs[f"demo_clean_n200-{task}-aloha_agilex-ee"] = {
                "dataset_dir": [pd("demo_clean_n200", task, "aloha_agilex")],
                "num_episodes": [200], "episode_len": EPLEN, "camera_names": CAMS,
            }
            added.append(f"{task}/k1T160 (200)")
        else:
            skipped.append(f"{task}/k1T160 (needs 200 from demo_clean_n200, has {have})")

    tmp = CFGS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfgs, indent=1))
    json.loads(tmp.read_text())          # validate before replacing the live file
    tmp.replace(CFGS)
    print(f"  registered {len(added)}:")
    for x in added:
        print(f"    + {x}")
    print(f"  blocked on collection {len(skipped)}:")
    for x in skipped:
        print(f"    - {x}")


if __name__ == "__main__":
    main()
