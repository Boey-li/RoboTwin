"""Build the T=32 scene-controlled assignment for one or more tasks.

The T=32 row of the volume x diversity grid holds SCENE COVERAGE FIXED across k.
Taking the first 40/k demos per robot would instead make coverage *fall* as diversity
rises (40 -> 26 -> 18 distinct scenes on burger), confounding the very contrast the
row exists to measure. So all four k values draw from ONE shared 40-scene set.

This is a partition problem, not an intersection one. The base set S (|S|=40) is a
subset of aloha's demo scenes; for each k it must split into per-robot parts where
every scene lands on a robot that actually collected it. The 4-way intersection is
only 25-29 scenes on three of the four tasks, so a naive "scenes everyone has"
approach cannot reach 40 -- but a partition can, because k=4 only needs 10 scenes
per robot, not 40 shared ones.

Feasibility per k is decided exactly by a min-cost assignment over 40 scenes x 40
slots (slots = capacity copies of each robot), cost 0 for a legal pair and 1 for an
illegal one. Optimal cost 0 <=> a valid partition exists.

    python scripts/build_scene_assignment.py --tasks open_laptop stack_bowls_two --rng-seed 20260901
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "controlB_assignment.json"
# Sizes match the published burger/press rows exactly: 40 total, k=3 is 14/13/13
# because 40 is not divisible by 3.
LAYOUT = {
    1: [("aloha_agilex", 40)],
    2: [("aloha_agilex", 20), ("arx_x5", 20)],
    3: [("aloha_agilex", 14), ("arx_x5", 13), ("franka", 13)],
    4: [("aloha_agilex", 10), ("arx_x5", 10), ("franka", 10), ("ur5", 10)],
}


def demo_seeds(task: str, emb: str) -> list[int]:
    """Episode index -> scene seed, in collection order."""
    return [int(x) for x in (ROOT / f"data/raw_data/{task}/{emb}/seed.txt").read_text().split()]


def solve(scenes: list[int], layout: list[tuple[str, int]], has: dict[str, set[int]]):
    """Exact partition of `scenes` onto robots at the given capacities, or None."""
    slots = [emb for emb, cap in layout for _ in range(cap)]
    if len(slots) != len(scenes):
        raise ValueError(f"{len(slots)} slots vs {len(scenes)} scenes")
    cost = np.array([[0 if s in has[e] else 1 for e in slots] for s in scenes], dtype=float)
    r, c = linear_sum_assignment(cost)
    if cost[r, c].sum() > 0:
        return None
    out: dict[str, list[int]] = {emb: [] for emb, _ in layout}
    for i, j in zip(r, c):
        out[slots[j]].append(scenes[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--rng-seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    assign = json.loads(OUT.read_text()) if OUT.exists() else {}
    for task in a.tasks:
        embs = sorted({e for layout in LAYOUT.values() for e, _ in layout})
        seq = {e: demo_seeds(task, e) for e in embs}
        has = {e: set(v) for e, v in seq.items()}

        # Prefer scenes many robots share -- they keep the k=4 partition feasible.
        # Ties broken by a declared RNG so the choice is reproducible, not a prefix.
        rng = random.Random(a.rng_seed)
        cand = sorted(has["aloha_agilex"], key=lambda s: (-sum(s in has[e] for e in embs), rng.random()))
        S = cand[: a.n]
        if len(S) < a.n:
            raise SystemExit(f"{task}: aloha has only {len(S)} demo scenes, need {a.n}")

        parts = {}
        for k, layout in LAYOUT.items():
            got = solve(S, layout, has)
            if got is None:
                raise SystemExit(f"{task}: no feasible k={k} partition of the chosen {a.n} scenes")
            parts[k] = got

        # Emit EPISODE INDICES, which is what TASK_CONFIGS consumes. Index != scene
        # seed here: these tasks skip failed seeds, so the sequences differ per robot.
        for k, got in parts.items():
            key = f"{task}|k{k}"
            assign[key] = {e: sorted(seq[e].index(s) for s in v) for e, v in got.items() if v}
            union = [seq[e][i] for e, idxs in assign[key].items() for i in idxs]
            assert len(set(union)) == a.n and set(union) == set(S), f"{key} scene set drifted"
        print(f"  {task}: 40-scene base set fixed across k=1..4  "
              f"(coverage {min(sum(s in has[e] for e in embs) for s in S)}-"
              f"{max(sum(s in has[e] for e in embs) for s in S)} robots/scene)")

    OUT.write_text(json.dumps(assign, indent=1))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
