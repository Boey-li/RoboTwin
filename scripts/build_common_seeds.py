"""Build one common eval scene set per task, shared by every rollout.

The eval harness accepts a candidate seed only if the scripted expert solves that
scene, and expert feasibility depends on the robot - so runs starting from the same
seed evaluated aloha and ARX on only 64-71% the same scenes. This intersects each
task's expert-PASS sets across both embodiments and draws a fixed random sample.

Selection uses ONLY the expert's verdict. No policy is consulted, so the sample
cannot favour any action space or embodiment by their measured performance.

The draw is a uniform sample without replacement under a declared RNG seed, so it
is reproducible and not a prefix - seeds map to structured scene variants (object
base, left/right side), and a contiguous block can over-represent some of them.

    python scripts/build_common_seeds.py --n 100 --rng-seed 20260821
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "seed_sets"
CONFIGS = {"aloha": "demo_clean", "arx": "demo_clean_arx",
           "franka": "demo_clean_franka", "piper": "demo_clean_piper",
           "ur5": "demo_clean_ur5"}


def load_pass(task: str, cfg: str) -> tuple[set[int], int]:
    """Union every scanned range for this task+config."""
    # NOT "{cfg}*.txt": that pattern makes demo_clean match demo_clean_arx and
    # silently unions the two embodiments into one pool.
    files = sorted(
        glob.glob(str(SEED_DIR / f"{task}__{cfg}.txt"))
        + glob.glob(str(SEED_DIR / f"{task}__{cfg}__*.txt"))
    )
    if not files:
        raise FileNotFoundError(f"no scan output for {task}/{cfg}")
    passed, scanned = set(), set()
    for f in files:
        for line in open(f, encoding="utf-8"):
            parts = line.split()
            if len(parts) < 2:
                continue
            s = int(parts[0])
            scanned.add(s)
            if parts[1] == "PASS":
                passed.add(s)
    return passed, len(scanned)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rng-seed", type=int, required=True,
                    help="declared RNG seed; recorded in the header of each output file")
    ap.add_argument("--tasks", nargs="+", default=["place_burger_fries", "press_stapler"])
    a = ap.parse_args()

    ok = True
    for task in a.tasks:
        pools, scans = {}, {}
        for label, cfg in CONFIGS.items():
            pools[label], scans[label] = load_pass(task, cfg)
        common = sorted(set.intersection(*(pools[k] for k in CONFIGS)))
        print(f"{task}:")
        for label in CONFIGS:
            print(f"    {label:6s} {len(pools[label]):4d} pass / {scans[label]:4d} scanned")
        print(f"    common {len(common):4d}")
        if len(common) < a.n:
            print(f"    *** only {len(common)} common seeds, need {a.n} - widen the scan ***")
            ok = False
            continue
        # Sample per task from its own pool, with a per-task-deterministic stream so
        # adding a task later cannot change another task's draw.
        rng = random.Random(f"{a.rng_seed}:{task}")
        sample = sorted(rng.sample(common, a.n))
        frac = 100.0 * a.n / len(common)
        out = SEED_DIR / f"common__{task}.txt"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"# {a.n} scenes drawn uniformly without replacement from {len(common)} "
                     f"expert-solvable-by-all-{len(CONFIGS)} seeds ({frac:.0f}% of pool)\n")
            fh.write(f"# rng=random.Random('{a.rng_seed}:{task}'); selection used expert "
                     f"feasibility only, no policy results\n")
            fh.write("\n".join(str(s) for s in sample) + "\n")
        print(f"    sampled {a.n} ({frac:.0f}% of pool) -> {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
