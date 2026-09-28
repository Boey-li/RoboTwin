"""Render the k x T transfer grid for open_laptop / stack_bowls_two.

Each cell is one training arm evaluated into HELD-OUT PIPER on that task's 100 pinned
scenes, split into two disjoint 50-scene shards. A shard writes _result.txt only when
all 50 finish, so the cell rate is exactly the mean of the two shard rates -- no
weighting needed, and partially-run shards are excluded rather than counted short.

Directories WITHOUT _result.txt are husks left by preempted jobs and are ignored, the
same rule merge_xemb_results.py applies. Counting directories instead would mark a cell
done the moment it launched.

    python scripts/grid_status2.py [--epoch 2000|6000]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (k, T) -> (arm, bench). k=3 rungs are T=31/63/159: k*int(0.8d)==T has no exact
# solution at k=3 for 32/64/160.
CELLS = {
    (1, 32):  ("scene40k1",    "demo_clean"),
    (2, 32):  ("scene40k2",    "demo_clean"),
    (3, 32):  ("gridk3T32",    "demo_clean"),
    (4, 32):  ("scene40k4",    "demo_clean"),
    (1, 64):  ("gridk1T64",    "demo_clean"),
    (2, 64):  ("gridk2T64",    "demo_clean"),
    (3, 64):  ("gridk3T64",    "demo_clean"),
    (4, 64):  ("scene80k4",    "demo_clean"),
    (1, 160): ("aloha_agilex", "demo_clean_n200"),
    (2, 160): ("gridk2T160",   "demo_clean"),
    (3, 160): ("gridk3T160",   "demo_clean"),
    (4, 160): ("cotrain4",     "demo_clean"),
}


def cell(task: str, arm: str, bench: str, epoch: int):
    """(rate_pct, n_scenes, in_flight) -- quick view only.

    NOT authoritative: a result dir records no shard id, so this pairs dirs by mtime and
    cannot detect a duplicated shard. Use scripts/grid_from_logs.py, which keys on the
    eval job name (ev<task><arm><epoch><shard>) and is shard-exact.
    """
    name = f"{bench}-{task}-{arm}-ee-0" if epoch == 6000 else f"{bench}-{task}-{arm}ep{epoch}-ee-0"
    base = f"{ROOT}/eval_result/{task}/ACT/demo_clean_piper/{name}"
    rates, live = [], 0
    for d in sorted(glob.glob(base + "/*/"), key=os.path.getmtime):
        r = d + "_result.txt"
        if os.path.exists(r):
            tail = [x for x in open(r).read().split("\n") if x.strip()]
            rates.append(float(tail[-1]))
        else:
            live = max(live, len(glob.glob(d + "*-success.mp4")) + len(glob.glob(d + "*-failure.mp4")))
    if len(rates) >= 2:
        return 100.0 * sum(rates[:2]) / 2, 100, 0
    if len(rates) == 1:
        return 100.0 * rates[0], 50, live
    return None, 0, live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=2000)
    ap.add_argument("--tasks", nargs="+", default=["open_laptop", "stack_bowls_two"])
    a = ap.parse_args()
    for task in a.tasks:
        print(f"\n  {task}   -> Piper @ep{a.epoch}")
        hdr = "k \\ T"
        print(f"   {hdr:>6}" + "".join(f"{T:>15}" for T in (32, 64, 160)))
        for k in (1, 2, 3, 4):
            row = f"   {k:>6}"
            for T in (32, 64, 160):
                arm, bench = CELLS[(k, T)]
                rate, n, live = cell(task, arm, bench, a.epoch)
                if rate is None:
                    row += f"{('~' + str(live) + '/50' if live else '-'):>15}"
                elif n == 100:
                    row += f"{f'{rate:.0f}% (100)':>15}"
                else:
                    row += f"{f'{rate:.0f}%* (50)':>15}"
            print(row)
    print("\n  * one shard only (50 scenes); ~n/50 = shard in flight; - = not started")


if __name__ == "__main__":
    main()
