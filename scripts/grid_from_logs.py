"""Authoritative grid reader: rates come from eval JOB LOGS, keyed by job name.

Why not read the result directories: a cell can accumulate more result dirs than shards
(a shard that dies after its sibling finished gets resubmitted by index), and nothing in
a result dir records WHICH shard it was -- _result.txt holds only a timestamp and a rate.
Pairing dirs by mtime then silently averages two copies of one shard.

Matching dirs to jobs via the log text does not work either: an eval log's tail contains
~17 different timestamp-format strings, so a regex grabs unrelated directories.

The job NAME is unambiguous: ev<task><arm><epoch><shard>, e.g. evsbs4420 =
stack_bowls_two / scene40k4 / ep2000 / shard 0. The log ends with the harness's own
"Final success rate: N/50". Keyed that way, duplicate runs of a shard collapse correctly
and a cell is complete only when BOTH shard 0 and shard 1 have a finished run.
"""
from __future__ import annotations

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = {"ol": "open_laptop", "sb": "stack_bowls_two"}
AB = {"s41": "scene40k1", "s42": "scene40k2", "g32": "gridk3T32", "s44": "scene40k4",
      "g16": "gridk1T64", "g26": "gridk2T64", "g36": "gridk3T64", "s84": "scene80k4",
      "k16": "aloha_agilex", "g21": "gridk2T160", "g31": "gridk3T160", "ct4": "cotrain4"}
CELLS = {(1, 32): "scene40k1", (2, 32): "scene40k2", (3, 32): "gridk3T32", (4, 32): "scene40k4",
         (1, 64): "gridk1T64", (2, 64): "gridk2T64", (3, 64): "gridk3T64", (4, 64): "scene80k4",
         (1, 160): "aloha_agilex", (2, 160): "gridk2T160", (3, 160): "gridk3T160", (4, 160): "cotrain4"}


# Arms whose policy_epoch_2000 was overwritten by a resume that lacked START_EPOCH
# (see ckpts/_contaminated_*). Only evals that finished INSIDE the bad window read the
# over-trained checkpoint: earlier ones read the original weights, and later ones come
# from the clean retrain. Excluding the whole pre-quarantine period instead would throw
# away valid measurements -- these arms' ep2000 cells were verified read-before-overwrite.
BAD_WINDOW = {
    ("open_laptop", "gridk1T64"):    (1788313680, 1788331740),
    ("open_laptop", "aloha_agilex"): (1788321000, 1788331740),
    ("stack_bowls_two", "cotrain4"): (1788320880, 1788331740),
}


def collect():
    """(task, arm, epoch, shard) -> (jobid, rate) keeping the newest finished run."""
    out = {}
    for log in glob.glob(f"{ROOT}/sbatch_logs/act_eval_ev*.out"):
        b = os.path.basename(log)
        m = re.match(r"act_eval_ev(ol|sb)([a-z0-9]{3})([26])(\d)_(\d+)\.out$", b)
        if not m:
            continue
        tp, ab, ep, sh, jid = m.groups()
        if ab not in AB:
            continue
        try:
            sz = os.path.getsize(log)
            with open(log, errors="ignore") as fh:
                fh.seek(max(0, sz - 20000))
                tail = fh.read()
        except OSError:
            continue
        fm = re.search(r"Final success rate:\s*(\d+)/(\d+)", tail)
        if not fm:
            continue                      # still running, or died before finishing
        succ, tot = int(fm.group(1)), int(fm.group(2))
        key = (TP[tp], AB[ab], int(ep) * 1000, int(sh))
        win = BAD_WINDOW.get((TP[tp], AB[ab]))
        if win and win[0] <= os.path.getmtime(log) <= win[1]:
            continue
        prev = out.get(key)
        if prev is None or int(jid) > prev[0]:
            out[key] = (int(jid), succ, tot)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=2000)
    ap.add_argument("--tasks", nargs="+", default=["open_laptop", "stack_bowls_two"])
    a = ap.parse_args()
    data = collect()
    for task in a.tasks:
        print(f"\n  {task}   -> Piper @ep{a.epoch}   [from job logs]")
        hdr = "k \\ T"   # not inline: an f-string expression cannot contain a backslash
        print(f"   {hdr:>6}" + "".join(f"{T:>15}" for T in (32, 64, 160)))
        for k in (1, 2, 3, 4):
            row = f"   {k:>6}"
            for T in (32, 64, 160):
                arm = CELLS[(k, T)]
                s0 = data.get((task, arm, a.epoch, 0))
                s1 = data.get((task, arm, a.epoch, 1))
                got = [x for x in (s0, s1) if x]
                if len(got) == 2:
                    succ = got[0][1] + got[1][1]; tot = got[0][2] + got[1][2]
                    row += f"{f'{100*succ/tot:.0f}% ({tot})':>15}"
                elif len(got) == 1:
                    succ, tot = got[0][1], got[0][2]
                    row += f"{f'{100*succ/tot:.0f}%* ({tot})':>15}"
                else:
                    row += f"{'-':>15}"
            print(row)
    print("\n  * only one of the two 50-scene shards has finished")


if __name__ == "__main__":
    main()
