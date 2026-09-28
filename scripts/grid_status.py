#!/usr/bin/env python3
"""Render the volume x diversity grid for both tasks.

Cell states:
  <pct>   eval complete  -> success rate into piper @ep2000 (n shown)
  ~pct    eval partial
  [train] checkpoint exists, eval not started
  (run)   training in flight
  (data)  waiting on demo collection
  --      not built
"""
import os, subprocess, glob

TASKS = ["place_burger_fries", "press_stapler"]
# (k, T_nominal) -> (ckpt_arm_name_template, actual_T)
CELLS = {
 (1,32):  ("demo_clean-{t}-scene40k1-ee-0", 32),
 (2,32):  ("demo_clean-{t}-scene40k2-ee-0", 32),
 (3,32):  ("demo_clean-{t}-gridk3T32-ee-0", 31),
 (4,32):  ("demo_clean-{t}-scene40k4-ee-0", 32),
 (1,64):  ("demo_clean-{t}-gridk1T64-ee-0", 64),
 (2,64):  ("demo_clean-{t}-gridk2T64-ee-0", 64),
 (3,64):  ("demo_clean-{t}-gridk3T64-ee-0", 63),
 (4,64):  ("demo_clean-{t}-scene80k4-ee-0", 64),
 (1,160): ("demo_clean_n200-{t}-aloha_agilex-ee-0", 160),
 (2,160): ("demo_clean-{t}-gridk2T160-ee-0", 160),
 (3,160): ("demo_clean-{t}-gridk3T160-ee-0", 159),
 (4,160): ("demo_clean-{t}-cotrain4-ee-0", 160),
}

def queue():
    try:
        return subprocess.run(["timeout","45","squeue","-u",os.environ.get("USER",""),"-h","-o","%j|%T"],
                              capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""

def evald(t, ckpt):
    """success/total into piper at ep2000 (falls back to final-epoch dir)."""
    for tag in (ckpt.replace("-ee-0","ep2000-ee-0"), ckpt):
        d = f"eval_result/{t}/ACT/demo_clean_piper/{tag}"
        s = len(glob.glob(d+"/*/*-success.mp4")); f = len(glob.glob(d+"/*/*-failure.mp4"))
        if s+f: return s, s+f, tag.endswith("ep2000-ee-0")
    return None

def cell(t, k, T, q):
    arm, actual = CELLS[(k,T)]
    ck = arm.format(t=t)
    e = evald(t, ck)
    if e:
        s, n, is2k = e
        mark = "" if is2k else "*"
        return f"{100*s/n:.0f}%{mark} ({n})" if n >= 100 else f"~{100*s/n:.0f}% ({n})"
    task_tag = t[:2]
    short = ck.split("-")[-3] if "grid" in ck else ""
    if any(f"g_{task_tag}_{short.replace('grid','')}|RUNNING" in ln for ln in q.split("\n")):
        return f"(run) T={actual}"
    if os.path.exists(f"ckpts/ACT/{t}/{ck}/policy_last.ckpt"):
        return f"[train] T={actual}"
    if T == 160 and k in (2,3):
        return "(data)"
    return f"(run) T={actual}"

q = queue()
for t in TASKS:
    print(f"\n  {t}   -> Piper @ep2000   (* = final-epoch only)")
    hdr = "k \\ T"
    print(f"  {hdr:>6} {'32':>14} {'64':>14} {'160':>14}")
    for k in (1,2,3,4):
        row = "".join(f"{cell(t,k,T,q):>15}" for T in (32,64,160))
        print(f"  {k:>6}{row}")
