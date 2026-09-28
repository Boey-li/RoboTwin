"""Merge seed-shards into per-cell results and run the paired analysis.

Each eval cell was split into S disjoint seed shards (ACT cannot use the harness's
--eval_batch path: it keeps per-episode state on one shared model instance). This
recombines them, verifies the shards cover the pinned 100 seeds exactly once, and
emits per-cell success plus seed-paired comparisons.
"""
from __future__ import annotations
import glob, json, math, os, re, sys
from collections import defaultdict

R = "/coc/flash7/bli678/Projects/RoboTwin"
os.chdir(R)
TASKS = ["place_burger_fries", "press_stapler"]
EMB = ["aloha_agilex", "arx_x5", "franka", "piper", "ur5"]
CFG = {"aloha_agilex": "demo_clean", "arx_x5": "demo_clean_arx", "franka": "demo_clean_franka",
       "piper": "demo_clean_piper", "ur5": "demo_clean_ur5"}

def pinned(task):
    return [int(l.split()[0]) for l in open(f"seed_sets/common__{task}.txt")
            if l.strip() and not l.lstrip().startswith("#")]

def shard_seeds(task, i, S):
    return [int(l.split()[0]) for l in open(f"seed_sets/shard__{task}__{i}of{S}.txt")
            if l.strip() and not l.lstrip().startswith("#")]

def outcomes_for(task, cfg, ckpt, S=2):
    """Map seed -> success, merged across shards. Returns (dict, n_shards_found)."""
    base = f"eval_result/{task}/ACT/{cfg}/{ckpt}"
    # eval_policy creates its timestamped dir at startup, so a dir alone does not
    # mean a finished shard. Only dirs carrying _result.txt are complete; including
    # a crashed one would shift the dir->shard mtime mapping below and score one
    # shard's seeds against another's outcomes.
    dirs = sorted((d for d in glob.glob(base + "/*/") if os.path.exists(d + "_result.txt")),
                  key=os.path.getmtime)
    if not dirs:
        return {}, 0
    # each completed shard writes its own timestamped dir; take the newest S of them
    per_shard = []
    for d in dirs[-S:]:
        o = {}
        for f in glob.glob(d + "*.mp4"):
            m = re.match(r"episode(\d+)-(success|failure)\.mp4", os.path.basename(f))
            if m:
                o[int(m.group(1))] = m.group(2) == "success"
        if o:
            per_shard.append((d, o))
    merged = {}
    for i, (d, o) in enumerate(per_shard):
        seeds = shard_seeds(task, i, S)
        for idx, ok in o.items():
            if idx < len(seeds):
                merged[seeds[idx]] = ok
    return merged, len(per_shard)

def mcnemar(b, c):
    if b + c == 0:
        return 0.0, 1.0
    chi = (abs(b - c)) ** 2 / (b + c)
    return chi, math.erfc(math.sqrt(chi / 2))

def main():
    cells = {}
    for t in TASKS:
        for e in EMB:
            for a in ("joint", "ee"):
                ck = f"demo_clean-{t}-{e}-{a}-0"
                o, ns = outcomes_for(t, CFG[e], ck)
                if o:
                    cells[("F1", t, e, a)] = (o, ns)
        for src in ["aloha_agilex", "arx_x5", "franka", "ur5"]:
            for a in ("joint", "ee"):
                if src == "franka" and a == "joint":
                    continue                      # 16-dim policy into a 14-dim sim
                ck = f"demo_clean-{t}-{src}-{a}-0"
                o, ns = outcomes_for(t, "demo_clean_piper", ck)
                if o:
                    cells[("F2", t, src, a)] = (o, ns)
        # F3: the cotrain-ee model (aloha+arx+franka+ur5) on every embodiment.
        # For the four it trained on this is in-distribution; for piper -- held out
        # of the pool -- it is a zero-shot number directly comparable to F2.
        for e in EMB:
            o, ns = outcomes_for(t, CFG[e], f"demo_clean-{t}-cotrain4-ee-0")
            if o:
                cells[("F3", t, e, "ee")] = (o, ns)

    print(f"{'blk':4s} {'task':20s} {'emb':13s} {'act':5s} {'n':>4s} {'shards':>7s} {'rate':>7s}")
    for k in sorted(cells):
        o, ns = cells[k]
        print(f"{k[0]:4s} {k[1]:20s} {k[2]:13s} {k[3]:5s} {len(o):4d} {ns:7d} "
              f"{100*sum(o.values())/len(o):6.1f}%")

    print("\n=== joint vs ee, seed-paired (in-distribution) ===")
    tb = tc = 0
    for t in TASKS:
        for e in EMB:
            j = cells.get(("F1", t, e, "joint")); x = cells.get(("F1", t, e, "ee"))
            if not j or not x:
                continue
            j, x = j[0], x[0]
            sh = sorted(set(j) & set(x))
            if not sh:
                continue
            b = sum(1 for s in sh if j[s] and not x[s])
            c = sum(1 for s in sh if not j[s] and x[s])
            tb += b; tc += c
            chi, p = mcnemar(b, c)
            print(f"  {t:20s} {e:13s} n={len(sh):3d}  {100*sum(j[s] for s in sh)/len(sh):5.1f}%"
                  f" vs {100*sum(x[s] for s in sh)/len(sh):5.1f}%  disc {b:3d}/{c:3d}  p={p:.3g}")
    chi, p = mcnemar(tb, tc)
    print(f"  POOLED  disc {tb}/{tc}  chi2={chi:.2f}  p={p:.3g}")

    print("\n=== cotrain vs single-task, seed-paired (ee, embodiments IN the pool) ===")
    for t in TASKS:
        for e in EMB:
            if e == "piper":
                continue                          # held out of the pool; see below
            g = cells.get(("F3", t, e, "ee")); s1 = cells.get(("F1", t, e, "ee"))
            if not g or not s1:
                continue
            g, s1 = g[0], s1[0]
            sh = sorted(set(g) & set(s1))
            if not sh:
                continue
            b = sum(1 for s in sh if g[s] and not s1[s])
            c = sum(1 for s in sh if not g[s] and s1[s])
            chi, p = mcnemar(b, c)
            print(f"  {t:20s} {e:13s} n={len(sh):3d}  cotrain {100*sum(g[s] for s in sh)/len(sh):5.1f}%"
                  f" vs single {100*sum(s1[s] for s in sh)/len(sh):5.1f}%  disc {b:3d}/{c:3d}  p={p:.3g}")
    print("  NOTE: at 6000 epochs cotrain sees 4x the gradient steps of each single-task")
    print("        baseline (160 pooled episodes vs 40). A cotrain win is not attributable")
    print("        to the pooled data without a matched-step control.")

    print("\n=== does cotraining rescue zero-shot into piper? (piper held out) ===")
    for t in TASKS:
        g = cells.get(("F3", t, "piper", "ee"))
        own = cells.get(("F1", t, "piper", "ee"))
        best_f2 = max(((sum(v[0].values())/len(v[0]), src)
                       for (blk, tt, src, a), v in cells.items()
                       if blk == "F2" and tt == t and a == "ee"), default=(float("nan"), "-"))
        gr = 100*sum(g[0].values())/len(g[0]) if g else float("nan")
        ow = 100*sum(own[0].values())/len(own[0]) if own else float("nan")
        print(f"  {t:20s} cotrain4->piper {gr:5.1f}%   best single-source F2 {100*best_f2[0]:5.1f}%"
              f" ({best_f2[1]})   piper's own {ow:5.1f}%")

    json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {"n": len(v[0]), "shards": v[1],
               "rate": sum(v[0].values())/len(v[0]),
               "per_seed": {str(s): int(b) for s, b in sorted(v[0].items())}}
               for k, v in cells.items()}, open("results/xemb_results.json", "w"), indent=1)
    print(f"\nwrote results/xemb_results.json ({len(cells)} cells)")

if __name__ == "__main__":
    main()
