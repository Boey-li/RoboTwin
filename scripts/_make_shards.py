"""Split a pinned seed list into S disjoint shards for parallel evaluation.

ACT cannot use the harness's --eval_batch/--num_workers path: it keeps per-episode
state (self.t, all_time_actions, obs_cache) on one shared model instance and does not
implement prepare_case, so concurrent episodes would corrupt temporal aggregation.
Independent sequential jobs over disjoint seed slices give the same speedup with no
shared state.
"""
import sys, os
task, S = sys.argv[1], int(sys.argv[2])
src = f"seed_sets/common__{task}.txt"
seeds = [l.split()[0] for l in open(src) if l.strip() and not l.lstrip().startswith("#")]
assert len(seeds) == 100, f"{src} has {len(seeds)} seeds, expected 100"
out = []
for i in range(S):
    part = seeds[i::S]                      # stride, so每 shard spans the whole range
    p = f"seed_sets/shard__{task}__{i}of{S}.txt"
    with open(p, "w") as fh:
        fh.write(f"# shard {i+1}/{S} of {src}: {len(part)} seeds\n")
        fh.write("\n".join(part) + "\n")
    out.append((p, len(part)))
tot = sum(n for _, n in out)
assert tot == 100, f"shards sum to {tot}, not 100"
union = set()
for p, _ in out:
    union |= {l.split()[0] for l in open(p) if not l.lstrip().startswith("#") and l.strip()}
assert union == set(seeds), "shard union != original seed set"
for p, n in out:
    print(f"  {p}  {n} seeds")
