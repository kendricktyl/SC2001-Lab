"""
SC2001 Project 1 - experiment driver.

Usage:
    python3 experiments.py exp1      # fixed S, varying n
    python3 experiments.py exp2      # fixed n, varying S
    python3 experiments.py exp3      # optimal S across several n
    python3 experiments.py exp4      # pure vs hybrid on n = 10,000,000
    python3 experiments.py all

Every experiment appends its results to results/<name>.json so that the
plotting script and the report can be regenerated without re-running the
(expensive) benchmarks.

Timing notes
------------
* CPU time is measured with time.process_time() (user+sys CPU of this
  process), which is what the project asks for; wall time is recorded too.
* The array copy handed to the sorter is made OUTSIDE the timed region.
* Key-comparison counts are deterministic for a given dataset, so they need
  no repetition; timings are averaged over `repeats` runs.
"""

import gc
import json
import os
import sys
import time

from hybrid_sort import (
    generate_dataset,
    sort_hybrid,
    sort_hybrid_shift,
    sort_pure,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 2001            # fixed seed => every experiment is reproducible


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def save(name, payload):
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> wrote {path}")


def timed_sort(data, S=None, shift=False, repeats=1):
    """Sort a copy of `data`; return (comparisons, best cpu time, best wall time).

    S=None means the original merge sort; `shift` selects the shift-based
    leaf variant of the hybrid sort (ignored when S is None). The minimum
    over `repeats` runs is reported because the minimum is the least noisy
    estimator of the true cost (noise from the OS can only ever add time).
    """
    comparisons = None
    best_cpu = float("inf")
    best_wall = float("inf")

    for _ in range(repeats):
        A = data[:]                     # fresh unsorted copy, outside the timer
        gc.collect()
        gc.disable()
        c0, w0 = time.process_time(), time.perf_counter()
        if S is None:
            c = sort_pure(A)
        elif shift:
            c = sort_hybrid_shift(A, S)
        else:
            c = sort_hybrid(A, S)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        gc.enable()

        assert comparisons is None or comparisons == c, "non-deterministic count"
        comparisons = c
        best_cpu = min(best_cpu, cpu)
        best_wall = min(best_wall, wall)
        del A

    return comparisons, best_cpu, best_wall


# --------------------------------------------------------------------------
# Experiment 1 - fixed S, varying n
# --------------------------------------------------------------------------

EXP1_SIZES = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
EXP1_S_VALUES = [8, 32]                 # 8 ~ comparison-optimal, 32 ~ a typical library cutoff


def exp1():
    print("EXP 1: fixed S, varying n")
    rows = []
    for n in EXP1_SIZES:
        data = generate_dataset(n, seed=SEED)
        repeats = 3 if n <= 1_000_000 else 1

        c, cpu, wall = timed_sort(data, S=None, repeats=repeats)
        rows.append(dict(n=n, S=None, algo="pure", comparisons=c, cpu=cpu, wall=wall))
        print(f"  n={n:>9}  pure      comps={c:>12}  cpu={cpu:8.3f}s")

        for S in EXP1_S_VALUES:
            c, cpu, wall = timed_sort(data, S=S, repeats=repeats)
            rows.append(dict(n=n, S=S, algo="hybrid", comparisons=c, cpu=cpu, wall=wall))
            print(f"  n={n:>9}  hybrid S={S:<3} comps={c:>12}  cpu={cpu:8.3f}s")

        del data
        gc.collect()
        save("exp1", rows)
    return rows


# --------------------------------------------------------------------------
# Experiment 2 - fixed n, varying S
# --------------------------------------------------------------------------

EXP2_N = 1_000_000
EXP2_S_RANGE = list(range(1, 101))


def exp2():
    print(f"EXP 2: n = {EXP2_N} fixed, S = 1..100")
    data = generate_dataset(EXP2_N, seed=SEED)
    rows = []
    for S in EXP2_S_RANGE:
        c, cpu, wall = timed_sort(data, S=S, repeats=1)
        rows.append(dict(n=EXP2_N, S=S, comparisons=c, cpu=cpu, wall=wall))
        print(f"  S={S:>3}  comps={c:>12}  cpu={cpu:7.3f}s", flush=True)
        if S % 10 == 0:
            save("exp2", rows)
    save("exp2", rows)
    return rows


# --------------------------------------------------------------------------
# Experiment 3 - optimal S across dataset scales
# --------------------------------------------------------------------------

# Full sweep for the small sizes, a coarse (but still dense near the optimum)
# grid for the two largest sizes, where each run is expensive.
EXP3_PLAN = [
    (1_000,      list(range(1, 101))),
    (10_000,     list(range(1, 101))),
    (100_000,    list(range(1, 101))),
    (1_000_000,  None),                 # reuse exp2
    (5_000_000,  [1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 100]),
    (10_000_000, [1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 100]),
]


def exp3():
    print("EXP 3: sweep S for several n")
    rows = []
    for n, s_values in EXP3_PLAN:
        if s_values is None:
            continue                    # exp2 already covers n = 1,000,000
        data = generate_dataset(n, seed=SEED)
        repeats = 3 if n <= 100_000 else 1
        for S in s_values:
            c, cpu, wall = timed_sort(data, S=S, repeats=repeats)
            rows.append(dict(n=n, S=S, comparisons=c, cpu=cpu, wall=wall))
            print(f"  n={n:>9} S={S:>3}  comps={c:>12}  cpu={cpu:7.3f}s", flush=True)
        del data
        gc.collect()
        save("exp3", rows)
    return rows


# --------------------------------------------------------------------------
# Experiment 4 - head-to-head on 10 million integers
# --------------------------------------------------------------------------

EXP4_N = 10_000_000
EXP4_REPEATS = 2


def exp4(optimal_s=None):
    if optimal_s is None:
        optimal_s = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"EXP 4: n = {EXP4_N}, pure vs hybrid (S = {optimal_s})")
    data = generate_dataset(EXP4_N, seed=SEED)
    rows = []

    c, cpu, wall = timed_sort(data, S=None, repeats=EXP4_REPEATS)
    rows.append(dict(algo="pure merge sort", S=1, comparisons=c, cpu=cpu, wall=wall))
    print(f"  pure       comps={c:>13}  cpu={cpu:8.3f}s  wall={wall:8.3f}s")

    for S in sorted({optimal_s, 8, 16, 32}):
        c, cpu, wall = timed_sort(data, S=S, repeats=EXP4_REPEATS)
        rows.append(dict(algo=f"hybrid (S={S})", S=S, comparisons=c, cpu=cpu, wall=wall))
        print(f"  hybrid S={S:<3} comps={c:>13}  cpu={cpu:8.3f}s  wall={wall:8.3f}s")
        save("exp4", rows)

    save("exp4", rows)
    return rows


# --------------------------------------------------------------------------
# Experiment 5 - CPU time vs S (repeated, because timings are noisy)
# --------------------------------------------------------------------------

EXP5_N = 1_000_000
EXP5_S = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]
EXP5_REPEATS = 3


def exp5():
    """Comparison counts are exact after one run; CPU times are not.

    Each configuration is run EXP5_REPEATS times and the MINIMUM is kept:
    scheduling noise can only ever make a run slower, so the minimum is the
    cleanest estimate of the intrinsic cost.
    """
    print(f"EXP 5: CPU time vs S at n = {EXP5_N} (min of {EXP5_REPEATS})")
    data = generate_dataset(EXP5_N, seed=SEED)
    rows = []

    for S in EXP5_S:
        c, cpu, wall = timed_sort(data, S=S, repeats=EXP5_REPEATS)
        cs, cpus, walls = timed_sort(data, S=S, shift=True, repeats=EXP5_REPEATS)
        assert c == cs, "leaf variants must agree on comparison count"
        rows.append(dict(n=EXP5_N, S=S, comparisons=c,
                         cpu_swap=cpu, wall_swap=wall,
                         cpu_shift=cpus, wall_shift=walls))
        print(f"  S={S:>3}  comps={c:>12}  cpu(swap)={cpu:7.3f}s  "
              f"cpu(shift)={cpus:7.3f}s", flush=True)
        save("exp5", rows)
    return rows


# --------------------------------------------------------------------------

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("exp1", "all"):
        exp1()
    if which in ("exp2", "all"):
        exp2()
    if which in ("exp3", "all"):
        exp3()
    if which in ("exp5", "all"):
        exp5()
    if which in ("exp4", "all"):
        exp4()
    print(f"done in {time.time() - t0:.1f}s")
