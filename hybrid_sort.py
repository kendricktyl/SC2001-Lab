"""
SC2001 - Algorithm Design and Analysis
Project 1: Integration of Merge Sort & Insertion Sort

Index convention used throughout: A[start:end], start INCLUSIVE, end
EXCLUSIVE, mid = start + (end - start) // 2.

KEY COMPARISON COUNTING
-----------------------
Every routine RETURNS the number of key comparisons it performed.
A "key comparison" is one evaluation of a comparison operator between two
ARRAY ELEMENTS, e.g. `A[j] > A[j + 1]` or `Bl[i] <= Br[j]`.

Loop-bound / index tests such as `j >= 0` or `i < len(Bl)` are NOT counted.
Python's `and` short-circuits, so writing the index test first (exactly as
the lecture pseudocode does) means the key comparison is only reached - and
only counted - when the index test passes. The counters below are therefore
exact, not estimates.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------
# 1. Insertion Sort
# --------------------------------------------------------------------------


def _insertion_sort(A, start, end, shift):
    """Shared body for insertion_sort / insertion_sort_shift.

    `shift` picks the move strategy once per call (not per comparison, so
    CPU-time comparisons between the two stay meaningful): swap-based moves
    a pair on every step (three list operations), shift-based holds the
    key in a register and shifts larger elements right by one (a single
    store). Both produce identical KEY COMPARISON counts - only the
    constant factor of data movement differs.
    """
    comparisons = 0

    if shift:
        for i in range(start + 1, end):
            key = A[i]
            j = i - 1
            while j >= start:
                comparisons += 1          # the test A[j] > key happens now
                if A[j] > key:
                    A[j + 1] = A[j]       # shift right, no swap
                    j -= 1
                else:
                    break
            A[j + 1] = key
    else:
        for i in range(start + 1, end):
            j = i - 1
            while j >= start:
                comparisons += 1          # the test A[j] > A[j+1] happens now
                if A[j] > A[j + 1]:
                    A[j], A[j + 1] = A[j + 1], A[j]
                    j -= 1
                else:
                    break                 # element has reached its place

    return comparisons


def insertion_sort(A, start, end):
    """Sort A[start:end] in place using Insertion Sort (swap-based).

    In-place, stable, O(m^2) comparisons, O(m^2) swaps, close to linear on
    nearly-sorted input.

    Returns
    -------
    int : number of key comparisons performed.
    """
    return _insertion_sort(A, start, end, shift=False)


def insertion_sort_shift(A, start, end):
    """Same algorithm and key-comparison count as insertion_sort, but with
    fewer data movements (see _insertion_sort) - the honest choice for CPU
    time comparisons.

    Returns
    -------
    int : number of key comparisons performed (identical to insertion_sort).
    """
    return _insertion_sort(A, start, end, shift=True)


# --------------------------------------------------------------------------
# 2. Merging (shared by both merge sorts)
# --------------------------------------------------------------------------


def merge(A, start, mid, end):
    """Merge the two sorted runs A[start:mid] and A[mid:end] in place.

    The `<=` (rather than `<`) below is what makes the merge - and therefore
    the whole merge sort - STABLE: ties are resolved in favour of the left
    run. Requires O(m) extra memory for Bl and Br, so merge sort is not
    in-place.

    Returns
    -------
    int : number of key comparisons performed.
    """
    comparisons = 0

    Bl = A[start:mid]                     # left run
    Br = A[mid:end]                       # right run
    n_left = len(Bl)

    i = 0                                 # cursor into Bl
    k = start                             # write cursor into A

    for j in range(len(Br)):              # place every element of Br
        right = Br[j]
        # Flush all remaining Bl entries that are <= the current Br entry.
        while i < n_left:
            comparisons += 1              # the test Bl[i] <= Br[j] happens now
            if Bl[i] <= right:
                A[k] = Bl[i]
                i += 1
                k += 1
            else:
                break
        A[k] = right
        k += 1

    # Bl may still hold elements larger than everything in Br: no more
    # comparisons are needed, they are simply copied across.
    while i < n_left:
        A[k] = Bl[i]
        i += 1
        k += 1

    return comparisons


# --------------------------------------------------------------------------
# 3. Original (pure) Merge Sort
# --------------------------------------------------------------------------


def merge_sort(A, start, end):
    """Sort A[start:end] in place with the textbook recursive Merge Sort.

    Returns
    -------
    int : number of key comparisons performed.
    """
    if end - start <= 1:
        return 0

    mid = start + (end - start) // 2
    comparisons = merge_sort(A, start, mid)
    comparisons += merge_sort(A, mid, end)
    comparisons += merge(A, start, mid, end)
    return comparisons


# --------------------------------------------------------------------------
# 4. Hybrid Merge Sort + Insertion Sort
# --------------------------------------------------------------------------


def _hybrid_merge_sort(A, start, end, S, shift):
    """Shared body for hybrid_merge_sort / hybrid_merge_sort_shift.

    Once a subproblem has size (end - start) <= S the recursion is cut off
    and Insertion Sort finishes that block; otherwise the subproblem is split
    and merged exactly as in the pure version. `shift` selects the leaf
    variant (see _insertion_sort) and is threaded through the recursion
    rather than checked per comparison.

    Note S = 1 reproduces the original merge sort: a block of size 1 is
    already sorted and Insertion Sort on it costs zero comparisons.
    """
    if end - start <= S:
        return _insertion_sort(A, start, end, shift)

    mid = start + (end - start) // 2
    comparisons = _hybrid_merge_sort(A, start, mid, S, shift)
    comparisons += _hybrid_merge_sort(A, mid, end, S, shift)
    comparisons += merge(A, start, mid, end)
    return comparisons


def hybrid_merge_sort(A, start, end, S):
    """Sort A[start:end] in place with the hybrid algorithm (swap-based leaves).

    Returns
    -------
    int : number of key comparisons performed.
    """
    return _hybrid_merge_sort(A, start, end, S, shift=False)


def hybrid_merge_sort_shift(A, start, end, S):
    """Hybrid variant whose leaves use the shift-based insertion sort.

    Produces exactly the same key-comparison count as hybrid_merge_sort but
    runs faster, because it performs about one third of the data movement.
    """
    return _hybrid_merge_sort(A, start, end, S, shift=True)


# --------------------------------------------------------------------------
# 5. Convenience wrappers
# --------------------------------------------------------------------------


def sort_pure(A):
    """Sort the whole array with the original merge sort. Returns comparisons."""
    return merge_sort(A, 0, len(A))


def sort_hybrid(A, S):
    """Sort the whole array with the hybrid algorithm. Returns comparisons."""
    return hybrid_merge_sort(A, 0, len(A), S)


def sort_hybrid_shift(A, S):
    """Hybrid using the shift-based insertion sort. Returns comparisons."""
    return hybrid_merge_sort_shift(A, 0, len(A), S)


# --------------------------------------------------------------------------
# 6. Data generation
# --------------------------------------------------------------------------

X_MAX = 10_000_000          # largest key allowed in a dataset


def generate_dataset(n, x=X_MAX, seed=None):
    """Return a list of n random integers drawn uniformly from [1, x].

    A fixed `seed` makes an experiment reproducible; with x = 10^7 and
    n <= 10^7 duplicates are present but rare, which is the "random data"
    regime the average-case analysis assumes.
    """
    rng = random.Random(seed)
    return [rng.randint(1, x) for _ in range(n)]


DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]


# --------------------------------------------------------------------------
# 7. Self-test
# --------------------------------------------------------------------------


def _self_test():
    """Correctness + stability + counter sanity checks."""
    rng = random.Random(42)

    # (a) Correctness on random, sorted, reverse-sorted and duplicate-heavy data.
    for n in [0, 1, 2, 3, 7, 8, 33, 100, 257, 1000]:
        cases = {
            "random":    [rng.randint(1, 50) for _ in range(n)],
            "sorted":    list(range(n)),
            "reverse":   list(range(n, 0, -1)),
            "duplicate": [rng.randint(1, 3) for _ in range(n)],
        }
        for name, data in cases.items():
            expected = sorted(data)

            A = data[:]
            insertion_sort(A, 0, len(A))
            assert A == expected, f"insertion_sort failed on {name} n={n}"

            A = data[:]
            sort_pure(A)
            assert A == expected, f"merge_sort failed on {name} n={n}"

            for S in [1, 2, 5, 16, 32, 1000]:
                A = data[:]
                sort_hybrid(A, S)
                assert A == expected, f"hybrid S={S} failed on {name} n={n}"

    # (b) S = 1 must reproduce the pure merge sort exactly (same comparisons).
    for n in [1, 10, 100, 1000, 4321]:
        data = [rng.randint(1, 10_000) for _ in range(n)]
        a, b = data[:], data[:]
        assert sort_pure(a) == sort_hybrid(b, 1), f"S=1 != pure at n={n}"

    # (b2) The shift-based leaf variant must sort correctly and produce the
    #      identical comparison count.
    for n in [0, 1, 5, 40, 333, 2048]:
        data = [rng.randint(1, 500) for _ in range(n)]
        a, b = data[:], data[:]
        assert insertion_sort(a, 0, n) == insertion_sort_shift(b, 0, n) and a == b
        for S in [1, 4, 9, 32]:
            a, b = data[:], data[:]
            assert sort_hybrid(a, S) == sort_hybrid_shift(b, S) and a == b == sorted(data)

    # (c) A block of size <= S is handled purely by insertion sort.
    data = [rng.randint(1, 100) for _ in range(20)]
    a, b = data[:], data[:]
    assert sort_hybrid(a, 20) == insertion_sort(b, 0, 20)

    # (d) Stability: sort (key, tag) pairs by key only and check tag order.
    pairs = [(rng.randint(1, 5), t) for t in range(200)]
    keys = [p[0] for p in pairs]
    for sorter in (lambda A: sort_pure(A), lambda A: sort_hybrid(A, 8)):
        A = pairs[:]
        sorter(A)
        assert A == sorted(pairs, key=lambda p: p[0]), "not stable"
    del keys

    # (e) Known worst-case bounds.
    #     merge of a and b elements uses at most a + b - 1 comparisons;
    #     insertion sort on m elements uses at most m(m-1)/2.
    for n in [2, 3, 16, 64, 129]:
        A = list(range(n, 0, -1))
        c = insertion_sort(A, 0, n)
        assert c == n * (n - 1) // 2, f"insertion worst case wrong at n={n}"
        A = list(range(n))
        c = insertion_sort(A, 0, n)
        assert c == n - 1, f"insertion best case wrong at n={n}"

    for _ in range(500):
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        left = sorted(rng.randint(1, 100) for _ in range(a))
        right = sorted(rng.randint(1, 100) for _ in range(b))
        A = left + right
        c = merge(A, 0, a, a + b)
        assert A == sorted(left + right)
        assert c <= a + b - 1, f"merge exceeded a+b-1 ({c} > {a+b-1})"

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
