"""
SC2001 Project 1 - exact average-case theory.

Everything here is derived for a uniformly random permutation of n distinct
keys (which is what a random dataset over [1, 10^7] is, up to the rare tie).

-------------------------------------------------------------------------
1. INSERTION SORT
-------------------------------------------------------------------------
For iteration i (i = 1 .. m-1) let d_i be the number of swaps performed,
i.e. the number of elements of the sorted prefix A[0:i] that are larger
than A[i].  The while loop performs

        d_i comparisons that succeed  +  1 final failing comparison,

except when the element runs off the front (d_i = i), where the loop is
stopped by the index test j >= 0, which we do not count.  Hence

        C_ins(m) = sum_i d_i + #{ i : d_i < i }
                 = (number of inversions) + (elements not reaching the front)

    worst case (reverse sorted): d_i = i          -> m(m-1)/2
    best case  (already sorted): d_i = 0          -> m - 1
    average:  E[d_i] = i/2 ,  P(d_i < i) = i/(i+1)

        E[C_ins(m)] = sum_{i=1}^{m-1} ( i/2 + i/(i+1) )
                    = m(m-1)/4 + (m-1) - (H_m - 1)
                    = m^2/4 + 3m/4 - H_m                      (exact)

-------------------------------------------------------------------------
2. MERGING two sorted runs of sizes a and b
-------------------------------------------------------------------------
Every element that is written is either decided by a key comparison, or is
written for free after one of the runs is exhausted.  If r elements remain
in the non-exhausted run then

        C_merge = a + b - r ,        a + b - 1 in the worst case.

Over all C(a+b, a) equally likely interleavings,

        E[r] = a/(b+1) + b/(a+1)
        E[C_merge(a,b)] = a + b - a/(b+1) - b/(a+1)   ->  a + b - 2

-------------------------------------------------------------------------
3. PURE MERGE SORT
-------------------------------------------------------------------------
        C(n) = C(floor(n/2)) + C(ceil(n/2)) + E[C_merge]

With the approximation E[C_merge] = n - 2 and C(1) = 0 the closed form is

        C(n) ~ n log2(n) - 2n + 2

-------------------------------------------------------------------------
4. HYBRID MERGE SORT
-------------------------------------------------------------------------
        C(m,S) = E[C_ins(m)]                                  if m <= S
                 C(floor(m/2),S) + C(ceil(m/2),S)
                                 + E[C_merge(floor(m/2), ceil(m/2))]   else

Closed-form approximation.  The recursion is cut at depth
k = ceil(log2(n/S)); it produces about n/S leaves of size about S:

        C(n,S)  ~  (n/S) * (S^2/4)        [insertion sort at the leaves]
                 + n * log2(n/S)          [log2(n/S) merge levels, ~n each]

                =  n * ( S/4 + log2 n - log2 S )

Differentiating with respect to S:

        dC/dS = n ( 1/4 - 1/(S ln 2) ) = 0    =>    S* = 4/ln 2 ~ 5.77

and d2C/dS2 > 0, so this is a minimum.  Two consequences:

  * the comparison-optimal threshold is SMALL, about S = 6;
  * n cancels, so S* does not depend on the input size.

The exact recurrence above also reproduces the STAIRCASE seen empirically:
merge sort halves the array, so the leaf sizes can only be about n/2^k.
Raising S has no effect at all until it crosses the next leaf size, at
which point the whole bottom level of the recursion collapses at once.
"""

from functools import lru_cache
from math import log2


# --------------------------------------------------------------------------
# harmonic numbers
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def harmonic(m):
    return sum(1.0 / i for i in range(1, m + 1))


# --------------------------------------------------------------------------
# component costs
# --------------------------------------------------------------------------

def ins_avg(m):
    """Exact expected key comparisons of insertion sort on m random keys."""
    if m <= 1:
        return 0.0
    return m * m / 4 + 3 * m / 4 - harmonic(m)


def ins_worst(m):
    return m * (m - 1) / 2 if m > 1 else 0.0


def merge_avg(a, b):
    """Exact expected key comparisons when merging sorted runs of size a, b."""
    if a == 0 or b == 0:
        return 0.0
    return a + b - a / (b + 1) - b / (a + 1)


# --------------------------------------------------------------------------
# whole-algorithm costs (exact recurrences, memoised over the O(log n)
# distinct subproblem sizes that actually occur)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def pure_avg(n):
    """Expected key comparisons of the original merge sort on n random keys."""
    if n <= 1:
        return 0.0
    left = n // 2                      # mid = start + (end-start)//2
    right = n - left
    return pure_avg(left) + pure_avg(right) + merge_avg(left, right)


@lru_cache(maxsize=None)
def hybrid_avg(n, S):
    """Expected key comparisons of the hybrid algorithm on n random keys."""
    if n <= S:
        return ins_avg(n)
    left = n // 2
    right = n - left
    return hybrid_avg(left, S) + hybrid_avg(right, S) + merge_avg(left, right)


# --------------------------------------------------------------------------
# closed-form approximations (the ones quoted in the report)
# --------------------------------------------------------------------------

def pure_closed_form(n):
    """n log2 n - 2n + 2"""
    return n * log2(n) - 2 * n + 2 if n > 1 else 0.0


def hybrid_closed_form(n, S):
    """n ( S/4 + log2(n/S) ) - the leading-order model used to optimise S."""
    if S >= n:
        return n * n / 4
    return n * (S / 4 + log2(n / S))


OPTIMAL_S_CONTINUOUS = 4 / 0.6931471805599453             # 4 / ln 2 = 5.7708


# --------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"continuous comparison-optimal S* = 4/ln2 = {OPTIMAL_S_CONTINUOUS:.3f}\n")

    print("n           pure(exact)      pure(closed)     hybrid S=8      hybrid S=32")
    for n in [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]:
        print(f"{n:>10}  {pure_avg(n):>14.0f}  {pure_closed_form(n):>14.0f}"
              f"  {hybrid_avg(n, 8):>14.0f}  {hybrid_avg(n, 32):>14.0f}")

    print("\nexact model, n = 1,000,000, S = 1..16 (watch the staircase)")
    for S in range(1, 17):
        print(f"  S={S:>2}  {hybrid_avg(1_000_000, S):>12.0f}")

    best = min(range(1, 101), key=lambda S: hybrid_avg(1_000_000, S))
    print(f"\nexact-model optimum at n=10^6: S = {best}")
