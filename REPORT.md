# SC2001 Project 1 — Integration of Merge Sort & Insertion Sort

**Deliverables:** `hybrid_sort.py` (algorithms), `theory.py` (exact average-case model),
`experiments.py` (benchmarks), `plots.py` (figures), `results/*.json` (raw data),
`figures/*.png` (charts).

**Headline result.** The hybrid *never* reduces the number of key comparisons — the count is
non-decreasing in `S`, and pure merge sort (`S = 1`) is the comparison optimum. What the hybrid
buys is **CPU time**, by deleting the bottom levels of the recursion tree. Any claim that
"`S = 32` is optimal" is therefore only true under a *time* objective, and the report below
separates the two objectives carefully.

---

## 1. Algorithm and implementation

All three routines follow the lecture index convention: `A[start:end]` with `start` inclusive,
`end` exclusive, subproblem size `m = end − start`, base case `m ≤ 1`, split at
`mid = start + (end − start) // 2` (so the left half has `⌊m/2⌋` elements and the right half
`⌈m/2⌉`).

```
HybridMergeSort(A, start, end, S)
1   If end − start ≤ S Then
2       Return InsertionSort(A, start, end)
3   mid := start + (end − start) / 2
4   HybridMergeSort(A, start, mid, S)
5   HybridMergeSort(A, mid, end, S)
6   Merge(A, start, mid, end)
```

`S = 1` reproduces the original merge sort exactly: a block of one element is already sorted and
insertion sort on it costs zero comparisons. This is asserted in the self-test, so "pure merge
sort" and "hybrid at `S = 1`" are guaranteed to be the same algorithm, which makes the
head-to-head comparison in §6 fair.

### What counts as a key comparison

A key comparison is **one evaluation of a comparison operator between two array elements** —
`A[j] > A[j+1]` in insertion sort, `Bl[i] <= Br[j]` in the merge. Loop-bound tests (`j >= start`,
`i < n_left`, `start < end`) are *not* counted; they compare indices, not keys.

The implementation exploits short-circuit evaluation to make the counter **exact rather than
estimated**: the index test is written first, exactly as in the lecture pseudocode, so the
counter is incremented at the moment the key comparison is actually reached.

```python
while j >= start:
    comparisons += 1          # the test A[j] > A[j+1] happens now
    if A[j] > A[j + 1]:
        A[j], A[j + 1] = A[j + 1], A[j]
        j -= 1
    else:
        break
```

Counts are returned by value and accumulated in local integers rather than kept in a global —
faster in CPython, and it keeps the functions side-effect free.

### Correctness

`python3 hybrid_sort.py` runs a self-test covering: random / already-sorted / reverse-sorted /
duplicate-heavy inputs at 10 sizes; `S ∈ {1, 2, 5, 16, 32, 1000}`; the identity
`sort_hybrid(A, 1) == sort_pure(A)` on comparison counts; **stability** (sorting `(key, tag)`
pairs by key and checking the tags stay in order); insertion sort's exact best case `m − 1` and
worst case `m(m−1)/2`; and 500 randomised merges checked against the `a + b − 1` upper bound.

---

## 2. Theoretical analysis

Throughout, the input is a uniformly random permutation of `n` distinct keys. (With keys drawn
from `[1, 10⁷]` and `n ≤ 10⁷`, ties exist but are rare enough not to move any of the numbers
below; the empirical agreement in §4 confirms this.)

### 2.1 Insertion sort

For iteration `i` (`i = 1 … m−1`), let `dᵢ` be the number of swaps performed — equivalently, the
number of elements of the sorted prefix `A[0:i]` that exceed `A[i]`. The `while` loop performs
`dᵢ` *successful* comparisons plus **one final failing comparison**, except when the element runs
all the way off the front (`dᵢ = i`), where the loop is stopped by the index test, which is not
counted. Hence

$$C_{\text{ins}}(m) \;=\; \sum_{i=1}^{m-1} d_i \;+\; \#\{\,i : d_i < i\,\}$$

i.e. **(number of inversions) + (number of elements that do not reach the front)**.

| case | `dᵢ` | comparisons |
|---|---|---|
| best (already sorted) | `0` | `m − 1` |
| worst (reverse sorted) | `i` | `m(m−1)/2` |
| average | `E[dᵢ] = i/2`, `P(dᵢ < i) = i/(i+1)` | see below |

$$\mathbb{E}[C_{\text{ins}}(m)] \;=\; \sum_{i=1}^{m-1}\left(\frac{i}{2}+\frac{i}{i+1}\right)
\;=\; \frac{m(m-1)}{4} + (m-1) - (H_m - 1)
\;=\; \boxed{\frac{m^2}{4} + \frac{3m}{4} - H_m}$$

where `H_m = Σ_{k=1..m} 1/k ≈ ln m + γ`. Sanity checks: `m = 1 → 0`, `m = 2 → 1`. So insertion
sort is `Θ(m²)` on average with leading constant **1/4**, not 1/2 — the worst case is twice the
average.

### 2.2 Merging two runs of sizes `a` and `b`

Every element written to `A` is either *decided by a key comparison* or written for free after one
run is exhausted. If `r` elements remain in the run that was not exhausted,

$$C_{\text{merge}} = a + b - r, \qquad\text{so } C_{\text{merge}} \le a+b-1 \text{ (worst case).}$$

Over all `C(a+b, a)` equally likely interleavings, `E[r] = a/(b+1) + b/(a+1)`, hence

$$\mathbb{E}[C_{\text{merge}}(a,b)] \;=\; a + b - \frac{a}{b+1} - \frac{b}{a+1} \;\xrightarrow{\ a,b\ \text{large}\ }\; a+b-2 .$$

The `≤` (not `<`) in the merge test is what makes the algorithm **stable**. Note that for *small*
runs the `−a/(b+1) − b/(a+1)` correction matters a lot: merging `1+1` costs **1** comparison, not
2. This is exactly why the bottom levels of the recursion are cheap, and it is the reason the
naive optimisation in §2.4 over-estimates the hybrid's benefit.

### 2.3 Pure merge sort

$$C(n) = C(\lfloor n/2\rfloor) + C(\lceil n/2\rceil) + \mathbb{E}[C_{\text{merge}}], \qquad C(1)=0 .$$

Using `E[C_merge] ≈ n − 2` and substituting `C(n) = n log₂ n + αn + β`:

$$2\Big[\tfrac n2(\log_2 n - 1) + \tfrac{\alpha n}{2} + \beta\Big] + n - 2
= n\log_2 n + \alpha n + 2\beta - 2 \;\Rightarrow\; \beta = 2,\ \alpha = -2$$

$$\boxed{C_{\text{pure}}(n) \approx n\log_2 n - 2n + 2} \qquad \in \Theta(n\log n)$$

(Worst case `n⌈log₂n⌉ − 2^{⌈log₂ n⌉} + 1`; best case `≈ (n log₂ n)/2`.)

### 2.4 Hybrid: the naive optimisation, and why it is wrong

The textbook argument: the recursion is cut at depth `k = log₂(n/S)`, producing `≈ n/S` leaves of
size `≈ S`; each leaf costs `S²/4`; each of the remaining `log₂(n/S)` merge levels costs `≈ n`:

$$C(n,S) \;\approx\; \frac{n}{S}\cdot\frac{S^2}{4} \;+\; n\log_2\!\frac nS
\;=\; n\left(\frac S4 + \log_2 n - \log_2 S\right)$$

$$\frac{\partial C}{\partial S} = n\left(\frac14 - \frac{1}{S\ln 2}\right) = 0
\quad\Longrightarrow\quad S^\* = \frac{4}{\ln 2} \approx 5.77 ,$$

with `∂²C/∂S² > 0`, a genuine minimum — and note **`n` cancels, so `S*` is independent of the
input size**. That last conclusion survives; the value `5.77` does not, because the model prices
every skipped merge level at a full `n` comparisons. It does not: by §2.2 a level that merges runs
of size `j` costs only `n − n/(j+1)` and the *bottom* levels (`j = 1, 2, 4`) are far cheaper than
`n`. The model also silently drops insertion sort's `+3m/4` term.

### 2.5 Hybrid: the correct model

Replacing the merge-sorting of one leaf block by insertion sort changes the cost by
`Δ(S) = E[C_ins(S)] − E[C_pure(S)]`, and there are about `n/S` such blocks:

$$\boxed{\;C_{\text{hyb}}(n,S) \;\approx\; C_{\text{pure}}(n) \;+\; \frac nS\,\Delta(S),
\qquad \Delta(S) = \mathbb{E}[C_{\text{ins}}(S)] - \mathbb{E}[C_{\text{pure}}(S)]\;}$$

| `m` | `E[C_ins(m)]` | `E[C_mergesort(m)]` | `Δ(m)` | `Δ(m)/m` (extra comparisons **per element**) |
|---|---|---|---|---|
| 2 | 1.00 | 1.00 | **0.00** | 0.000 |
| 3 | 2.67 | 2.67 | **0.00** | 0.000 |
| 4 | 4.92 | 4.67 | 0.25 | 0.063 |
| 8 | 19.28 | 15.73 | 3.55 | 0.444 |
| 16 | 72.62 | 45.69 | 26.93 | 1.683 |
| 32 | 275.94 | 121.50 | 154.45 | 4.826 |
| 64 | 1067.26 | 305.05 | 762.20 | 11.909 |
| 100 | 2569.81 | 541.84 | 2027.97 | 20.280 |

`Δ(m) ≥ 0` for every `m`, with equality only at `m ≤ 3`. Therefore:

> **On key comparisons the hybrid can never beat pure merge sort.** `C_hyb(n,S)` is
> non-decreasing in `S`, the optimum is `S ∈ {1, 2, 3}`, and the penalty grows like
> `n·(S/4 − log₂S)` — roughly `+1.7n` at `S = 16` and `+4.8n` at `S = 32`.

Expanding `Δ(S)/S` gives the closed form used for the curves:

$$C_{\text{hyb}}(n,S) \;\approx\; n\left[\log_2\frac nS + \frac S4 + \frac34 - \frac{H_S + 2}{S}\right]$$

which correctly reduces to `n(log₂ n − 2)` at `S = 1`.

### 2.6 The staircase

`C_hyb(n, S)` is a **step function of `S`**, not a smooth curve. Merge sort halves the array, so a
leaf can only have size `⌊n/2^k⌋` or `⌈n/2^k⌉`. Increasing `S` changes nothing at all until it
crosses the next leaf size, at which point an entire level of the recursion collapses into
insertion sort at once. For `n = 10⁶` the leaf sizes are ≈ `15, 30, 61, 122, …`, so the empirical
curve is flat on `S ∈ [16, 30]`, jumps, is flat on `S ∈ [31, 61]`, and so on. This is predicted
exactly by the recurrence in `theory.py` and confirmed in Figure 2.

### 2.7 Time complexity of the hybrid

Comparisons: `Θ(n log(n/S) + nS)`. With `S` a **constant**, `log(n/S) = log n − log S = Θ(log n)`
and `nS = Θ(n)`, so the hybrid is `Θ(n log n)` — the same asymptotic class as merge sort, which is
also the `Ω(n log n)` lower bound for comparison sorting. `S` only moves the constant factor.
If instead `S` were allowed to *grow* with `n` (say `S = Θ(n)`), the `nS` term would dominate and
the algorithm would degrade to `Θ(n²)`.

Space: `O(n)` auxiliary for the merge buffers plus `O(log(n/S))` stack — the hybrid strictly
reduces both, since it removes the bottom `log₂ S` levels of allocations and calls.

---

## 3. Experiment (c)(i) — fixed `S`, varying `n`

Datasets: uniform random integers in `[1, 10⁷]`, seed 2001,
`n ∈ {10³, 10⁴, 10⁵, 10⁶, 5·10⁶, 10⁷}`. See `figures/fig1_comparisons_vs_n.png`.

| `n` | pure | theory (exact) | err | hybrid `S=8` | hybrid `S=32` | `C_pure/(n log₂n)` |
|---|---|---|---|---|---|---|
| 1,000 | 8,715 | 8,707 | +0.09% | 9,101 | 13,472 | 0.874 |
| 10,000 | 120,521 | 120,451 | +0.06% | 121,549 | 143,658 | 0.907 |
| 100,000 | 1,536,201 | 1,536,367 | −0.01% | 1,558,307 | 1,860,883 | 0.925 |
| 1,000,000 | 18,674,131 | 18,674,241 | −0.001% | 19,072,445 | 23,185,656 | 0.937 |
| 5,000,000 | 105,051,276 | 105,050,350 | +0.001% | 105,555,212 | 116,169,506 | 0.944 |
| 10,000,000 | 220,103,984 | 220,100,699 | +0.001% | 221,112,286 | 242,323,226 | 0.947 |

**Reading the table.** The exact recurrence of §2.3/§2.5 predicts the measured counts to within
**0.01% for `n ≥ 10⁵`** (0.7% at `n = 10³`, where the finite-size `H_m` and floor/ceiling terms
still matter). The closed form `n log₂n − 2n + 2` is deliberately looser — it underestimates by
3.4% at `n = 10⁷` — because it prices every merge at `a + b − 2` even at the bottom of the tree.

The last column is the empirical proof of `Θ(n log n)`: `C/(n log₂ n)` is not constant but creeps
from 0.874 to 0.947 across four orders of magnitude, exactly as `1 − 2/log₂n` predicts for a
`n log₂n − 2n` law. A pure `Θ(n²)` law would have made this column grow by a factor of ~10⁴, and a
pure `Θ(n)` law would have shrunk it by ~24. On a log-log plot (Figure 1, left) all three curves
are straight and parallel with slope slightly above 1 — the fixed vertical offsets between them
are the constant-factor penalties `Δ(S)·n/S` of §2.5, not a change in growth rate.

## 4. Experiment (c)(ii) — fixed `n`, varying `S`

`n = 10⁶`, `S = 1 … 100`, all 100 values. See `figures/fig2_comparisons_vs_S.png`.

The measured curve is a **staircase**, exactly as predicted in §2.6. The steps are:

| `S` range | key comparisons | vs pure |
|---|---|---|
| 1 – 3 | 18,674,131 | 1.000 |
| 4 – 6 | 18,727,475 | 1.003 |
| 7 | 18,820,771 | 1.008 |
| 8 – 14 | 19,072,445 | 1.021 |
| 15 | 19,886,585 | 1.065 |
| 16 – 29 | 20,222,709 | 1.083 |
| 31 – 60 | 23,185,656 | 1.242 |
| 62 – 100 | 29,893,419 | 1.601 |

**The trade-off, stated correctly.** As `S` grows, two things happen:

1. **Merge levels are removed.** `log₂S` levels of the recursion disappear, saving roughly
   `n·log₂S` comparisons — but only *roughly*, because the levels removed are the **cheapest**
   ones. A level merging runs of size `j` costs `n − n/(j+1)`, so the bottom level (`j = 1`)
   costs only `n/2`, not `n`.
2. **Insertion sort takes over the leaves.** Each of the `n/S` leaves now costs `S²/4 + 3S/4 − H_S`
   instead of `≈ S log₂S − 2S`, i.e. a quadratic term replaces a linearithmic one.

Effect 2 always dominates effect 1 (§2.5, `Δ(S) ≥ 0`), so **the curve is monotone non-decreasing
and there is no interior minimum in comparisons.** The flat regions are where `S` sits between two
consecutive leaf sizes and changes nothing; the jumps are where an entire bottom level collapses
into insertion sort at once. The step edges at `S = 4, 8, 16, 31, 62` are the leaf sizes
`⌈10⁶/2ᵏ⌉` — the `31` and `62` rather than `32` and `64` come from the `⌊m/2⌋` / `⌈m/2⌉` split of a
non-power-of-two array.

## 5. Experiment (c)(iii) — determining the optimal `S`

Full sweeps `S = 1…100` at `n = 10³, 10⁴, 10⁵, 10⁶`; grid `S ∈ {1,2,4,6,8,12,16,24,32,64,100}`
at `n = 5·10⁶, 10⁷`. See `figures/fig3_optimal_S.png` and `figures/fig4_cpu_time_vs_S.png`.

**On key comparisons.** The minimum lands on `S ∈ {1, 2, 3}` at *every* size tested — there is no
crossover point and no scale-dependent optimum. When the curves are normalised by their own
`S = 1` value they **collapse onto essentially one curve**, which is the empirical confirmation of
the theoretical claim in §2.4 that `n` cancels out of `∂C/∂S`. Larger `n` collapses slightly
*lower* only because `Δ(S)·n/S` is compared against a larger `n log₂n` baseline.

**On CPU time** (`n = 10⁶`, minimum of 3 runs, both leaf variants):

| `S` | comparisons | rel. | CPU swap-based (s) | CPU shift-based (s) | speed-up |
|---|---|---|---|---|---|
| 1 | 18,674,131 | 1.000 | 1.900 | 1.881 | 1.00× |
| 4 | 18,727,475 | 1.003 | 1.741 | 1.679 | 1.12× |
| 6 | 18,727,475 | 1.003 | 1.686 | 1.645 | 1.14× |
| **10** | 19,072,445 | 1.021 | 1.766 | **1.603** | **1.17×** |
| 12 | 19,072,445 | 1.021 | 1.673 | 1.646 | 1.14× |
| 16 | 20,222,709 | 1.083 | 1.681 | 1.669 | 1.13× |
| 24 | 20,222,709 | 1.083 | 1.772 | 1.698 | 1.11× |
| 32 | 23,185,656 | 1.242 | 2.031 | 1.777 | 1.06× |
| 64 | 29,893,419 | 1.601 | 2.634 | 2.149 | 0.88× |
| 100 | 29,893,419 | 1.601 | 2.728 | 2.195 | 0.86× |

The time curve **does** have an interior minimum, near `S ≈ 10`, with a broad flat basin over
roughly `S ∈ [6, 24]` and clear degradation beyond `S = 32`. Run-to-run noise is about ±3%, so the
honest statement is *"any `S` in 8–16 is optimal within measurement error"*, not a single value.

**Why time and comparisons disagree.** The time model adds a term the comparison model does not
have — per-call overhead. With `≈ 2n/S` recursive calls, each allocating two slice buffers:

$$t(n,S) \;\approx\; c_{\text{ins}}\frac{nS}{4} \;+\; c_{\text{mrg}}\,n\log_2\frac nS \;+\; c_{\text{call}}\frac{2n}{S}$$

$$\frac{\partial t}{\partial S}=0 \;\Longrightarrow\; \frac{c_{\text{ins}}}{4}S^2 - \frac{c_{\text{mrg}}}{\ln 2}S - 2c_{\text{call}} = 0$$

The `−2c_call/S²` term is what pushes the optimum away from `S = 1` and up to single or low double
digits. It also explains the shape: below `S ≈ 8` call overhead dominates, above `S ≈ 32` the
quadratic insertion term dominates.

**Recommended answer for the presentation:**

* if the objective is **key comparisons** → `S = 1 … 3` (i.e. don't hybridise);
* if the objective is **running time** → `S ≈ 8–16`, and we use **`S = 16`** below.

## 6. Part (d) — original merge sort vs hybrid at `n = 10,000,000`

Same dataset (seed 2001), CPU time = `time.process_time()`, minimum of 2 runs, garbage collector
disabled inside the timed region, array copy made outside it.
See `figures/fig5_head_to_head_10M.png`.

| algorithm | key comparisons | vs pure | CPU time (s) | wall (s) | vs pure |
|---|---|---|---|---|---|
| **Original merge sort** (`S = 1`) | **220,103,984** | — | 27.75 | 27.79 | — |
| Hybrid, `S = 8` | 221,112,286 | +0.46% | 26.50 | 26.54 | −4.5% |
| **Hybrid, `S = 16`** (optimal) | **226,423,260** | **+2.87%** | **26.20** | **26.22** | **−5.6%** |
| Hybrid, `S = 32` | 242,323,226 | +10.09% | 28.04 | 28.07 | +1.0% |

**Conclusion.** At the optimal threshold the hybrid trades **+2.9% more key comparisons** for
**−5.6% CPU time**. The saving comes from deleting the bottom four levels of the recursion:
about `2n/S ≈ 1.25 million` function calls and the same number of slice allocations disappear,
and each leaf is then sorted with a tight, allocation-free, cache-resident loop. At `S = 32` the
quadratic insertion-sort term has already eaten the entire benefit.

Both algorithms are `Θ(n log n)`; nothing here changes the asymptotics. The measured ratio
`220,103,984 / 18,674,131 = 11.79` against the predicted
`(10⁷·log₂10⁷ − 2·10⁷)/(10⁶·log₂10⁶ − 2·10⁶) = 11.78` is a further check on the model.

**Implementation caveat worth stating out loud.** The lecture's insertion sort swaps a pair on
every step (three list operations); `insertion_sort_shift` holds the key in a register and shifts
(one store), with an **identical comparison count**. At `S = 32` this is worth 1.78 s vs 2.03 s —
a 12% difference caused purely by data movement, invisible to the comparison counter. This is why
the measured CPU gain in CPython (≈ 6%) is smaller than the gain typically quoted for C libraries,
which use cutoffs of 16–32: in C the per-comparison cost of insertion sort is far below the cost
of a merge step and a function call, which both shrinks `c_ins` and raises the optimal `S`.

## 7. Presentation outline (10 minutes)

| min | content |
|---|---|
| 0:00–1:30 | Problem, hybrid pseudocode, what we count as a key comparison and why the counter is exact |
| 1:30–3:30 | Theory: `E[C_ins(m)] = m²/4 + 3m/4 − H_m`; `E[C_merge] = a+b−a/(b+1)−b/(a+1)`; `C_pure ≈ n log₂n − 2n + 2` |
| 3:30–4:30 | Fig 1 — `Θ(n log n)` confirmed; theory matches to 0.01% |
| 4:30–6:00 | Fig 2 — the staircase, and why comparisons *increase* with `S` (`Δ(S) ≥ 0` table) |
| 6:00–7:00 | Fig 3 & 4 — curves collapse ⇒ `S*` independent of `n`; comparisons say `S ≤ 3`, time says `S ≈ 8–16` |
| 7:00–8:00 | Fig 5 — the 10M table: +2.9% comparisons, −5.6% time |
| 8:00–10:00 | Q&A |

**Three questions the TA will probably ask, and the answers:**

1. *"Why did your comparison count go up? Isn't the hybrid supposed to be faster?"* — Faster in
   time, not in comparisons. Insertion sort is never cheaper than merge sort in comparisons even on
   a block of 4 (4.92 vs 4.67); the win is in call overhead and memory traffic.
2. *"Why is the curve flat between `S = 16` and `S = 30`?"* — Leaf sizes are quantised to `≈ n/2ᵏ`;
   `S` has no effect until it crosses the next one.
3. *"Does the optimal `S` depend on `n`?"* — No. `n` factors out of `∂C/∂S`; confirmed at six
   sizes spanning four orders of magnitude.

## 8. Reproducing everything

```bash
python3 hybrid_sort.py          # self-tests: correctness, stability, exact bounds
python3 theory.py               # exact average-case model tables
python3 experiments.py all      # ~25 min single-core; writes results/*.json
python3 plots.py                # writes figures/*.png
python3 plots.py --show         # interactive matplotlib windows (pan / zoom / read-out)
```

All experiments use `seed = 2001`, so every number in this report is reproducible.
