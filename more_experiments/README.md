# gsw_stability — the Gram–Schmidt walk under reduced precision

Experiments on how float32/float16 arithmetic and ill-conditioning affect the
Gram–Schmidt walk (Bansal–Dadush–Garg–Lovett 2017; Harshaw–Sävje–Spielman–Zhang
2023). They measure the subgaussian constant, the bias, path divergence and the
per-step error.

## Running

```sh
nix-shell                                  # or: python3 -m venv .venv && pip install "numpy>=2" scipy matplotlib pytest
python -m pytest -q tests
python run_experiments.py --smoke          # n=50, R=200 -> results/smoke/, then a full-grid time projection
python run_experiments.py                  # full grid -> results/ (resumable; re-run to continue)
python run_experiments.py --report-only    # rebuild table and figures from results/raw/
```

The grid reported below was run as:

```sh
python run_experiments.py \
  --R-rules "lstsq@200=200,lstsq@500=10,gs@500=300,gs/float16@500=100,chol@500=500,chol_rk10@500=500" \
  --diag-runs 10 --diag-runs-by-n 500:0
```

To estimate the time left on a running grid, run the same command with `--status` added.

### On a many-core workstation (e.g. Threadripper)

The default settings run the full grid: R = 2000 everywhere except `walk_lstsq` at
n = 500 (R = 300), with 20 diagnostic runs per cell (5 at n = 500). Each worker is
single-threaded and `--workers` defaults to every hardware thread. Write to a
separate output directory: BLAS on another machine may build the test matrices
slightly differently, so raw files from different machines should not be mixed.

```sh
cd more_experiments
tmux                                                             # keeps the run alive if the terminal closes
nix-shell                                                        # or activate the venv from "Running"
python -m pytest -q tests
python run_experiments.py --smoke --outdir results_threadripper/smoke   # ~5 min; prints a time projection
python run_experiments.py --outdir results_threadripper 2>&1 | tee results_threadripper.log
```

- **Expected time:** very roughly 9 hours on 64 cores and 18 hours on 32 cores.
  About half the work is float16 `walk_lstsq` at n = 500.
- **Smoke-test projection:** it times walks one at a time, so it is optimistic.
  `walk_gs_compress` at n = 500 is limited by memory bandwidth and slows down
  considerably when every core is busy.
- **Checking progress:** after 1–2 hours, run the same command with `--status` added
  (`python run_experiments.py --outdir results_threadripper --status`).
- **If the estimate is too long:** press Ctrl-C and rerun with fewer n = 500
  `walk_lstsq` runs, e.g. `--R-rules "lstsq@500=100"`. Finished tasks are kept.

`python run_experiments.py --help` lists the knobs:
- `--R` sets the runs per configuration. `--R-by-n` overrides it per n, and
  `--R-rules "impl[/dtype]@n=R,..."` overrides it per implementation and dtype.
- `--diag-runs` sets how many runs get per-step logging.
- `--ns`, `--families`, `--dtypes`, `--impls` and `--workers` restrict the grid.

Each task writes one file to `results/raw/`. Tables and figures are always rebuilt from those files.

Run a whole grid on one machine. Test matrices are built with LAPACK/BLAS, which
may differ in the last bits across machines. Mixing raw files from two machines
would mix paths from slightly different B.

## Layout

| file | contents |
|---|---|
| `gsw_stability/linalg_lowp.py` | all precision-sensitive linear algebra: `tree_sum`, Householder QR + back substitution, Cholesky, rank-one update/downdate, Sherman–Morrison removal |
| `gsw_stability/walks.py` | the shared walk skeleton and the three implementations |
| `gsw_stability/matrices.py` | test matrix families a–d |
| `gsw_stability/metrics.py` | ψ₂ estimators, bias, CCDF, path divergence |
| `gsw_stability/diagnostics.py` | per-step comparison against a float64 solve |
| `gsw_stability/runner.py` | grid, multiprocessing, raw-result cache, timing projection |
| `gsw_stability/report.py`, `plots.py` | `results/table.{csv,md}`, `results/figures/` |

## Arithmetic

All walk arithmetic happens in the requested dtype: z, u, the step sizes, the probability and every factorization.
- numpy's float16 ufuncs (`+ − × ÷ √`) round each result to float16.
- numpy's float16 reductions (`sum`, `dot`, `@`) accumulate in float32. So nothing
  in `linalg_lowp.py` uses them. Every inner product goes through `tree_sum`, a
  pairwise sum built from elementwise adds (a test checks it rounds every add).
- The same routines run for float64, float32 and float16, so runs differ only in
  rounding. No LAPACK or BLAS is called inside a walk.
- Vector norms are scaled by max|x| (as in LAPACK `nrm2`), so float16 squares do
  not underflow.

**Randomness.** Each run pre-draws a float64 array `U[t] = (pivot draw, coin draw)`
from `default_rng([seed, run])`.
- Step t uses `U[t,0]` to pick a pivot, if one is needed. The pivot is the
  ⌊U·|A|⌋-th alive index in sorted order.
- Step t takes δ⁺ iff `U[t,1] < p̂`, where p̂ = δ̂⁻/(δ̂⁺+δ̂⁻) is computed in the dtype.

Every implementation and dtype therefore sees the same draws. Any divergence
between precisions comes from rounding. Comparing a float64 uniform against the
dtype-valued p̂ samples Bernoulli(p̂) exactly.

**Ratio test and snapping** (shared by all implementations):
- Entries with |u_i| ≤ 10·eps(dtype) are skipped in the ratio test, as in GSWDesign.jl.
- The coordinate(s) attaining the ratio-test minimum are set to exactly ±1. This
  guarantees termination in ≤ n steps.
- Any other alive coordinate within `snap_c·eps(dtype)` of ±1 (default
  `snap_c = 10`) is snapped to ±1.
- Any value outside [−1, 1] is clamped.
- Each kind of correction is counted and its magnitude recorded. GSWDesign.jl's
  freeze tolerance of 100·eps would be 0.098 in float16, so the tolerance here is
  dtype-relative and configurable.

**Failures.** A run is marked failed and excluded from ψ₂/bias when it produces
any of:
- a non-finite u or p̂,
- a Cholesky pivot ≤ 0,
- a downdate with s² ≥ 1 (Julia throws `PosDefException` for s² > 1).

The table reports the fraction of failed runs and the reasons.

## Implementations

1. **`walk_lstsq`** (reference). At each step it solves
   u_{A∖p} = −argmin‖B_{A∖p}c − b_p‖ from scratch, using the custom Householder QR
   and back substitution. It works directly on B.
2. **`walk_harshaw_cholesky`** follows GSWDesign.jl (`sample_gs_walk`,
   `_gs_walk_recur`, `compute_step_direction`) for B = [√φ I; √(1−φ) Xᵀ], with X
   rows scaled by the max row norm.
   - Like the reference implementation, it factors the **d×d** matrix
     M = (φ/(1−φ)) I + X_S X_Sᵀ over the non-pivot alive set S (Woodbury form),
     not the n×n B_SᵀB_S.
   - Initialization is `cholesky(c·I)` followed by one rank-one update per unit.
   - The unit chosen as pivot is downdated, and each non-pivot unit is downdated
     when it freezes.
   - The update and downdate mirror Julia's `lowrankupdate!`/`lowrankdowndate!`
     (`uplo='U'`), including its `givensAlgorithm` power-of-two rescaling.
   - The step direction reproduces Julia's arithmetic literally:
     `a = L(U x_p) − c x_p;  a = M⁻¹a;  a = ((1−φ)/φ)(a − x_p);  u = X_Sᵀ a`.
     In exact arithmetic this equals −X_Sᵀ M⁻¹ x_p.
   - Julia's recursion (`targ_frozen`) only saves memory and is omitted.
   - `refactor_every=k` refactors from scratch every k steps. It forms
     M_S = cI + X_S X_Sᵀ in the dtype, using tree sums, and applies the custom
     Cholesky.
   - The factor drift reported is ‖UᵀU − M_S‖_F / ‖M_S‖_F, computed in float64 from the dtype-rounded X.
3. **`walk_gs_compress`** is `kernel_gs_walk_cubic` from Low-Rank Thinning
   (Carrell, Gong, Shetty, Dwivedi, Mackey, arXiv 2502.12063), App. B.6,
   Alg. GS-Halve-Cubic.
   - It works on the Gram matrix Q = BᵀB, formed in the dtype, rather than on B.
   - It keeps an explicit inverse C = (Q_{S,S})⁻¹. The initial C comes from a
     Cholesky factorization and triangular solves; the paper does not specify how
     to compute it.
   - C is updated by block inversion plus Sherman–Morrison when an index leaves S:
     C ← D − D q qᵀ D / (Q_ii + qᵀ D q).
   - u_S = −C Q_{S,p}.
   - It removes **one** index per iteration, the smallest one at ±1. When several
     coordinates reach ±1 at once, the rest are removed on later, zero-length
     iterations. Its step count can therefore exceed n.
   - In GS-Thin, B is the paired-difference feature matrix, so Q_ij is the
     paired-difference kernel. The halving and compress recursion around the walk
     is not implemented; here the walk runs on the same B as the others.

**Agreement.** In float64 all three produce the same z, pivots and discrepancy on
Harshaw-type B with the same seed (`tests/test_walks.py`, ‖D₁ − D₂‖ ≤ 1e-10).

## Test matrices (built in float64, cast to the dtype at walk entry)

- **a. Gaussian:** G divided by its largest column norm.
- **b. Controlled conditioning:** U diag(s) Vᵀ with log-spaced s and
  κ ∈ {1, 1e2, 1e4, 1e6, 1e8}. It is divided by the largest column norm, a single
  scalar, so κ is unchanged.
- **c. Near-collinear pairs:** n/2 interleaved pairs (v_j, (v_j + η g_j)/‖·‖) with
  v, g random unit vectors and η ∈ {1e-1, 1e-3, 1e-5, 1e-7}. Read literally,
  "(1+η)v_j/‖·‖" normalizes back to v_j exactly, so a random perturbation of size
  η is used instead.
- **d. Harshaw designs:** X is n×d with d ∈ {5, 50}, either Gaussian or
  correlated. The correlated version has column singular values log-spaced over
  1e4, giving κ(X) ≈ 1e4. φ ∈ {0.5, 0.1, 0.01} and m = n + d.
  `walk_lstsq`, `walk_harshaw_cholesky` (refactor off and k = 10) and
  `walk_gs_compress` all run on the same B.

## Measurements

Unless stated otherwise, measurements use float64 on the original B and the final z ∈ {±1}ⁿ.

- **Directions:** D = B z (z₀ = 0) and Y_v = ⟨D, v⟩ for 20 fixed random unit v,
  plus the top and bottom left singular vectors of B.
- **ψ₂ estimates:**
  - (i) the Orlicz estimate: the smallest K with mean exp(Y²/K²) ≤ 2, found by bisection;
  - (ii) the moment estimate: max over p ∈ {2,4,6,8} of (E|Y|^p)^{1/p}/√p.
  - For reference, N(0,1) gives (i) √(8/3) ≈ 1.63 and (ii) 1/√2 ≈ 0.71.
  - `psi2_ratio` divides by the float64 `walk_lstsq` value, computed on the runs both succeeded on.
- **Bias:** max over v of |mean Y_v|, reported with that direction's standard error.
- **Path divergence:** the first step at which the pivot, or the set of
  coordinates reaching ±1, differs from float64 `walk_lstsq` with the same seed.
- **Per-step diagnostics** (the first `--diag-runs` runs):
  - At each step the reference is computed from the walk's own
    (z_t, A_t, p_t) and the dtype-rounded B, all upcast to float64. It measures
    the arithmetic error of the step, not the representation error of B.
  - For the Cholesky and GS implementations the walk's inputs are the rounded X
    or Q rather than the rounded B. Their error floor therefore includes an
    O(eps) mismatch between those roundings.
  - Logged per step:
    - ‖B(û − u⁽⁶⁴⁾)‖₂, ‖û − u⁽⁶⁴⁾‖∞ and ‖u⁽⁶⁴⁾‖∞;
    - κ(B_{A∖p}) and |p̂ − p⁽⁶⁴⁾|;
    - the Cholesky drift, and for GS ‖CQ − I‖_F/√k.

## Results

Full numbers are in `results/table.md` / `results/table.csv`, and figures in `results/figures/`.
The grid ran on a 6-core/12-thread laptop with 12 workers and took about 13 hours of
wall-clock time.

**Runs per configuration** (the table's `R` column):

| | n=50 | n=200 | n=500 |
|---|---|---|---|
| `walk_lstsq` (incl. the float64 baseline) | 2000 | 200 | 10 |
| `walk_gs_compress` float64/float32 | 2000 | 2000 | 300–1000, varies by configuration |
| `walk_gs_compress` float16 | 2000 | 2000 | 100 |
| `walk_harshaw_cholesky`, refactor off and k=10 | 2000 | 2000 | 500 |

- **Per-step diagnostics:** 10 runs per cell at n ≤ 200 and none at n = 500, so the
  `max_err_*` and `max_drift` columns are empty at n = 500.
- **Ratios at n = 500:** ψ₂ ratios and divergence use only runs shared with the
  float64 `walk_lstsq` baseline, which at n = 500 is at most 10.
- **Uneven `gs` counts:** the n = 500 `walk_gs_compress` counts vary because the grid
  was restarted with a lower setting partway through.

### Sanity checks

- Every successful run ended with z ∈ {±1}ⁿ in at most n steps, for all
  implementations. `walk_gs_compress` never needed an extra zero-length step.
- Float64 `walk_lstsq` on Gaussian B gives ψ₂ (Orlicz, max over directions) of 0.98,
  0.98 and 0.94 at n = 50, 200 and 500 (R = 2000, 200 and 10).
- In float64, `walk_harshaw_cholesky` (refactor off and k = 10) and
  `walk_gs_compress` follow the same path as `walk_lstsq` on all 36 Harshaw
  configurations (0 divergence).
- The only float64 failures are `walk_gs_compress` on near-collinear pairs with
  η = 1e-7, at every n. There, the initial Cholesky of Q_{A∖p} fails.

### Failures (fraction of runs)

- **`walk_lstsq`** fails only in float16 on near-collinear pairs, with a non-finite u:
  - η = 1e-5: 99.5%, 90.5% and 10% at n = 50, 200 and 500;
  - η = 1e-7: 100%.

  It did not fail on the conditioned family at any κ ≤ 1e8, in any precision.
- **`walk_gs_compress`** fails when forming the initial inverse (Cholesky breakdown):
  - float32 at κ ≥ 1e6 and η ≤ 1e-3; also 9% at η = 0.1 for n = 500;
  - float16 at κ ≥ 1e4 and at every η. At κ = 1e2 it fails 0.7%, 96% and 100% of
    runs at n = 50, 200 and 500.
- **`walk_harshaw_cholesky` in float16** fails with rank-one downdate breakdown
  (s² ≥ 1):
  - averaged over the 12 designs: 14%, 50% and 75% of runs at n = 50, 200 and 500;
  - 100% of runs in 6 designs at n = 200 and in 8 designs at n = 500;
  - no failures in float32.
- **Refactoring (k = 10) in float16** reduces this to at most 1.8% of runs (one design
  at n = 500), and at most 0.1% elsewhere.

### Path divergence from float64 `walk_lstsq`

| conditioned, fraction diverged | κ=1e2 | κ=1e4 | κ=1e6 | κ=1e8 |
|---|---|---|---|---|
| `walk_lstsq` float32, n=50 | 0.0005 | 0.0015 | 0.135 | 0.966 |
| `walk_lstsq` float32, n=200 | 0.005 | 0.025 | 0.67 | 1 |
| `walk_lstsq` float32, n=500 (R=10) | 0 | 0.2 | 1 | 1 |
| `walk_lstsq` float16, all n | 0.97–1 | 1 | 1 | 1 |
| `walk_gs_compress` **float64**, n=50/200/500 | 0 | 0 | 0 / 0.02 / 0.3 | 0.99–1 |

- **Float64 `walk_gs_compress` on near-collinear pairs** also diverges from float64
  `walk_lstsq`:
  - η = 1e-3: 0.05%, 12% and 60% of runs at n = 50, 200 and 500;
  - η = 1e-5: 100%.
- **Harshaw designs:** the mean fraction diverged is at most 4% in float32 for every
  implementation, and 32–76% in float16.

### ψ₂ and bias

**ψ₂ ratio** to float64 `walk_lstsq`, computed over the runs where both succeeded:
- **Conditioned family, `walk_lstsq`:**
  - float32: 0.90–1.12;
  - float16: 0.81–1.39 up to n = 200, and 1.32 and 1.72 at κ = 1e6 and 1e8 for
    n = 500 (R = 10).
- **Harshaw designs, float32:** 1.00–1.04 for every implementation.
- **Harshaw designs, float16:**
  - `walk_lstsq`: max 1.14 at n = 200 and 1.49 at n = 500;
  - `walk_gs_compress`: max 1.86 at n = 200 and 6.74 at n = 500;
  - `walk_harshaw_cholesky`: max 2.12, at n = 50, computed on the 2.6% of runs that
    did not fail;
  - with refactoring (k = 10): median 1.12 at n = 200 and 1.43 at n = 500. The
    maximum is 10.7 at n = 200 and 30.4 at n = 500, both on correlated X with d = 5
    and φ = 0.01.

**Near-collinear pairs, η = 1e-3.** The float64 ψ₂ is set by 9 of the 2000 runs at
n = 50. In those runs, 1–5 of the 25 pairs end with equal signs, giving
max |Y| ≈ 1.1. Without them, ψ₂ ≈ 1.1e-3 in every precision. Float32 has 8 such runs
and float16 has none, so the float16 ψ₂ ratio is 0.003–0.006.

**Bias.** max_v |mean Y_v| / SE over the 22 directions:

| rows | median | 95th percentile |
|---|---|---|
| float64 | 1.98 | 2.92 |
| float32 | 2.02 | 2.98 |
| float16 | 1.93 | 3.13 |

The largest value, 5.83, is float16 `walk_lstsq` on the Harshaw design
corr = 0, d = 5, φ = 0.5 at n = 500 (R = 10). The same configuration in float64
gives 4.44.

### Step-direction error (per-step diagnostics, n ≤ 200)

**Conditioned family, `walk_lstsq`:** medians by decade of κ(B_{A∖p}) at the step:

| | κ ~ 1 | 1e2 | 1e4 | 1e6 |
|---|---|---|---|---|
| float32 ‖BΔu‖₂ | 1e-7 | 3e-7 | 4e-7 | 6e-7 |
| float32 ‖Δu‖∞ | 8e-8 | 3e-6 | 2e-4 | 2e-2 |
| float16 ‖BΔu‖₂ | 1e-3 | 2e-3 | 2e-3 | 7e-3 |
| float16 ‖Δu‖∞ | 6e-4 | 2e-2 | 7e-1 | 3 |

**Fitted slopes** of the median log error against log κ, all families pooled:

| | ‖BΔu‖₂ float32 | ‖BΔu‖₂ float16 | ‖Δu‖∞ float32 | ‖Δu‖∞ float16 |
|---|---|---|---|---|
| `walk_lstsq` | 0.08 | 0.14 | 0.91 | 0.68 |
| `walk_gs_compress` | 1.87 | 1.24 | 2.09 | 1.95 |

- **`walk_harshaw_cholesky`:** κ(B_{A∖p}) stays below 100 on the Harshaw designs, too
  narrow a range for a slope.
- **Near-collinear pairs, `walk_lstsq` float32:** ‖BΔu‖₂ rises from 4e-8 at κ ~ 1 to
  4e-3 at κ ~ 1e5, then returns to about 1e-6 for κ ≥ 1e6.
- **Excluded points:** 13 float16 steps on near-collinear pairs have
  κ(B_{A∖p}) > 1e17, because B_{A∖p} is singular after rounding to float16. They are
  left out of `figures/step_error_vs_kappa.png`.

### Factor drift (Harshaw designs, `figures/factor_drift_phi*.png`)

- **`walk_harshaw_cholesky`:** ‖UᵀU − M‖_F / ‖M‖_F grows along the walk.
  - At n = 200 and φ = 0.01 it goes from about 5e-7 to 1e-3 in float32, and from about
    4e-3 to 0.1–0.3 in float16.
  - Float16 lines stop where runs fail.
  - The float16 maximum over all designs is 0.62 at n = 200 and 0.81 at n = 50.
- **With refactoring (k = 10):** a sawtooth with period 10, about 1e-7 to 5e-7 in
  float32 and 3e-4 to 5e-3 in float16. It rises to 0.08–0.14 (float16) in the last
  steps.
- **`walk_gs_compress`:** ‖CQ − I‖_F / √k starts near 1e-4 in float32 and near 1 in
  float16, then decreases as k shrinks. The float16 maximum is 1.47.

### Snapping

- **float64:** no tolerance snaps or clamps; forced-snap distance ≤ 4.4e-16.
- **float32:**
  - tolerance snaps: up to 223 per run (`walk_lstsq`, near-collinear, n = 500), and
    at most 0.07 per run for the other implementations;
  - clamps: at most 0.2 per run;
  - forced-snap distance ≤ 2.4e-7.
- **float16:**
  - tolerance snaps: up to 236 per run (`walk_lstsq`, near-collinear, n = 500) and
    about 95 per run (`walk_lstsq`, conditioned, n = 500);
  - clamps: at most 1.2 per run (`walk_gs_compress`);
  - forced-snap distance ≤ 2.0e-3.

### Surprises

- **Low precision removed the ψ₂ outliers on near-collinear pairs (η = 1e-3).**
  Float16 `walk_lstsq` produced none of the equal-sign-pair runs that set the float64
  ψ₂. Its ψ₂ ratio is therefore about 0.005, not near or above 1.
- **Float64 implementations diverge on ill-conditioned inputs.** `walk_gs_compress`
  and `walk_lstsq` take different paths in float64 at κ ≥ 1e6 and η ≤ 1e-3, while
  agreeing on every Harshaw design.
- **`walk_lstsq` never failed on the conditioned family,** including float16 at
  κ = 1e8. On that family its discrepancy-space step error is nearly independent of
  κ (slope ≈ 0.1), while its coefficient error grows about linearly in κ.
- **Refactoring removed the float16 downdate failures but not the ψ₂ inflation.**
  With k = 10, `walk_harshaw_cholesky` has almost no float16 failures, and its runs
  have the largest float16 ψ₂ ratios in the grid (up to 10.7 with R = 2000).
- **`walk_gs_compress` slowed sharply under load at n = 500.** In float64 it ran at
  9.6 s per run with 12 workers busy, against 0.41 s alone; in float32, 2.2 s against
  0.25 s. The other implementations slowed by about 2× under the same load.
