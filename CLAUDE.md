# CLAUDE.md

Research code for studying how **low-precision (reduced-mantissa) arithmetic** affects the
**Gram-Schmidt Walk (GSW)** — a discrepancy-minimization / self-balancing random walk. Given a
matrix `B` (m×n, unit-norm columns), GSW produces an assignment `z ∈ {−1,+1}^n` such that the
discrepancy vector `Bz = B @ z` is small and every linear functional `d·Bz` is ≈1-subgaussian.
The experiments ask: as we round the arithmetic in the walk's inner solve to few mantissa bits
(fp8-ish), how do the discrepancy and subgaussianity degrade, and how does that compare to
injecting additive noise?

## Module map

- **gsw.py** — the algorithm. `gram_schmidt_walk(B, chop=None, noise=None, record_trajectory=False)
  → WalkResult(assignment, Bz, trajectory)`. Each iteration: pick a pivot among the active
  (unfrozen) coords, solve `min_v ‖B_free·v + B_p‖` for the step direction `u` (pivot fixed to 1),
  take a martingale-preserving random step, freeze coords that hit ±1. `chop` (a rounding
  callable) routes the solve through `lpla.lstsq`; otherwise `np.linalg.lstsq`. `noise` adds a
  per-coordinate perturbation to `z` each step (the additive-noise model).
- **lpla.py** — low-precision linear algebra (the core of the precision modeling).
  - `make_round(sig_bits)` → a fast round-to-nearest-even **mantissa-only** rounder (via
    `frexp`/`ldexp`). Rounds the significand to `sig_bits` fraction bits with an **unbounded
    exponent, so overflow never happens**. This is the current rounder (replaced pychop — see
    Gotchas).
  - `lstsq(A, c, r)` → hand-rolled shape-adaptive **Householder QR (tall) / LQ min-norm (wide)**
    least-squares in which every op is rounded by `r`. Reductions round products but accumulate
    in fp64 (BLAS) — the solver is genuinely low-precision without per-accumulation-step cost.
    At full precision it matches `np.linalg.lstsq` to ~1e-15 (self-check in its `__main__`).
- **rollouts.py** — `run_samples(B, directions, num_samples, *, sig_bits, noise_mean, noise_std,
  workers, seed) → (bz_means, projections)`. Runs the independent Monte-Carlo walks across a
  `ProcessPoolExecutor`. Handles the two parallelism gotchas: independent per-worker RNG
  (`SeedSequence`) and single-threaded BLAS.
- **n_sweep.py** — `n_sweep(m, n_values, ...)` (fixed m, sweep n) and `mn_sweep(N_values, ...)`
  (square, m=n=N). Plots mean(Bz) and estimated subgaussian σ vs the swept dimension. Home of the
  shared `_test_directions` and `_subgaussian_sigma` helpers (duplicated into precision_sweep.py).
- **precision_sweep.py** — `precision_sweep(m, n, ...)`, sweeps mantissa bits 2–52 (combines
  rounding with a tiny 2⁻³² noise).
- **step_landing.py** — `step_landing(m, n, sig_bits, samples)`. Walks the **exact fp64
  trajectory** and at each step also solves in low precision, reporting the per-step additive
  "epsilon" (`u_lp − u_id`) — how the low-precision fixed-point step lands vs ideal — keyed by
  free-set size `k`. Non-accumulating (a fresh per-step comparison).
- **walk_step.py** — interactive step-through of a single walk (uses `record_trajectory`).
- **old/** — retired implementations (Woodbury/Gram-update walk, old sweeps, subgaussian tests).
  Not imported by anything current; ignore unless explicitly asked.

## Key parameters & conventions

- **`sig_bits`** = mantissa (fraction) bits kept; the *only* precision knob. `2^-sig_bits` is the
  rounding resolution. There is **no `exp_bits`** (removed) — the exponent is unbounded, overflow
  never happens, by design (we study rounding, not IEEE range). fp8-E4M3 ≙ sig_bits=3,
  E5M2 ≙ 2, fp16 ≙ 10, bf16 ≙ 7, fp32 ≙ 23.
- **B construction** (same everywhere): `u = randn(m); u/=‖u‖; B = u[:,None] + (1/√m)·randn(m,n);
  B /= ‖·‖columns`. Columns are unit-norm and clustered around a shared direction `u`.
- **Plotting**: matplotlib **`Agg` backend** (non-interactive — never pop up a window); figures
  saved to `results/`. Chop-mode filenames use `m{sig_bits}`; noise-mode use `noise_N(...)`.
- Run scripts directly, e.g. `python n_sweep.py`, `python step_landing.py`. Conda env `gsw`
  (python 3.13; numpy, matplotlib, tqdm, scipy, pandas, scikit-learn, dask). No test suite —
  verification is running `python lpla.py` (solver self-check) and the sweeps.

## Gotchas

- **The low-precision path is ~20× slower than fp64** (per-op rounding). The `chop=None` path
  stays on `np.linalg.lstsq` and is untouched/fast. Use `workers=` to parallelize (~5× on this
  6-physical-core box; saturates near physical, not logical, cores).
- **Ill-conditioning at `k ≈ m`**: when the free set becomes square, `B_free`'s columns are all
  near `u`, so it's nearly singular and the low-precision solve blows up (huge per-step error,
  even zero pivots at very low bits). This dominates `std`-based statistics — **prefer median/IQR**
  (step_landing.py reports an IQR-based robust σ for exactly this reason).
- **Rounder history**: pychop was the original rounder but (a) had ~80µs/call overhead that
  dominated runtime and (b) had inconsistent/non-physical overflow handling. `lpla.make_round`
  replaced it — bit-identical to pychop for in-range values, ~4× faster, no overflow. `pychop` is
  still in the env but no longer used in the active code.

## Findings so far (project notes)

- Rounding is largely **absorbed** by the walk: discrepancy (rms of Bz) is flat for sig_bits ≳ 6
  and only degrades below ~5 bits — a plateau. Mechanism: the step is the solution of a
  minimization (flat objective near the optimum → error is quadratically suppressed) and the
  output is a hard `sign(z)`.
- Equal-magnitude **additive noise perturbs the bulk more than rounding does**; the two are only
  cleanly distinguishable above the "absorption budget". Rounding's distinctive signature is the
  `k≈m` spike, which additive noise lacks.
- Relevant literature for formalizing this: backward error analysis (Higham), inexact/relaxed
  Krylov methods (Simoncini–Szyld) for the "outer method absorbs inexact inner solves" plateau,
  and probabilistic rounding-error analysis (Higham–Mary). Finite-precision analysis of GSW
  itself appears to be an open gap.
