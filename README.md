# Gram–Schmidt walk under reduced-precision arithmetic

Code for studying how low-precision (reduced-mantissa) arithmetic degrades the
Gram–Schmidt walk (Bansal–Dadush–Garg–Lovett). Given `B ∈ R^{m×n}` with
unit-norm columns, the walk produces a signing `z ∈ {−1,+1}^n` such that the
discrepancy `Bz` is small and every linear functional `d·Bz` is approximately
1-subgaussian. The experiments measure how the discrepancy and the subgaussian
constant degrade as the walk's inner least-squares solve is rounded to few
mantissa bits, and compare deterministic rounding against matched additive
noise.

Main phenomena studied:

- A **precision plateau**: at moderate mantissa widths the walk absorbs
  rounding error and matches the fp64 baseline; below a threshold the
  subgaussian scale inflates.
- **Scaling laws**: on real data (Higgs) the inflation is well described by
  `n·2^-b`; on an adversarial matrix it reaches the lower-bound rate
  `n²·2^-b`.
- **Rounding vs. noise**: matched i.i.d. additive noise accumulates
  incoherently (`√n·ε`), while rounding accumulates coherently.
- **Mechanism**: the per-step solve error is concentrated where the free-set
  size `k ≈ m`, i.e. where the active submatrix is nearly singular.

## Setup

```sh
conda env create -f environment.yml   # env name: gsw, python 3.13
conda activate gsw
```

or a plain venv with `numpy`, `scipy`, `matplotlib`, `tqdm`, `pandas`,
`scikit-learn`. Run the noise-control, reproducibility, and cache tests with
`python -m unittest -v test_rollouts`. The solver self-check is `python lpla.py`
(compares the hand-rolled solver against `np.linalg.lstsq` at full precision).

## Core modules

| file | contents |
|---|---|
| `gsw.py` | the walk. `gram_schmidt_walk(B, chop=None, noise=None, record_trajectory=False) → WalkResult`. `chop` routes the direction solve through `lpla.lstsq`; `noise` perturbs unfrozen `z` coordinates each step. `Bz` is always measured in fp64 on the unrounded input. |
| `lpla.py` | the precision model. `make_round(sig_bits)` is a round-to-nearest-even mantissa rounder with unbounded exponent (rounding, not IEEE range). `lstsq` is a shape-adaptive Householder QR / LQ min-norm solver with every elementary op rounded; reductions accumulate in fp64. |
| `rollouts.py` | parallel Monte-Carlo walks over `ProcessPoolExecutor` with per-rollout `SeedSequence` streams; returns discrepancy means, direction projections, and full `Bz` samples. |
| `walk_step.py` | interactive single-walk step-through. |

## Gaussian noise controls and new-run provenance

Main-walk sweeps now default to **no added Gaussian noise**, including their
float64 references. Use `--noise-std VALUE` for a finite, nonnegative standard
deviation, or `--no-noise` to disable it explicitly. These options are available
in `n_subgauss.py`, `bits_subgauss.py`, `higgs_sweep.py`, `precision_sweep.py`,
`shape_persist.py`, `higgs_norms.py`, `higgs_qq.py`, `identity_sweep.py`,
`n_sweep.py`, and `phase_heatmap.py`. Their callable sweep functions also accept
`noise_std=0.0`. The lower-bound experiments retain their separate deterministic
error model; the dtype-based suite in `more_experiments/` is unchanged.

```sh
python n_subgauss.py --no-noise --save ablation.png
python n_subgauss.py --noise-std 2.3283064365386963e-10 --save ablation.png
python identity_sweep.py --n 4 8 --sig-bits 3 52 --num-samples 10 --workers 1 --no-noise
```

The first two commands use standard deviations 0 and `2^-32`, respectively.
Outputs receive a `_noise<standard-deviation>` suffix even with `--save`:
`ablation_noise0.png` and `ablation_noise2.3283064365386963e-10.png`, with
correspondingly named caches. Existing unlabelled figures and caches are not
renamed or rewritten. Each rollout batch prints its noise parameters, and new
figures display the selected setting. New NPZ caches store a JSON `metadata`
field; PNGs embed the same JSON in their `Description` field. Metadata records
noise parameters, seed, experiment settings, RNG scheme, timestamp, Git revision,
and whether the checkout has local changes. A dirty checkout is recorded as
such, not represented as an exact committed revision.

Cache readers reject missing metadata and mismatched requested settings. In
particular, an old cache cannot be silently reused as a zero-noise result.
For `--plot-only`, pass the same experiment settings and base `--save` path as
in the generating command. Replotting preserves the cache's generating metadata.
Heatmap reuse is explicit and checked:

```sh
python phase_heatmap.py --no-noise --rounding-cache n_subgauss_higgs_noise0_cache.npz
```

The dedicated `noise_vs_chop.py` comparison has two arms. `--noise-std` /
`--no-noise` select the noise setting of the cached rounding arm. Its intentional
noise-only arm uses `--noise-scale S` times `2^-b` (default `S=1`); set
`--noise-scale 0` to disable that arm's added noise. Both settings are recorded
and labelled, and the output filename also includes `_scale<S>`. Generate a
compatible `n_subgauss.py` cache first or provide `--rounding-cache PATH`.

Gaussian draws now use a separate child RNG stream per rollout, so sampling
noise does not itself consume walk random draws. Results are reproducible across
worker counts for fixed inputs and seeds, but adaptive trajectories can still
diverge. This RNG change means noisy runs are not bitwise reproductions of the
old shared-RNG implementation.


## Experiment scripts

Scripts are run directly. Several also save summary NPZ caches for `--plot-only`.
The names below identify the existing figures; new main-walk outputs use the
noise suffix described above.

| script | figure | question |
|---|---|---|
| `phase_heatmap.py` | `phase_heatmap.png` | σ̂ over the (n, bits) plane with the empirical knee `n ≈ 10·2^b`. |
| `n_subgauss.py` | `n_subgauss_higgs.png` | σ̂ vs n on nested prefixes of the Higgs `2×n` matrix, incl. a fixed-threshold exceedance check and direction-spread bands. |
| `bits_subgauss.py` | `bits_subgauss.png` | σ̂ vs mantissa bits at fixed n, with plateau/growth model fits. |
| `noise_vs_chop.py` | `noise_vs_chop.png` | rounding vs matched additive noise, and the `n·ε` / `√n·ε` collapse comparison. |
| (composite of the two npz caches) | `scaling_collapse.png` | Higgs collapse on `n·2^-b` and lower-bound collapse on `n²·2^-b` side by side, with fitted slopes. Regenerate from `n_subgauss_higgs_cache.npz` + `lb_n_subgauss_cache.npz`. |
| `lb_subgauss.py` / `lower_bound_sweep.py` | `lb_n_subgauss.png`, `lb_bits_subgauss.png` | adversarial matrix `v_i = (e_0 + e_i)/√2` with injected per-step error; realizes `Ω(a·n²)` growth. Open markers flag cells violating `a < 1/(8n)`. |
| `identity_sweep.py` | `identity_sweep.png` | `B = I_n` control; the output stays near the Rademacher baseline at every precision. |
| `step_landing.py` | `results/step_landing_*.png` | non-accumulating diagnostic: runs the exact fp64 trajectory while also solving each step in low precision, reporting `u_lp − u_id` keyed by free-set size `k`. Shows the `k ≈ m` spike; reports median/IQR and an IQR-based effective σ. |
| `higgs_sweep.py` | `higgs_sweep.png` | precision sweep on the Higgs matrix (discrepancy + σ̂ + CCDF). |
| `shape_persist.py` | `shape_persist.png` | standardized tail shape of `d·Bz` as σ inflates. |
| `higgs_qq.py` / `higgs_norms.py` | `higgs_qq.png`, `higgs_norms.png` | Gaussianity diagnostics: Q–Q plot and Mahalanobis-whitened norms vs χ₂/max-of-iid references. |
| `n_sweep.py` / `precision_sweep.py` | `results/n_sweep_*.png`, `precision_sweep.png` | earlier exploratory sweeps; home of the shared test-direction and σ̂-estimator helpers and the `(t², log-survival)` CCDF panel. |

## Conventions

- `sig_bits` = mantissa fraction bits retained; `2^-sig_bits` is the rounding
  resolution. fp8-E5M2 ≙ 2, fp8-E4M3 ≙ 3, bf16 ≙ 7, fp16 ≙ 10, fp32 ≙ 23,
  fp64 ≙ 52.
- The clustered ensemble is `B = u·1ᵀ + m^{-1/2}·G` (Gaussian `G`, random unit
  `u`), columns normalized. Higgs uses the two low-level angular-momentum
  features of the Higgs two-sample benchmark, loaded via `higgs_sweep.load_higgs_matrix`.
- σ̂ is estimated per direction three ways: a moment bound over `k ≤ 3`, a tail
  bound from empirical order statistics, and the ψ₂ Orlicz norm by bisection;
  reported as a max over a direction set (random unit vectors + top singular
  vectors + `e_0` for the adversarial matrix).
- Plots use matplotlib's `Agg` backend and save to the repo root or `results/`.
- The low-precision path is ~20× slower than fp64. Use the `workers` argument
  to parallelize; near-singular solves around `k ≈ m` emit warnings at very
  low bits, which is expected.

## `more_experiments/`

A self-contained second study (`gsw_stability` package) using **true IEEE
float64/32/16 dtypes** rather than the mantissa-only model, comparing three
step-direction implementations (from-scratch QR least squares, the
Cholesky/Woodbury scheme of GSWDesign.jl, and the explicit-inverse
Sherman–Morrison scheme) under a shared walk skeleton with common random
numbers. Includes a pytest suite, resumable result grids, and a cited error
analysis in `stability_references.md`. See `more_experiments/README.md`.

## `old/`

Retired implementations (Gram-matrix/Woodbury walk, early sweeps). Not
imported by current code.
