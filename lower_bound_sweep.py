"""Lower-bound matrix from Theorem 4.1 and GSW sub-Gaussianity ablation.

The matrix has unit-norm columns v_i = (e_0 + e_i) / sqrt(2) in R^{n+1}.
The walk injects a bounded adversarial error a = 2^{-sig_bits} into every
active non-pivot coordinate during the first L = floor(n/4) iterations,
which makes the e_0 discrepancy Omega(a n^2).  Sweeping n and the effective
precision exposes the unavoidable n^2 * 2^{-sig_bits} growth.

Usage:
    python lower_bound_sweep.py
    python lower_bound_sweep.py --n 50 100 200 400 800 1000 --sig-bits 10 11 12 13 14 20 30 40 52 --num-samples 200
"""

import os
os.environ.setdefault("chop_backend", "numpy")
# single-threaded BLAS so any parallel work does not oversubscribe the cores
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from typing import Callable
import matplotlib
matplotlib.use("Agg")  # non-interactive: save figures, never pop up a window
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import lpla
from precision_sweep import _subgaussian_sigma, _plot_inf_norm_ccdf


def lower_bound_matrix(n: int) -> np.ndarray:
    """B in R^{(n+1) x n} with columns v_i = (e_0 + e_i) / sqrt(2)."""
    B = np.zeros((n + 1, n))
    c = 1.0 / np.sqrt(2.0)
    B[0, :] = c
    B[1:, :] = np.eye(n) * c
    return B


def lower_bound_walk(
    B: np.ndarray,
    a: float,
    chop: Callable | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """One GSW rollout on the lower-bound matrix with adversarial error a.

    The construction follows Theorem 4.1: for the first L = floor(n/4)
    iterations, every active non-pivot coordinate receives an additive
    error a; the pivot and frozen coordinates receive none.  This produces
    an e_0 discrepancy of Theta(a n^2) when a is small enough.
    """
    m, n = B.shape
    r = chop if chop is not None else lambda x: x
    B0 = np.asarray(B, dtype=float)
    B = r(B0.copy())
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    L = n // 4

    for t in range(n):
        active = np.where(np.abs(z) < 1 - 1e-9)[0]
        k = active.size
        if k == 0:
            break

        p = active[0]
        free = active[active != p]

        u = np.zeros(n)
        u[p] = 1.0
        u[free] = float(r(np.array(-1.0 / k)))
        u = r(u)

        # feasible interval for the active coordinates, exactly as in gsw.py
        z_a = z[active]
        u_a = u[active]
        r1 = r((-1.0 - z_a) / u_a)
        r2 = r((1.0 - z_a) / u_a)
        lo = r(np.minimum(r1, r2))
        hi = r(np.maximum(r1, r2))
        delta_min = float(r(np.array(np.max(lo))))
        delta_max = float(r(np.array(np.min(hi))))
        total = float(r(np.array(abs(delta_max) + abs(delta_min))))
        if total < 1e-15:
            break

        prob = float(r(np.array(abs(delta_min) / total)))
        delta = delta_max if rng.random() < prob else delta_min
        delta = float(r(np.array(delta)))

        # ideal GSW step
        z = r(z + r(delta * u))
        # deterministic adversarial error e_t for the first L iterations
        if t < L:
            z[free] = r(z[free] + a)

        # freeze anything that hit a boundary
        z = r(np.where(np.abs(z) > 1 - 1e-9, np.sign(z), z))

    # final Bz measured in full float64 (row 0 is the e_0 projection)
    return B0 @ z


def _valid_for_adversarial(n: int, sig_bits: int) -> bool:
    """Error a = 2^{-sig_bits} must satisfy a < 1/(8n) for the construction."""
    a = 2.0 ** (-sig_bits)
    return a * 8.0 * n < 1.0


def lower_bound_sweep(
    n_values: list[int] | None = None,
    sig_bits_values: list[int] | None = None,
    num_samples: int = 1000,
    save_path: str = "lower_bound_sweep.png",
) -> None:
    """Run the lower-bound matrix across n and mantissa bit widths."""
    if n_values is None:
        n_values = [50, 100, 200, 400, 800, 1000]
    if sig_bits_values is None:
        sig_bits_values = [10, 11, 12, 13, 14, 30, 40, 52]

    fig, (ax_n, ax_bits, ax_cc) = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Lower-bound matrix: worst-case sub-Gaussian growth")

    # single pass over all (sig_bits, n) configs; reuse for every panel
    bz_samples = {}
    for sig_bits in tqdm(sig_bits_values, desc="sig_bits"):
        a = 2.0 ** (-sig_bits)
        chop = None if sig_bits == 52 else lpla.make_round(sig_bits)
        for n in n_values:
            B = lower_bound_matrix(n)
            bz_samples[(sig_bits, n)] = np.stack([
                lower_bound_walk(B, a, chop, seed=n + 12345 + s)
                for s in range(num_samples)
            ], axis=1)  # (m, num_samples)

    def sigma_est(bz):
        vals = bz[0]  # e_0 projection
        sm, _ = _subgaussian_sigma(vals)
        k = min(5, len(vals))
        fold_sigmas = [_subgaussian_sigma(fold)[0] for fold in np.array_split(vals, k)]
        return sm, float(np.std(fold_sigmas))

    # left plot: sigma vs n for each sig_bits
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    for sig_bits, col in zip(sig_bits_values, colors):
        sigmas, sigma_errs = [], []
        for n in n_values:
            sm, sm_err = sigma_est(bz_samples[(sig_bits, n)])
            sigmas.append(sm)
            sigma_errs.append(sm_err)
        ax_n.errorbar(n_values, sigmas, yerr=sigma_errs, marker="o", markersize=4, color=col, label=f"{sig_bits}b")

    ax_n.set_xlabel("n")
    ax_n.set_ylabel("σ (moment estimate)")
    ax_n.set_title("σ vs n")
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.legend(title="mantissa bits")

    # middle plot: sigma vs sig_bits for each n
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_values)))
    for n, col in zip(n_values, colors):
        sigmas, sigma_errs = [], []
        for sig_bits in sig_bits_values:
            sm, sm_err = sigma_est(bz_samples[(sig_bits, n)])
            sigmas.append(sm)
            sigma_errs.append(sm_err)
        ax_bits.errorbar(sig_bits_values, sigmas, yerr=sigma_errs, marker="s", markersize=4, color=col, label=f"n={n}")

    ax_bits.set_xlabel("mantissa bits (sig_bits)")
    ax_bits.set_ylabel("σ (moment estimate)")
    ax_bits.set_title("σ vs precision")
    ax_bits.set_yscale("log")
    ax_bits.legend(title="n")

    # right plot: ||Bz||_inf CCDF per sig_bits at the largest n
    n_max = max(n_values)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    for sig_bits, col in zip(sig_bits_values, colors):
        _plot_inf_norm_ccdf(ax_cc, bz_samples[(sig_bits, n_max)],
                            f"{sig_bits}b", col)
    ax_cc.set_yscale("log")
    ax_cc.set_xlabel("t²")
    ax_cc.set_ylabel("Pr[‖Bz‖∞ > t]")
    ax_cc.set_title(f"‖Bz‖∞ CCDF at n={n_max} (dashed: union bound)")
    ax_cc.legend(title="mantissa bits", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lower-bound matrix GSW sub-Gaussianity ablation"
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=None,
        help="list of n values (default: 50 100 200 400 800 1000)"
    )
    parser.add_argument(
        "--sig-bits", type=int, nargs="+", default=None,
        help="list of mantissa bit widths (default: 10 11 12 13 14 20 30 40 52)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="independent GSW walks per (n, sig_bits) pair"
    )
    parser.add_argument("--save", default="lower_bound_sweep.png",
                        help="output plot path")
    args = parser.parse_args()

    lower_bound_sweep(
        n_values=args.n,
        sig_bits_values=args.sig_bits,
        num_samples=args.num_samples,
        save_path=args.save,
    )
