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
from precision_sweep import _subgaussian_sigma


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
) -> float:
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

    # projection onto e_0; final Bz measured in full float64
    d = np.zeros(m)
    d[0] = 1.0
    Bz = B0 @ z
    return float(d @ Bz)


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

    fig, (ax_n, ax_bits) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Lower-bound matrix: worst-case sub-Gaussian growth")

    # left plot: sigma vs n for each sig_bits
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    for sig_bits, col in zip(sig_bits_values, colors):
        a = 2.0 ** (-sig_bits)
        chop = None if sig_bits == 52 else lpla.make_round(sig_bits)
        valid_ns = []
        sigmas = []
        sigma_errs = []
        for n in tqdm(n_values, desc=f"sig_bits={sig_bits}"):
            B = lower_bound_matrix(n)
            vals = np.array([
                lower_bound_walk(B, a, chop, seed=n + 12345 + s)
                for s in range(num_samples)
            ])
            sm, _ = _subgaussian_sigma(vals)
            k = min(5, len(vals))
            fold_sigmas = [_subgaussian_sigma(fold)[0] for fold in np.array_split(vals, k)]
            sm_err = float(np.std(fold_sigmas))
            valid_ns.append(n)
            sigmas.append(sm)
            sigma_errs.append(sm_err)
        if valid_ns:
            ax_n.errorbar(valid_ns, sigmas, yerr=sigma_errs, marker="o", markersize=4, color=col, label=f"{sig_bits}b")

    ax_n.set_xlabel("n")
    ax_n.set_ylabel("σ (moment estimate)")
    ax_n.set_title("σ vs n")
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.legend(title="mantissa bits")

    # right plot: sigma vs sig_bits for each n
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_values)))
    for n, col in zip(n_values, colors):
        B = lower_bound_matrix(n)
        valid_bits = []
        sigmas = []
        sigma_errs = []
        for sig_bits in tqdm(sig_bits_values, desc=f"n={n}"):
            a = 2.0 ** (-sig_bits)
            chop = None if sig_bits == 52 else lpla.make_round(sig_bits)
            vals = np.array([
                lower_bound_walk(B, a, chop, seed=n + 12345 + s)
                for s in range(num_samples)
            ])
            sm, _ = _subgaussian_sigma(vals)
            k = min(5, len(vals))
            fold_sigmas = [_subgaussian_sigma(fold)[0] for fold in np.array_split(vals, k)]
            sm_err = float(np.std(fold_sigmas))
            valid_bits.append(sig_bits)
            sigmas.append(sm)
            sigma_errs.append(sm_err)
        if valid_bits:
            ax_bits.errorbar(valid_bits, sigmas, yerr=sigma_errs, marker="s", markersize=4, color=col, label=f"n={n}")

    ax_bits.set_xlabel("mantissa bits (sig_bits)")
    ax_bits.set_ylabel("σ (moment estimate)")
    ax_bits.set_title("σ vs precision")
    ax_bits.set_yscale("log")
    ax_bits.legend(title="n")

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
