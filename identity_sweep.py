"""Sub-Gaussianity of the identity matrix B = I_n.

This is the tight Rademacher instance: the optimal assignment is z_i in {+1,-1},
so for any unit test direction d, d^T B z is a sum of independent Rademachers
and has sigma^2 = 1.  The GSW should therefore give sigma ≈ 1 for all n and
all mantissa bit widths, unlike the lower-bound matrix.

Usage:
    python identity_sweep.py
    python identity_sweep.py --n 100 200 500 1000 --sig-bits 10 13 20 30 40 52 --num-samples 1000
"""

import os
os.environ.setdefault("chop_backend", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import rollouts
from precision_sweep import (
    _subgaussian_sigma, _test_directions, _plot_inf_norm_ccdf,
)


def identity_sweep(
    n_values: list[int] | None = None,
    sig_bits_values: list[int] | None = None,
    num_samples: int = 1000,
    num_random_dirs: int = 20,
    workers: int | None = None,
    save_path: str = "identity_sweep.png",
    noise_std: float = 0.0,
) -> None:
    """Run GSW on B = I_n across n and mantissa bits."""
    if n_values is None:
        n_values = [100, 200, 500, 1000]
    if sig_bits_values is None:
        sig_bits_values = [10, 13, 20, 30, 40, 52]
    save_path = rollouts.output_path(save_path, noise_std)
    metadata = rollouts.experiment_metadata(noise_std, seed=0, matrix="identity",
                                           n_values=list(n_values), sig_bits=list(sig_bits_values),
                                           num_samples=num_samples, num_dirs=num_random_dirs)
    np.random.seed(0)

    # single pass over all (sig_bits, n) configs; reuse for every panel
    mom_lists = {}
    bz_samples = {}
    for sig_bits in tqdm(sig_bits_values, desc="sig_bits"):
        for n in n_values:
            B = np.eye(n)
            directions = _test_directions(B, num_random=num_random_dirs)
            _, projections, bz = rollouts.run_samples(
                B, directions, num_samples,
                sig_bits=None if sig_bits == 52 else sig_bits,
                noise_std=noise_std,
                workers=workers,
                seed=0,
            )
            mom_lists[(sig_bits, n)] = [
                _subgaussian_sigma(proj)[0] for proj in projections
            ]
            bz_samples[(sig_bits, n)] = bz

    fig, (ax_n, ax_bits, ax_cc) = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(r"B = I_n: tight Rademacher instance ($\sigma^2 = 1$)")

    # left: sigma vs n for each sig_bits
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    for sig_bits, col in zip(sig_bits_values, colors):
        sigmas = [max(mom_lists[(sig_bits, n)]) for n in n_values]
        sigma_errs = [np.std(mom_lists[(sig_bits, n)]) for n in n_values]
        ax_n.errorbar(n_values, sigmas, yerr=sigma_errs, marker="o", markersize=4, color=col, label=f"{sig_bits}b")

    ax_n.set_xlabel("n")
    ax_n.set_ylabel("σ (moment estimate)")
    ax_n.set_title("σ vs n")
    ax_n.set_xscale("log")
    ax_n.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="σ=1")
    ax_n.legend(title="mantissa bits")

    # middle: sigma vs sig_bits for each n
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_values)))
    for n, col in zip(n_values, colors):
        sigmas = [max(mom_lists[(sig_bits, n)]) for sig_bits in sig_bits_values]
        sigma_errs = [np.std(mom_lists[(sig_bits, n)]) for sig_bits in sig_bits_values]
        ax_bits.errorbar(sig_bits_values, sigmas, yerr=sigma_errs, marker="s", markersize=4, color=col, label=f"n={n}")

    ax_bits.set_xlabel("mantissa bits (sig_bits)")
    ax_bits.set_ylabel("σ (moment estimate)")
    ax_bits.set_title("σ vs precision")
    ax_bits.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="σ=1")
    ax_bits.legend(title="n")

    # right: ||Bz||_inf CCDF per n at highest (solid) and lowest (dotted) precision
    hi_bits, lo_bits = max(sig_bits_values), min(sig_bits_values)
    for n, col in zip(n_values, colors):
        _plot_inf_norm_ccdf(ax_cc, bz_samples[(hi_bits, n)],
                            f"n={n}, {hi_bits}b", col)
        _plot_inf_norm_ccdf(ax_cc, bz_samples[(lo_bits, n)],
                            f"n={n}, {lo_bits}b", col, linestyle=":")
    ax_cc.set_yscale("log")
    ax_cc.set_xlabel("t²")
    ax_cc.set_ylabel("Pr[‖Bz‖∞ > t]")
    ax_cc.set_title("‖Bz‖∞ CCDF (dashed: union bound)")
    ax_cc.legend(fontsize=7)

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sub-Gaussianity of B = I_n (tight Rademacher instance)"
    )
    parser.add_argument("--n", type=int, nargs="+", default=None,
                        help="list of n values (default: 100 200 500 1000)")
    parser.add_argument("--sig-bits", type=int, nargs="+", default=None,
                        help="list of mantissa bit widths (default: 10 13 20 30 40 52)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="independent GSW walks per (n, sig_bits) pair")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of parallel workers")
    parser.add_argument("--save", default="identity_sweep.png",
                        help="output plot path")
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()

    identity_sweep(
        n_values=args.n,
        sig_bits_values=args.sig_bits,
        num_samples=args.num_samples,
        workers=args.workers,
        save_path=args.save, noise_std=args.noise_std,
    )
