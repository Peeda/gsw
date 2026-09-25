"""Rounding vs matched-magnitude additive noise — Higgs B, sigma-hat vs n.

For each fp format b, compares:
  * rounding:  chop = make_round(b)        (taken from n_subgauss cache)
  * noise:     additive per-step noise with std = 2^-b, fp64 arithmetic

Prediction: rounding error accumulates coherently (~n·2^-b) while independent
additive noise accumulates as a random walk (~sqrt(n)·2^-b) — different
exponents, so both should collapse on their own scaling variable.

Panels: (A) sigma-hat vs n, solid=rounding / dashed=noise;
        (B) rounding points vs n·2^-b (collapse check);
        (C) noise points vs sqrt(n)·2^-b (collapse check).

Usage: python noise_vs_chop.py [--plot-only]
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
from higgs_sweep import load_higgs_matrix
from n_subgauss import _directions_for, collect, FMT, _lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--dirs", type=int, default=64)
    p.add_argument("--t", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    cache_round = "n_subgauss_higgs_cache.npz"
    cache_noise = "noise_vs_chop_cache.npz"
    rng = np.random.default_rng(args.seed)

    # rounding stats from the n_sweep cache
    z = np.load(cache_round)
    sig_bits_values = [int(b) for b in z["sig_bits"]]
    n_values = [int(v) for v in z["n_values"]]
    stats_r = {(int(b), int(n)): {"sig_max": float(z["sig_max"][i, j])}
               for i, b in enumerate(sig_bits_values)
               for j, n in enumerate(n_values)}

    if args.plot_only:
        zn = np.load(cache_noise)
        stats_nz = {(int(b), int(n)): {"sig_max": float(zn["sig_max"][i, j])}
                    for i, b in enumerate(sig_bits_values)
                    for j, n in enumerate(n_values)}
    else:
        B_full = load_higgs_matrix(max(n_values), seed=args.seed)
        stats_nz = {}
        for sig_bits in tqdm(sig_bits_values, desc="noise sig_bits"):
            for n in n_values:
                B = B_full[:, :n]
                dirs = _directions_for(B, args.dirs, rng)
                _, projections, _ = rollouts.run_samples(
                    B, dirs, args.num_samples,
                    sig_bits=None, noise_std=2.0 ** (-sig_bits),
                    workers=args.workers, seed=args.seed,
                )
                stats_nz[(sig_bits, n)] = collect(projections, args.t)
        np.savez(cache_noise,
                 **{k: np.array([[stats_nz[(b, n)][k] for n in n_values]
                                 for b in sig_bits_values])
                    for k in stats_nz[sig_bits_values[0], n_values[0]]})
        print(f"cached noise stats -> {cache_noise}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    fig, (ax_s, ax_cr, ax_cn) = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle("Rounding vs matched additive noise — Higgs B (2x·), "
                 f"{args.num_samples} walks")

    for b, col in zip(sig_bits_values, colors):
        sr = [stats_r[(b, n)]["sig_max"] for n in n_values]
        sn = [stats_nz[(b, n)]["sig_max"] for n in n_values]
        ax_s.plot(n_values, sr, "o-", markersize=4, color=col,
                  label=_lab(b))
        ax_s.plot(n_values, sn, "x--", markersize=4, color=col, alpha=0.7)
        ax_cr.plot(np.array(n_values) * 2.0 ** (-b), sr, "o", markersize=4,
                   color=col, label=_lab(b))
        ax_cn.plot(np.sqrt(n_values) * 2.0 ** (-b), sn, "o", markersize=4,
                   color=col, label=_lab(b))

    ax_s.plot([], [], "k-", label="rounding")
    ax_s.plot([], [], "k--", label="additive noise")
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_s.set_xscale("log"); ax_s.set_yscale("log")
    ax_s.set_xlabel("n"); ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("σ̂ vs n — solid: rounding, dashed: noise")
    ax_s.legend(fontsize=6.5, ncol=2)

    xs = np.logspace(-6, 3, 200)
    ax_cr.plot(xs, np.sqrt(0.74 ** 2 + (0.08 * xs) ** 2), "k--", lw=0.8,
               label="√(0.74²+(0.08x)²)")
    ax_cr.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_cr.set_xscale("log"); ax_cr.set_yscale("log")
    ax_cr.set_xlabel("n · 2^{-bits}")
    ax_cr.set_title("rounding collapse on n·ε")
    ax_cr.legend(fontsize=7)

    ax_cn.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_cn.set_xscale("log"); ax_cn.set_yscale("log")
    ax_cn.set_xlabel("√n · 2^{-bits}")
    ax_cn.set_title("noise collapse on √n·ε")
    ax_cn.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("noise_vs_chop.png", dpi=150)
    plt.close(fig)
    print("saved noise_vs_chop.png")


if __name__ == "__main__":
    main()
