"""GSW precision ablation on the Higgs data matrix used in Low-Rank Thinning.

This script takes the two low-level angular-momentum features used by
Liu et al. (2020) / Domingo-Enrich et al. (2023) (the Higgs two-sample
experiment in Low-Rank Thinning, Sec. 6.2), normalizes the columns, and
sweeps mantissa precision exactly like precision_sweep.py.

Usage:
    python higgs_sweep.py
    python higgs_sweep.py --n 1024 --num-samples 1000 --workers 4
"""

import os
os.environ.setdefault("chop_backend", "numpy")
# single-threaded BLAS so parallel rollout workers don't oversubscribe the cores
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import matplotlib
matplotlib.use("Agg")  # non-interactive: save figures, never pop up a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from precision_sweep import (
    _subgaussian_sigma, _test_directions, _plot_inf_norm_ccdf,
)
import rollouts


def load_higgs_matrix(
    n: int,
    features: list[int] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Load the Higgs data from OpenML and return a column-normalised B matrix."""
    if features is None:
        features = [0, 1]  # two low-level angular-momentum features

    print(f"Loading Higgs from OpenML (data_id=23512), "
          f"features {features}, n={n}...")
    X, _ = fetch_openml(
        data_id=23512,
        as_frame=False,
        return_X_y=True,
        parser='liac-arff',
    )
    X = np.asarray(X, dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    X = X[idx][:, features]

    B = X.T  # GSW expects (m, n) with unit-norm columns
    norms = np.linalg.norm(B, axis=0)
    B = B / np.where(norms == 0, 1.0, norms)
    return B


def higgs_sweep(
    n: int = 500,
    num_samples: int = 1000,
    num_random_dirs: int = 20,
    sig_bits_min: int = 2,
    sig_bits_max: int = 52,
    sig_bits_step: int = 10,
    workers: int | None = None,
    seed: int = 0,
    save_path: str = "higgs_sweep.png",
    noise_std: float = 0.0,
) -> None:
    """Run GSW on a Higgs data submatrix across mantissa bit widths."""
    save_path = rollouts.output_path(save_path, noise_std)
    metadata = rollouts.experiment_metadata(noise_std, seed=seed, matrix="higgs", matrix_n=n,
                                           num_samples=num_samples, num_dirs=num_random_dirs,
                                           sig_bits=list(range(sig_bits_min, sig_bits_max + 1, sig_bits_step)))
    np.random.seed(seed)
    B = load_higgs_matrix(n, seed=seed)
    m, n_actual = B.shape
    print(f"B shape: {m} × {n_actual}")

    sig_range = range(sig_bits_min, sig_bits_max + 1, sig_bits_step)
    sig_list = list(sig_range)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_list)))
    fig, (ax_disc, ax_sg, ax_cc) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"GSW precision sweep — Higgs data ({m}×{n_actual}), "
        f"{num_samples} samples"
    )

    mean_discrepancies = []
    mean_discrepancy_errs = []
    sigma_moms = []
    sigma_mom_errs = []
    sigma_tails = []
    sigma_tail_errs = []

    directions = _test_directions(B, num_random=num_random_dirs)

    for sig_bits, col in zip(sig_list, tqdm(colors, desc="sig_bits")):
        bz_means, projections, bz_samples = rollouts.run_samples(
            B, directions, num_samples,
            sig_bits=None if sig_bits == 52 else sig_bits,
            noise_std=noise_std,
            workers=workers,
            seed=seed,
        )
        _plot_inf_norm_ccdf(ax_cc, bz_samples, f"{sig_bits}b", col)
        abs_bz = np.abs(bz_means)
        mean_discrepancies.append(abs_bz.mean())
        mean_discrepancy_errs.append(abs_bz.std())

        # worst-case sub-Gaussian parameter over all test directions
        mom_list = []
        tail_list = []
        for proj in projections:
            sm, st = _subgaussian_sigma(proj)
            mom_list.append(sm)
            tail_list.append(st)
        sigma_moms.append(max(mom_list))
        sigma_mom_errs.append(np.std(mom_list))
        sigma_tails.append(max(tail_list))
        sigma_tail_errs.append(np.std(tail_list))

    ax_disc.errorbar(sig_list, mean_discrepancies, yerr=mean_discrepancy_errs, marker="o", markersize=3)
    ax_disc.set_xlabel("mantissa bits (sig_bits)")
    ax_disc.set_ylabel("mean of |Bz| (dim 0)")
    ax_disc.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="ideal (0)")
    ax_disc.legend()

    ax_sg.errorbar(sig_list, sigma_moms, yerr=sigma_mom_errs, marker="o", markersize=3, label="moments")
    ax_sg.errorbar(sig_list, sigma_tails, yerr=sigma_tail_errs, marker="s", markersize=3, label="tails")
    ax_sg.set_xlabel("mantissa bits (sig_bits)")
    ax_sg.set_ylabel("estimated σ  (subgaussian parameter)")
    ax_sg.legend()

    ax_cc.set_yscale("log")
    ax_cc.set_xlabel("t²")
    ax_cc.set_ylabel("Pr[‖Bz‖∞ > t]")
    ax_cc.set_title("‖Bz‖∞ CCDF (dashed: union bound)")
    ax_cc.legend(title="mantissa bits")

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GSW mantissa-precision ablation on the Higgs data matrix"
    )
    parser.add_argument("--n", type=int, default=500,
                        help="number of Higgs points to use")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="independent GSW runs per precision level")
    parser.add_argument("--workers", type=int, default=None,
                        help="number of parallel workers (default: half cores)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--save", type=str, default="higgs_sweep.png",
                        help="output plot path")
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()

    higgs_sweep(
        n=args.n,
        num_samples=args.num_samples,
        workers=args.workers,
        seed=args.seed,
        save_path=args.save, noise_std=args.noise_std,
    )
