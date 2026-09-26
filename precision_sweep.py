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
from tqdm import tqdm

import rollouts


def _test_directions(B: np.ndarray, num_random: int = 20) -> np.ndarray:
    """Unit test directions in R^m: random + top left singular vectors of B."""
    m = B.shape[0]
    random_dirs = np.random.standard_normal((num_random, m))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    k = min(U.shape[1], 10)
    svd_dirs = U[:, :k].T  # shape (k, m)
    return np.vstack([random_dirs, svd_dirs])  # shape (num_random + k, m)


def _subgaussian_sigma(vals: np.ndarray) -> tuple[float, float]:
    """
    Estimate the subgaussian parameter sigma two ways.

    Moment estimate: tightest lower bound across k=1,2,3 from
        sigma >= (E[X^{2k}] / (2k-1)!!)^{1/(2k)}

    Tail estimate: invert the tail bound P(|X|>t) <= 2 exp(-t^2/(2 sigma^2))
    at every empirical order statistic and take the max over the upper fifth.
    """
    N = len(vals)
    sigma_mom = max(
        np.mean(vals ** 2) ** 0.5,           # k=1, (2k-1)!! = 1
        (np.mean(vals ** 4) / 3) ** 0.25,    # k=2, (2k-1)!! = 3
        (np.mean(vals ** 6) / 15) ** (1/6),  # k=3, (2k-1)!! = 15
    )

    abs_sorted = np.sort(np.abs(vals))[::-1]          # descending order stats
    ranks = np.arange(1, N + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_candidates = abs_sorted / np.sqrt(2 * np.log(2 * N / ranks))
    # only look at the upper tail (top 20%) to avoid log≈0 instability near median
    sigma_tail = float(np.nanmax(sigma_candidates[: max(1, N // 5)]))

    return sigma_mom, sigma_tail


def _plot_inf_norm_ccdf(ax, bz_samples, label, color, linestyle="-"):
    """CCDF of ||Bz||_inf on (t^2, log-survival) axes + union-bound reference.

    Pr[||Bz||_inf > t] <= sum_i 2 exp(-t^2 / (2 sigma_i^2)) with sigma_i the
    per-coordinate moment estimate. On these axes a Gaussian tail is a straight
    line with slope -1/(2 sigma^2).

    bz_samples: (m, num_samples) array of Bz vectors (bz_samples[i] = coord i).
    """
    t = np.sort(np.abs(bz_samples).max(axis=0))
    surv = np.arange(len(t), 0, -1) / len(t)
    ax.plot(t ** 2, surv, linestyle, color=color, label=label)
    sig_i = np.maximum(
        [_subgaussian_sigma(coord)[0] for coord in bz_samples], 1e-12
    )
    ub = (2.0 * np.exp(-(t ** 2) / (2.0 * sig_i[:, None] ** 2))).sum(axis=0)
    ax.plot(t ** 2, np.clip(ub, None, 1.0), "--", color=color,
            alpha=0.4, linewidth=1)


def precision_sweep(m: int, n: int, num_samples: int = 10000, *, workers: int | None = None,
                    noise_std: float = 0.0, seed: int = 0, save_path: str = "precision_sweep.png") -> None:
    """Sweep mantissa bits 2–52; plot mean of Bz (dim 0) and subgaussianity.

    workers: number of processes for the Monte-Carlo rollouts (default ≈ physical cores;
    pass workers=1 to run serially).
    """
    save_path = rollouts.output_path(save_path, noise_std)
    metadata = rollouts.experiment_metadata(noise_std, seed=seed, matrix="clustered", m=m, n=n,
                                           num_samples=num_samples, sig_bits=list(range(2, 53, 10)))
    np.random.seed(seed)
    u = np.random.randn(m); u /= np.linalg.norm(u)
    epsilon = 1 / np.sqrt(m)
    B = u[:, None] + epsilon * np.random.randn(m, n)
    B /= np.linalg.norm(B, axis=0)

    # B = np.random.standard_normal((m, n))
    # B[:, 0] *= 10
    # B /= np.linalg.norm(B, axis=0, keepdims=True)

    # B = np.eye(m)

    sig_range = range(2, 53, 10)
    sig_list = list(sig_range)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_list)))
    fig, (ax_disc, ax_sg, ax_cc) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"GSW precision sweep  —  B: ({m}×{n}), {num_samples} samples")

    mean_discrepancies = []
    mean_discrepancy_errs = []
    sigma_moms  = []
    sigma_mom_errs  = []
    sigma_tails = []
    sigma_tail_errs = []

    directions = _test_directions(B)  # (D, m)

    for sig_bits, col in zip(sig_list, tqdm(colors, desc="sig_bits")):
        bz_means, projections, bz_samples = rollouts.run_samples(
            B, directions, num_samples,
            sig_bits=None if sig_bits == 52 else sig_bits,
            noise_std=noise_std, workers=workers, seed=seed,
        )
        _plot_inf_norm_ccdf(ax_cc, bz_samples, f"{sig_bits}b", col)
        abs_bz = np.abs(bz_means)
        mean_discrepancies.append(abs_bz.mean())
        mean_discrepancy_errs.append(abs_bz.std())

        # Max subgaussian parameter over all test directions
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

    ax_sg.errorbar(sig_list, sigma_moms,  yerr=sigma_mom_errs,  marker="o", markersize=3, label="moments")
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
    parser = argparse.ArgumentParser(description="Clustered-matrix precision sweep")
    parser.add_argument("--m", type=int, default=500)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="precision_sweep.png")
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()
    precision_sweep(m=args.m, n=args.n, num_samples=args.num_samples,
                    workers=args.workers, noise_std=args.noise_std, seed=args.seed,
                    save_path=args.save)
