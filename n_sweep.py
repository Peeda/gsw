import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gsw

NOISE_STD = 2**(-8)


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


def n_sweep(m: int, n_values, num_samples: int = 250) -> None:
    """Sweep n (number of columns) with fixed m; plot subgaussianity of Bz under N(0, 2^-8) noise."""
    noise = lambda size: np.random.normal(0.0, NOISE_STD, size)

    mean_discrepancies = []
    sigma_moms  = []
    sigma_tails = []

    for n in tqdm(n_values, desc="n"):
        u = np.random.randn(m); u /= np.linalg.norm(u)
        epsilon = 1 / np.sqrt(m)
        B = u[:, None] + epsilon * np.random.randn(m, n)
        B /= np.linalg.norm(B, axis=0)

        directions = _test_directions(B)  # (D, m)
        bz_means = np.zeros(num_samples)
        projections = np.zeros((len(directions), num_samples))

        for i in tqdm(range(num_samples), desc=f"  n={n}", leave=False):
            r = gsw.gram_schmidt_walk(B, chop=None, noise=noise)
            bz_means[i] = r.Bz.mean()
            projections[:, i] = directions @ r.Bz

        mean_discrepancies.append(bz_means.mean())

        sigma_mom_max = sigma_tail_max = 0.0
        for proj in projections:
            sm, st = _subgaussian_sigma(proj)
            sigma_mom_max  = max(sigma_mom_max,  sm)
            sigma_tail_max = max(sigma_tail_max, st)
        sigma_moms.append(sigma_mom_max)
        sigma_tails.append(sigma_tail_max)

    n_list = list(n_values)
    fig, (ax_disc, ax_sg) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"GSW n sweep  —  m={m}, noise=N(0, 2^(−8)), {num_samples} samples")

    ax_disc.plot(n_list, mean_discrepancies, marker="o", markersize=3)
    ax_disc.set_xlabel("n (number of columns)")
    ax_disc.set_ylabel("mean of Bz")
    ax_disc.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="ideal (0)")
    ax_disc.legend()

    ax_sg.plot(n_list, sigma_moms,  marker="o", markersize=3, label="moments")
    ax_sg.plot(n_list, sigma_tails, marker="s", markersize=3, label="tails")
    ax_sg.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="ideal (1)")
    ax_sg.set_xlabel("n (number of columns)")
    ax_sg.set_ylabel("estimated σ (subgaussian parameter)")
    ax_sg.legend()

    fig.tight_layout()
    fig.savefig("results/n_sweep.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    n_sweep(m=30, n_values=range(25, 1001, 50), num_samples=250)
