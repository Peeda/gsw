import os
os.environ.setdefault("chop_backend", "numpy")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gsw
import lpla


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


def n_sweep(
    m: int,
    n_values,
    num_samples: int = 250,
    *,
    sig_bits: int | None = None,
    exp_bits: int = 11,
    noise_mean: float = 0.0,
    noise_std: float = 0.0,
) -> None:
    """Sweep n with fixed m; supports chop mode (exp_bits+sig_bits) or noise mode (noise_mean/noise_std), not both.

    Standard format parameters:
        fp8 E4M3:  exp_bits=4, sig_bits=3
        fp8 E5M2:  exp_bits=5, sig_bits=2
        fp16:      exp_bits=5, sig_bits=10
        bfloat16:  exp_bits=8, sig_bits=7
        fp32:      exp_bits=8, sig_bits=23
    """
    has_chop  = sig_bits is not None
    has_noise = noise_mean != 0.0 or noise_std != 0.0
    if has_chop and has_noise:
        raise ValueError("specify either sig_bits or noise parameters, not both")

    chop  = lpla.make_round(exp_bits, sig_bits) if has_chop else None
    noise = (lambda size: np.random.normal(noise_mean, noise_std, size)) if has_noise else None

    if has_chop:
        desc = f"chop=e{exp_bits}m{sig_bits} ({1 + exp_bits + sig_bits}-bit)"
        tag  = f"chop_e{exp_bits}m{sig_bits}"
    else:
        desc = f"noise=N({noise_mean}, {noise_std})"
        tag  = f"noise_N({noise_mean},{noise_std})"
    title    = f"GSW n sweep  —  m={m}, {desc}, {num_samples} samples"
    savepath = f"results/n_sweep_{tag}.png"

    mean_discrepancies = []
    sigma_moms  = []
    sigma_tails = []

    for n in tqdm(n_values, desc=f"n  [{tag}]"):
        u = np.random.randn(m); u /= np.linalg.norm(u)
        epsilon = 1 / np.sqrt(m)
        B = u[:, None] + epsilon * np.random.randn(m, n)
        B /= np.linalg.norm(B, axis=0)

        directions = _test_directions(B)  # (D, m)
        bz_means = np.zeros(num_samples)
        projections = np.zeros((len(directions), num_samples))

        for i in tqdm(range(num_samples), desc=f"  n={n}", leave=False):
            r = gsw.gram_schmidt_walk(B, chop=chop, noise=noise)
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
    fig.suptitle(title)

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
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


def mn_sweep(
    N_values,
    num_samples: int = 250,
    *,
    sig_bits: int | None = None,
    exp_bits: int = 11,
    noise_mean: float = 0.0,
    noise_std: float = 0.0,
) -> None:
    """Sweep N with m=n=N (square B); supports chop mode (exp_bits+sig_bits) or noise mode (noise_mean/noise_std), not both.

    Standard format parameters:
        fp8 E4M3:  exp_bits=4, sig_bits=3
        fp8 E5M2:  exp_bits=5, sig_bits=2
        fp16:      exp_bits=5, sig_bits=10
        bfloat16:  exp_bits=8, sig_bits=7
        fp32:      exp_bits=8, sig_bits=23
    """
    has_chop  = sig_bits is not None
    has_noise = noise_mean != 0.0 or noise_std != 0.0
    if has_chop and has_noise:
        raise ValueError("specify either sig_bits or noise parameters, not both")

    chop  = lpla.make_round(exp_bits, sig_bits) if has_chop else None
    noise = (lambda size: np.random.normal(noise_mean, noise_std, size)) if has_noise else None

    if has_chop:
        desc = f"chop=e{exp_bits}m{sig_bits} ({1 + exp_bits + sig_bits}-bit)"
        tag  = f"chop_e{exp_bits}m{sig_bits}"
    else:
        desc = f"noise=N({noise_mean}, {noise_std})"
        tag  = f"noise_N({noise_mean},{noise_std})"
    title    = f"GSW m=n sweep  —  {desc}, {num_samples} samples"
    savepath = f"results/mn_sweep_{tag}.png"

    mean_discrepancies = []
    sigma_moms  = []
    sigma_tails = []

    for N in tqdm(N_values, desc=f"N  [{tag}]"):
        u = np.random.randn(N); u /= np.linalg.norm(u)
        epsilon = 1 / np.sqrt(N)
        B = u[:, None] + epsilon * np.random.randn(N, N)
        B /= np.linalg.norm(B, axis=0)

        directions = _test_directions(B)  # (D, N)
        bz_means = np.zeros(num_samples)
        projections = np.zeros((len(directions), num_samples))

        for i in tqdm(range(num_samples), desc=f"  N={N}", leave=False):
            r = gsw.gram_schmidt_walk(B, chop=chop, noise=noise)
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

    N_list = list(N_values)
    fig, (ax_disc, ax_sg) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    ax_disc.plot(N_list, mean_discrepancies, marker="o", markersize=3)
    ax_disc.set_xlabel("N (m = n)")
    ax_disc.set_ylabel("mean of Bz")
    ax_disc.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="ideal (0)")
    ax_disc.legend()

    ax_sg.plot(N_list, sigma_moms,  marker="o", markersize=3, label="moments")
    ax_sg.plot(N_list, sigma_tails, marker="s", markersize=3, label="tails")
    ax_sg.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="ideal (1)")
    ax_sg.set_xlabel("N (m = n)")
    ax_sg.set_ylabel("estimated σ (subgaussian parameter)")
    ax_sg.legend()

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    m, n_values, num_samples = 30, range(500, 3001, 500), 250
    N_values = range(25, 201, 50)
    mn_sweep(N_values, num_samples, noise_mean=-2**(-16), noise_std=0.0)
    # n_sweep(m, n_values, num_samples, noise_mean=-2**(-10), noise_std=0.0)
    # n_sweep(m, n_values, num_samples, noise_mean=0.0, noise_std=2**(-8))
    # n_sweep(m, n_values, num_samples, sig_bits=3,  exp_bits=4)  # fp8 E4M3
    # n_sweep(m, n_values, num_samples, sig_bits=2,  exp_bits=5)  # fp8 E5M2
    # n_sweep(m, n_values, num_samples, sig_bits=10, exp_bits=5)  # fp16
    # n_sweep(m, n_values, num_samples, sig_bits=7,  exp_bits=8)  # bfloat16
