"""
Sweep over n and emulated mantissa-bit precision (10–52 bits)
to see how the empirical sub-Gaussian parameter sigma^2 of the
Gram-Schmidt Walk discrepancy varies.

For each (n, bits) pair we:
  1. Generate a random B (m x n) with unit-norm columns.
  2. Run GSW many times with mantissa_bits=bits.
  3. Estimate sigma^2 = sup_lambda  (2/lambda^2) log E[exp(lambda X)]
     over all coordinates j  (the "MGF sub-Gaussian parameter").
"""

import numpy as np
from multiprocessing import Pool
from gram_schmidt_walk import gram_schmidt_walk


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def stable_log_mgf(samples, lam):
    """log E[exp(lambda * X)] with log-sum-exp stability."""
    lx = lam * samples
    c = np.max(lx)
    return c + np.log(np.mean(np.exp(lx - c)))


def estimate_sigma_sq(disc_samples, lam_grid):
    """
    Estimate the sub-Gaussian parameter sigma^2 from discrepancy
    samples (num_samples, m).

    Returns the worst-case (over coordinates) of
        sup_lambda  (2/lambda^2) log E[exp(lambda X_j)]
    where X_j = (B z_t)_j centered to zero mean.
    """
    m = disc_samples.shape[1]
    worst = 0.0
    for j in range(m):
        s = disc_samples[:, j] - disc_samples[:, j].mean()
        for lam in lam_grid:
            val = 2.0 * stable_log_mgf(s, lam) / (lam ** 2)
            if val > worst:
                worst = val
    return worst


def _single_gsw(B_f64, mantissa_bits, seed):
    """Run one GSW and return the discrepancy row (in float64)."""
    z = gram_schmidt_walk(B_f64, seed=seed, mantissa_bits=mantissa_bits)
    return B_f64 @ z


def run_one(n, m, mantissa_bits, num_samples, lam_grid, seed=0):
    """Run GSW num_samples times at the given precision, return sigma^2."""
    rng = np.random.default_rng(seed)

    B_f64 = rng.standard_normal((m, n))
    B_f64 /= np.linalg.norm(B_f64, axis=0, keepdims=True)

    seeds = rng.integers(2**62, size=num_samples)
    with Pool() as pool:
        rows = pool.starmap(_single_gsw,
                            [(B_f64, mantissa_bits, int(s)) for s in seeds])
    disc = np.array(rows)

    return estimate_sigma_sq(disc, lam_grid)


# ------------------------------------------------------------------ #
#  Main sweep
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = 16
    #n_values = [20, 50, 100, 500, 1000]
    n_values = [2000]
    bits_values = [10, 14, 18, 23, 30, 40, 52]  # 10=~float16, 23=float32, 52=float64
    #bits_values = [10, 18, 30, 52]  # 10=~float16, 23=float32, 52=float64
    num_samples = 1000
    lam_grid = np.linspace(0.1, 5.0, 25)

    # results[n] = list of sigma^2 for each bits value
    results = {n: [] for n in n_values}

    for n in n_values:
        for bits in bits_values:
            print(f"n={n:4d}  bits={bits:2d} ... ", end="", flush=True)
            sig2 = run_one(n, m, bits, num_samples, lam_grid, seed=n)
            results[n].append(sig2)
            print(f"sigma^2 = {sig2:.4f}")

    # -------------------------------------------------------------- #
    #  Plot: one line per n, x = mantissa bits, y = sigma^2
    # -------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in n_values:
        ax.plot(bits_values, results[n], "o-", lw=2, ms=7, label=f"$n={n}$")

    ax.axhline(1.0, color="gray", ls=":", lw=1.5,
               label=r"$\sigma^2 = 1$ (theory)")
    # mark the standard float widths
    for bval, lbl in [(10, "f16"), (23, "f32"), (52, "f64")]:
        ax.axvline(bval, color="k", ls="--", lw=0.8, alpha=0.4)
        ax.text(bval + 0.5, ax.get_ylim()[0], lbl, fontsize=9,
                va="bottom", alpha=0.6)

    ax.set_xlabel("mantissa bits", fontsize=12)
    ax.set_ylabel(r"$\hat\sigma^2$  (empirical MGF bound)", fontsize=12)
    ax.set_title(
        rf"GSW sub-Gaussian parameter vs precision  ($m={m}$, "
        f"{num_samples} samples)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("precision_sweep.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to precision_sweep.png")
    plt.show()
