"""
Empirical verification that the Gram-Schmidt Walk produces 1-sub-Gaussian
discrepancy, i.e. for each coordinate j of B z_t:

    E[exp(lambda (B z_t)_j)] <= exp(lambda^2 / 2)   for all lambda

Equivalently (all with sigma^2 = 1):
  - Tails:    P(|(B z_t)_j| > t) <= 2 exp(-t^2 / 2)
  - Moments:  (E[|(B z_t)_j|^p])^{1/p} <= C sqrt(p)   for all p >= 1
  - MGF:      sup_lambda  (2/lambda^2) log E[exp(lambda X)] <= 1
"""

import numpy as np
from gram_schmidt_walk import gram_schmidt_walk
from joblib import Parallel, delayed

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def collect_discrepancies(B, num_samples, seed=42):
    """Run GSW num_samples times, return (num_samples, m) discrepancy array."""
    #m, n = B.shape
    #rng = np.random.default_rng(seed)
    #samples = np.zeros((num_samples, m))
    #for i in range(num_samples):
    #    z_t = gram_schmidt_walk(B, seed=rng.integers(2**62))
    #    samples[i] = B @ z_t
    #return samples
    m, n = B.shape
    rng = np.random.default_rng(seed)
    seeds = rng.integers(2**62, size=num_samples)

    results = Parallel(n_jobs=-1)(
        delayed(lambda s: B @ gram_schmidt_walk(B, seed=s))(s)
        for s in seeds
    )
    return np.array(results)


def stable_log_mgf(samples, lam):
    """Compute log E[exp(lambda * X)] with log-sum-exp stability."""
    lx = lam * samples
    c = np.max(lx)
    return c + np.log(np.mean(np.exp(lx - c)))


def empirical_tail_probs(samples, thresholds):
    """P(|X| > t) for each threshold t."""
    return np.array([np.mean(np.abs(samples) > t) for t in thresholds])


def moment_ratios(samples, p_values):
    """(E[|X|^p])^{1/p} / sqrt(p) for each p."""
    out = np.empty(len(p_values))
    for i, p in enumerate(p_values):
        out[i] = np.mean(np.abs(samples) ** p) ** (1.0 / p) / np.sqrt(p)
    return out


def mgf_bound_values(samples, lam_values):
    """(2 / lambda^2) log E[exp(lambda X)] for mean-zero samples."""
    s = samples - np.mean(samples)
    out = np.empty(len(lam_values))
    for i, lam in enumerate(lam_values):
        out[i] = 2.0 * stable_log_mgf(s, lam) / (lam ** 2)
    return out


# ------------------------------------------------------------------ #
#  Main test
# ------------------------------------------------------------------ #

def test_1_subgaussian(B, num_samples=10000, seed=42):
    """
    Test whether each coordinate of B z_t is 1-sub-Gaussian,
    i.e. sub-Gaussian with parameter sigma^2 = 1.

    Parameters
    ----------
    B : ndarray (m, n)   columns of norm <= 1
    num_samples : int    number of independent GSW signings to draw
    seed : int           random seed
    """
    SIGMA_SQ = 1.0        # the target: 1-sub-Gaussian
    SIGMA = 1.0

    m, n = B.shape
    rng = np.random.default_rng(seed)

    # --- collect samples ---
    print(f"Sampling {num_samples} GSW signings  (n={n}, m={m}) ...")
    gsw_disc = collect_discrepancies(B, num_samples, seed=seed)

    sigmas_emp = np.std(gsw_disc, axis=0)
    vars_emp = np.var(gsw_disc, axis=0)

    print(f"\nEmpirical variance per coordinate:")
    print(f"  mean Var[(Bz)_j] = {vars_emp.mean():.4f}   "
          f"max = {vars_emp.max():.4f}")
    print(f"  (1-sub-Gaussian implies Var <= sigma^2 = {SIGMA_SQ:.1f})")
    var_ok = vars_emp.max() <= SIGMA_SQ + 0.05
    print(f"  max variance {'<=' if var_ok else '>'} 1:  "
          f"{'PASS' if var_ok else 'FAIL'}")

    # ============================================================== #
    #  Test 1 — Tail bound  P(|X| > t) <= 2 exp(-t^2 / 2)
    # ============================================================== #
    print(f"\n{'='*64}")
    print(f"TEST 1: Tail bound   P(|(Bz)_j| > t) <= 2 exp(-t^2 / 2)")
    print(f"        (sigma^2 = 1 throughout)")
    print(f"{'='*64}")

    t_values = np.arange(0.5, 5.1, 0.5)
    gauss_bound = 2.0 * np.exp(-t_values ** 2 / 2.0)

    # check every coordinate against the FIXED sigma=1 bound
    tail_violations = 0
    worst_excess = 0.0
    worst_coord = 0
    for j in range(m):
        emp = empirical_tail_probs(gsw_disc[:, j], t_values)
        # finite-sample slack: 3 standard errors of the binomial estimator
        slack = 3.0 * np.sqrt(np.maximum(gauss_bound * (1 - gauss_bound), 0.0)
                               / num_samples)
        excess = np.max(emp - gauss_bound)
        if np.any(emp > gauss_bound + slack):
            tail_violations += 1
        if excess > worst_excess:
            worst_excess = excess
            worst_coord = j

    print(f"  Coordinates violating (with finite-sample slack): "
          f"{tail_violations}/{m}")

    # detailed table for worst coordinate
    emp_worst = empirical_tail_probs(gsw_disc[:, worst_coord], t_values)
    print(f"\n  Worst coordinate {worst_coord}  "
          f"(emp sigma={sigmas_emp[worst_coord]:.4f}):")
    print(f"  {'t':>6}  {'P(|X|>t) emp':>14}  {'2exp(-t^2/2)':>14}  {'pass':>6}")
    for k in range(len(t_values)):
        ok = "yes" if emp_worst[k] <= gauss_bound[k] else "NO"
        print(f"  {t_values[k]:6.1f}  {emp_worst[k]:14.6f}  "
              f"{gauss_bound[k]:14.6f}  {ok:>6}")

    # ============================================================== #
    #  Test 2 — Moment growth   (E|X|^p)^{1/p} / sqrt(p) <= C
    # ============================================================== #
    print(f"\n{'='*64}")
    print(f"TEST 2: Moment ratio  (E|X|^p)^(1/p) / sqrt(p)")
    print(f"        For 1-sub-Gaussian this should be bounded by ~1")
    print(f"{'='*64}")

    ps = np.array([1, 2, 3, 4, 6, 8, 10, 14, 20])

    # max ratio over all coordinates at each p
    gsw_max_ratio = np.zeros(len(ps))
    for j in range(m):
        r = moment_ratios(gsw_disc[:, j], ps)
        gsw_max_ratio = np.maximum(gsw_max_ratio, r)

    # reference: N(0,1) samples (sigma=1 Gaussian)
    gauss_ref = rng.normal(0, SIGMA, size=num_samples)
    gauss_ref_ratio = moment_ratios(gauss_ref, ps)

    print(f"  {'p':>4}  {'GSW (max/coords)':>16}  {'N(0,1) ref':>12}  "
          f"{'<= 1?':>6}")
    for k in range(len(ps)):
        ok = "yes" if gsw_max_ratio[k] <= 1.0 else "no"
        print(f"  {ps[k]:4d}  {gsw_max_ratio[k]:16.4f}  "
              f"{gauss_ref_ratio[k]:12.4f}  {ok:>6}")

    moment_pass = gsw_max_ratio.max() <= 1.0 + 0.05
    print(f"\n  sup over p:  {gsw_max_ratio.max():.4f}   "
          f"{'PASS (<= 1)' if moment_pass else 'EXCEEDS 1'}")

    # ============================================================== #
    #  Test 3 — MGF bound   (2/lambda^2) log E[exp(lambda X)] <= 1
    # ============================================================== #
    print(f"\n{'='*64}")
    print(f"TEST 3: MGF bound  sup_lam (2/lam^2) log E[exp(lam X)] <= 1")
    print(f"{'='*64}")

    lams = np.linspace(0.1, 5.0, 25)

    mgf_worst_coord = 0
    mgf_worst_sup = 0.0
    all_mgf = np.zeros((m, len(lams)))

    for j in range(m):
        vals = mgf_bound_values(gsw_disc[:, j], lams)
        all_mgf[j] = vals
        s = np.max(vals)
        if s > mgf_worst_sup:
            mgf_worst_sup = s
            mgf_worst_coord = j

    mgf_pass = mgf_worst_sup <= SIGMA_SQ + 0.05

    print(f"  Per-coordinate sup of (2/lam^2) log E[exp(lam X)]:")
    for j in range(m):
        s = np.max(all_mgf[j])
        ok = "yes" if s <= SIGMA_SQ else "no"
        print(f"    coord {j}: {s:.4f}  (<= 1? {ok})")

    print(f"\n  Overall sup:  {mgf_worst_sup:.4f}   "
          f"{'PASS (<= 1)' if mgf_pass else 'EXCEEDS 1'}")

    # detailed table for worst coordinate
    mgf_vals = all_mgf[mgf_worst_coord]
    print(f"\n  Detail for coordinate {mgf_worst_coord}:")
    print(f"  {'lambda':>8}  {'(2/lam^2)logMGF':>16}  {'<= 1?':>6}")
    for k in range(len(lams)):
        ok = "yes" if mgf_vals[k] <= SIGMA_SQ else "no"
        print(f"  {lams[k]:8.3f}  {mgf_vals[k]:16.4f}  {ok:>6}")

    # ============================================================== #
    #  Summary
    # ============================================================== #
    print(f"\n{'='*64}")
    print(f"SUMMARY  (target: 1-sub-Gaussian, sigma^2 = 1)")
    print(f"{'='*64}")
    print(f"  max Var[(Bz)_j]:           {vars_emp.max():.4f}  "
          f"(<= 1? {'PASS' if var_ok else 'FAIL'})")
    print(f"  Tail violations:           {tail_violations}/{m}  "
          f"({'PASS' if tail_violations == 0 else 'FAIL'})")
    print(f"  Moment ratio sup:          {gsw_max_ratio.max():.4f}  "
          f"(<= 1? {'PASS' if moment_pass else 'FAIL'})")
    print(f"  MGF psi_2 sup:             {mgf_worst_sup:.4f}  "
          f"(<= 1? {'PASS' if mgf_pass else 'FAIL'})")

    all_pass = var_ok and tail_violations == 0 and moment_pass and mgf_pass
    print(f"\n  {'CONSISTENT with 1-sub-Gaussian' if all_pass else 'NOT 1-sub-Gaussian at this sample size'}")

    # ============================================================== #
    #  Plots (if matplotlib is available)
    # ============================================================== #
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(r'Testing 1-sub-Gaussianity of $(Bz)_j$   '
                     rf'($n={n},\; m={m},\; {num_samples}$ samples)',
                     fontsize=13)

        # --- Plot 1: tail probabilities vs 2exp(-t^2/2) ---
        ax = axes[0]
        ts_fine = np.linspace(0.1, 5.0, 60)
        for j in range(m):
            emp_fine = empirical_tail_probs(gsw_disc[:, j], ts_fine)
            alpha = 1.0 if j == worst_coord else 0.25
            ax.semilogy(ts_fine, emp_fine, 'b-', lw=1.2, alpha=alpha)
        ax.semilogy(ts_fine, 2.0 * np.exp(-ts_fine**2 / 2.0),
                     'k--', lw=2, label=r'$2e^{-t^2/2}$  ($\sigma^2\!=\!1$)')
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$P(|X| > t)$')
        ax.set_title('Tail probabilities')
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0.5 / num_samples)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: moment ratios vs 1 ---
        ax = axes[1]
        ax.plot(ps, gsw_max_ratio, 'bo-', lw=2, ms=6,
                label='GSW (max over coords)')
        ax.plot(ps, gauss_ref_ratio, 'k--', lw=1.5, label='$N(0,1)$ ref')
        ax.axhline(1.0, color='r', ls=':', lw=1.5,
                    label=r'$\sigma=1$ bound')
        ax.set_xlabel('$p$')
        ax.set_ylabel(r'$(E|X|^p)^{1/p}\,/\,\sqrt{p}$')
        ax.set_title('Moment ratio')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # --- Plot 3: MGF bound vs 1 ---
        ax = axes[2]
        for j in range(m):
            alpha = 1.0 if j == mgf_worst_coord else 0.25
            ax.plot(lams, all_mgf[j], 'b-', lw=1.2, alpha=alpha)
        ax.axhline(SIGMA_SQ, color='r', ls=':', lw=2,
                    label=r'$\sigma^2=1$')
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$(2/\lambda^2)\,\log E[e^{\lambda X}]$')
        ax.set_title('MGF bound')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('subgaussian_test.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to subgaussian_test.png")
        plt.show()
    else:
        print("\n(Install matplotlib for diagnostic plots)")


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n, m = 3000, 8
    B = rng.standard_normal((m, n))
    B /= np.linalg.norm(B, axis=0, keepdims=True)

    test_1_subgaussian(B, num_samples=1000, seed=42)
