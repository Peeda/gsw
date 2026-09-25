"""Norm distributions of Bz for the Higgs 2x500 matrix.

higgs_sweep.py's right panel plots ||Bz||_inf on raw scale, where precisions
sit on wildly different t^2 ranges. Here each config's samples are WHITENED
by their empirical 2x2 covariance (Mahalanobis whitening), so under joint
Gaussianity the whitened coords are iid N(0,1) at every precision and the
norm distributions have exact references:

    ||W Bz||_inf : survival 1 - (1 - 2 Phi_bar(t))^2   (max of 2 iid N(0,1))
    ||W Bz||_2   : survival exp(-t^2/2)                 (Rayleigh / chi_2)

On (t^2, log-survival) axes the chi_2 reference is a straight line of
slope -1/2. Deviations = departures from joint Gaussianity (e.g. bounded
support, discreteness), NOT sigma inflation, which whitening removes.

Usage: python higgs_norms.py
"""

import os
os.environ.setdefault("chop_backend", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from higgs_sweep import load_higgs_matrix
import rollouts


def whiten(bz: np.ndarray) -> np.ndarray:
    """Mahalanobis-whiten columns of bz (m, S): returns (m, S) with cov ~ I."""
    S = np.cov(bz)
    evals, evecs = np.linalg.eigh(S)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return W @ bz


def main():
    B = load_higgs_matrix(500, seed=0)          # same B as higgs_sweep.py
    m, n = B.shape
    sig_list = [2, 12, 52]
    num_samples = 1000

    data = {}
    for sb in sig_list:
        _, _, bz = rollouts.run_samples(
            B, np.eye(m), num_samples,
            sig_bits=None if sb == 52 else sb,
            noise_std=2 ** (-32), workers=6, seed=0,
        )
        data[sb] = whiten(bz)
        print(f"{sb}b done")

    colors = {2: "#440154", 12: "#3b528b", 52: "#fde725"}
    fig, (ax_sc, ax_inf, ax_2) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(
        f"Whitened norms of Bz — Higgs B ({m}x{n}), {num_samples} walks\n"
        "(under joint Gaussianity whitened coords are iid N(0,1))"
    )

    # scatter of whitened coords + unit circle: isotropy check
    for sb in sig_list:
        z = data[sb]
        ax_sc.plot(z[0], z[1], ".", color=colors[sb], alpha=0.08,
                   markersize=3, label=f"{sb}b")
    th = np.linspace(0, 2 * np.pi, 200)
    ax_sc.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("whitened Bz_0")
    ax_sc.set_ylabel("whitened Bz_1")
    ax_sc.set_title("whitened scatter (unit circle)")
    ax_sc.legend(title="mantissa bits", markerscale=8)

    # ||W Bz||_inf survival vs max-of-2-iid reference
    for sb in sig_list:
        t = np.sort(np.abs(data[sb]).max(axis=0))
        surv = np.arange(len(t), 0, -1) / len(t)
        ax_inf.plot(t ** 2, surv, color=colors[sb], label=f"{sb}b")
    tt = np.linspace(0, 4.5, 200)
    ref = 1.0 - (1.0 - 2.0 * stats.norm.sf(tt)) ** 2   # max of 2 iid |N(0,1)|
    ax_inf.plot(tt ** 2, ref, "k--", lw=1, label="max of 2 iid N(0,1)")
    ax_inf.set_yscale("log")
    ax_inf.set_xlabel("t²")
    ax_inf.set_ylabel("Pr[‖W·Bz‖∞ > t]")
    ax_inf.set_title("‖·‖∞ tail (dashed: iid-Gaussian max)")
    ax_inf.legend(title="mantissa bits")

    # ||W Bz||_2 survival vs Rayleigh reference (straight line on these axes)
    for sb in sig_list:
        t = np.sort(np.linalg.norm(data[sb], axis=0))
        surv = np.arange(len(t), 0, -1) / len(t)
        ax_2.plot(t ** 2, surv, color=colors[sb], label=f"{sb}b")
    ax_2.plot(tt ** 2, np.exp(-(tt ** 2) / 2), "k--", lw=1,
              label="χ₂ tail e^(-t²/2)")
    ax_2.set_yscale("log")
    ax_2.set_xlabel("t²")
    ax_2.set_ylabel("Pr[‖W·Bz‖₂ > t]")
    ax_2.set_title("‖·‖₂ tail (dashed: Rayleigh)")
    ax_2.legend(title="mantissa bits")

    fig.tight_layout()
    fig.savefig("higgs_norms.png", dpi=150)
    print("saved higgs_norms.png")


if __name__ == "__main__":
    main()
