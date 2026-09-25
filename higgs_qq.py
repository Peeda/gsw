"""Gaussianity diagnostics for the Higgs-sweep Bz samples.

The (t^2, log-survival) CCDF in higgs_sweep.py is hard to read because m = 2
(||Bz||_inf is a max over only 2 coords) and sigma varies ~20x across
precisions. This re-runs a few bit widths and shows, per sig_bits:

  (a) Q-Q of each Bz coordinate, standardised by its own sigma-hat, against
      N(0,1) quantiles -- a straight diagonal means Gaussian;
  (b) survival of |coord| on (t^2, log-y) axes with the N(0,1) reference
      slope -- straight line of slope -1/2 means exactly Gaussian;
  (c) standardised density (log-y) vs the N(0,1) pdf.

Usage: python higgs_qq.py
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


def main():
    B = load_higgs_matrix(500, seed=0)      # same B as higgs_sweep.py
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
        data[sb] = bz                       # (m, num_samples)
        print(f"{sb}b done: std per coord = {bz.std(axis=1)}")

    colors = {2: "#440154", 12: "#3b528b", 52: "#fde725"}
    fig, (ax_qq, ax_sv, ax_de) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Gaussianity check — Higgs B ({m}x{n}), {num_samples} walks")

    # (a) standardised Q-Q vs N(0,1)
    q = stats.norm.ppf((np.arange(1, num_samples + 1) - 0.5) / num_samples)
    for sb in sig_list:
        for i in range(m):
            s = np.sort(data[sb][i] / data[sb][i].std())
            ax_qq.plot(q, s, color=colors[sb], alpha=0.8,
                       label=f"{sb}b" if i == 0 else None)
    lim = max(abs(ax_qq.get_xlim()[0]), abs(ax_qq.get_ylim()[1]))
    ax_qq.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax_qq.set_xlabel("N(0,1) quantile")
    ax_qq.set_ylabel("standardised Bz coord quantile")
    ax_qq.set_title("Q-Q vs Gaussian")
    ax_qq.legend(title="mantissa bits")

    # (b) survival of |standardised coord| on (t^2, log-y): N(0,1) = slope -1/2
    for sb in sig_list:
        for i in range(m):
            t = np.sort(np.abs(data[sb][i] / data[sb][i].std()))
            surv = np.arange(len(t), 0, -1) / len(t)
            ax_sv.plot(t ** 2, surv, color=colors[sb], alpha=0.8,
                       label=f"{sb}b" if i == 0 else None)
    tt = np.linspace(0, ax_sv.get_xlim()[1], 10)
    ax_sv.plot(tt, 2 * np.exp(-tt / 2), "k--", lw=0.8, label="N(0,1) tail")
    ax_sv.set_yscale("log")
    ax_sv.set_xlabel("t²")
    ax_sv.set_ylabel("Pr[|Bz_i|/σ > t]")
    ax_sv.set_title("standardised tail (dashed: Gaussian)")
    ax_sv.legend(title="mantissa bits")

    # (c) standardised density vs N(0,1) pdf
    for sb in sig_list:
        z = (data[sb] / data[sb].std(axis=1, keepdims=True)).ravel()
        ax_de.hist(z, bins=120, density=True, histtype="step",
                   color=colors[sb], label=f"{sb}b")
    xs = np.linspace(-6, 6, 400)
    ax_de.plot(xs, stats.norm.pdf(xs), "k--", lw=1, label="N(0,1)")
    ax_de.set_yscale("log")
    ax_de.set_xlabel("Bz_i / σ")
    ax_de.set_title("standardised density (log-y)")
    ax_de.legend(title="mantissa bits")

    fig.tight_layout()
    fig.savefig("higgs_qq.png", dpi=150)
    print("saved higgs_qq.png")


if __name__ == "__main__":
    main()
