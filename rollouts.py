"""Parallel Monte-Carlo rollouts of the Gram-Schmidt Walk.

Each rollout is an independent walk on a fixed B. They parallelize across *processes*:
the low-precision walk is Python/CPU-bound, so threads are GIL-limited — only separate
processes give a real speedup (which saturates near the physical-core count).

Every worker is seeded from an independent numpy SeedSequence, so the rollouts are
statistically independent (and reproducible when a base `seed` is supplied). Without
this, forked workers would inherit one RNG state and produce identical walks.

Keep BLAS single-threaded (see the NUM_THREADS env vars set by the sweep scripts) so
N worker processes don't each spawn a thread pool and oversubscribe the cores.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import gsw
import lpla

# Per-worker globals, populated once by _init so B isn't re-sent with every task.
_B = _DIRS = _CHOP = _NOISE = None


def _init(B, directions, sig_bits, noise_mean, noise_std):
    global _B, _DIRS, _CHOP, _NOISE
    _B, _DIRS = B, directions
    _CHOP = lpla.make_round(sig_bits) if sig_bits is not None else None
    has_noise = noise_mean != 0.0 or noise_std != 0.0
    _NOISE = (lambda size: np.random.normal(noise_mean, noise_std, size)) if has_noise else None


def _rollout(seed_seq):
    # Independent global RNG state per task; SeedSequence keeps the streams decorrelated.
    np.random.seed(seed_seq.generate_state(8))
    r = gsw.gram_schmidt_walk(_B, chop=_CHOP, noise=_NOISE)
    return r.Bz.mean(), _DIRS @ r.Bz


def default_workers():
    # Half the logical cores ≈ physical cores, where the CPU-bound speedup plateaus.
    return max(1, (os.cpu_count() or 2) // 2)


def run_samples(B, directions, num_samples, *, sig_bits=None,
                noise_mean=0.0, noise_std=0.0, workers=None, seed=None):
    """Run `num_samples` independent walks on B; return (bz_means, projections).

    bz_means:    shape (num_samples,)                 mean of Bz per rollout
    projections: shape (len(directions), num_samples) directions @ Bz per rollout

    chop mode (sig_bits) and noise (noise_mean/noise_std) may be combined — mutual
    exclusion, if wanted, is the caller's policy. workers=1 runs inline (no pool).
    """
    workers = default_workers() if workers is None else workers
    seed_seqs = np.random.SeedSequence(seed).spawn(num_samples)

    if workers == 1:
        _init(B, directions, sig_bits, noise_mean, noise_std)
        results = [_rollout(ss) for ss in seed_seqs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init,
            initargs=(B, directions, sig_bits, noise_mean, noise_std),
        ) as ex:
            chunk = max(1, num_samples // (workers * 4))
            results = list(ex.map(_rollout, seed_seqs, chunksize=chunk))

    bz_means = np.array([m for m, _ in results])
    projections = np.stack([p for _, p in results], axis=1)
    return bz_means, projections
