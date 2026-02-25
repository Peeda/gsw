"""
Gram-Schmidt Walk for Vector Balancing

Given n vectors b_1, ..., b_n in R^m with ||b_i|| <= 1, round a fractional
coloring z_0 in [-1,1]^n to an integral signing z in {-1,+1}^n such that
the signed sum (discrepancy) ||sum_i z_i b_i|| is small.

The algorithm performs a random walk in [-1,1]^n. At each step it picks a
random pivot, computes a walk direction by solving an m-dimensional linear
system derived from the Gram matrix I + B_K B_K^T, takes a random step
that preserves the martingale property E[z_{t+1}] = z_t, and freezes any
coordinate that reaches +-1.  After at most n steps every coordinate is
frozen and the output lies in {-1,+1}^n.

The walk direction at each step minimizes

    ||u||^2 + ||B u||^2 = u^T (I + B^T B) u

subject to u_pivot = 1, u_i = 0 for frozen i.  This corresponds to the
"augmented" vectors (e_i, b_i) from [1]: the walk moves in the direction
that grows slowest in the norm ||sum u_i (e_i, b_i)||.

Key guarantee (Theorem 1.2 of [1]):  for any vectors b_1,...,b_n of
Euclidean norm at most 1, the signing z produced by this algorithm
satisfies a sub-Gaussian tail bound on each coordinate of the discrepancy
vector B z.

References
----------
[1] Bansal, N., Dadush, D., Garg, S., & Lovett, S. (2019).
    "The Gram-Schmidt Walk: A Cure for the Banaszczyk Blues."
    Theory of Computing, 15(21), 1-36.

[2] Harshaw, C., Sáenz, F., Spielman, D., & Zhang, P. (2024).
    "Balancing Covariates in Randomized Experiments Using the
    Gram-Schmidt Walk Design."  JASA 119(546). arXiv:1911.03071.
"""

import numpy as np


def _quantize(x, scale, inv_scale):
    """Round x to a fixed number of mantissa bits: round(x * 2^b) / 2^b."""
    return np.round(x * scale) * inv_scale


def gram_schmidt_walk(B, z0=None, seed=None, dtype=None, mantissa_bits=None):
    """
    Gram-Schmidt Walk algorithm for vector balancing / discrepancy.

    Parameters
    ----------
    B : ndarray, shape (m, n)
        Matrix whose columns b_1, ..., b_n are vectors in R^m.
        Each column must satisfy ||b_i|| <= 1.
    z0 : ndarray, shape (n,), optional
        Starting fractional coloring in [-1, 1]^n.
        Default is the zero vector (each coordinate equally likely to
        end up as +1 or -1).
    seed : int or None, optional
        Random seed for reproducibility.
    dtype : numpy dtype, optional
        Floating-point precision to use for all arithmetic
        (e.g. np.float16, np.float32, np.float64).
        Default is np.float64.
    mantissa_bits : int or None, optional
        If set, emulate a floating-point type with this many mantissa
        bits via fixed-point quantization (round(x*2^b)/2^b) after
        every key operation.  Works in float64 internally.
        Useful values: 10 (float16-like) to 52 (float64 = no-op).
        Overrides dtype when set.

    Returns
    -------
    z_t : ndarray, shape (n,)
        Signing vector in {-1, +1}^n.

    Notes
    -----
    The signed vector sum (discrepancy) is B z_t = sum_i z_i b_i.

    The output is a martingale rounding of z0: E[z_i] = z0_i for every i.
    Starting from z0 = 0, each coordinate is +1 or -1 with equal marginal
    probability, but the coordinates are negatively correlated in a way
    that keeps B z_t small.

    The walk direction at step t minimises u^T M_S u with M_S = I + B_S^T B_S
    (the alive-set Gram matrix, where B_S = B[:, S]), subject to u_pivot = 1.
    Via the Woodbury identity this reduces to an m x m solve, making the
    per-step cost O(m^3 + n m) and the total cost O(n (m^3 + n m)).

    Raises
    ------
    ValueError
        If any input vector has norm exceeding 1 + 1e-6, or if z0 is not
        in [-1, 1]^n.
    """
    rng = np.random.default_rng(seed)

    # --- precision setup ---
    if mantissa_bits is not None:
        dtype = np.float64
        Q_scale = 2.0 ** mantissa_bits
        Q_inv   = 1.0 / Q_scale
        Q = lambda x: _quantize(x, Q_scale, Q_inv)
        eps = Q_inv                        # effective machine epsilon
    else:
        if dtype is None:
            dtype = np.float64
        dtype = np.dtype(dtype)
        Q = lambda x: x                   # no-op
        eps = np.finfo(dtype).eps

    B = Q(np.asarray(B, dtype=np.float64)).astype(dtype)
    m, n = B.shape

    # --- validate norms (always in float64 for safety) ---
    norms = np.linalg.norm(B.astype(np.float64), axis=0)
    norm_tol = max(1e-6, 1000 * float(eps))
    if np.any(norms > 1.0 + norm_tol):
        raise ValueError(
            f"All columns of B must have norm <= 1; "
            f"max norm is {norms.max():.6f}"
        )

    # --- initial fractional coloring ---
    if z0 is not None:
        z_t = Q(np.array(z0, dtype=np.float64))
        if z_t.shape != (n,):
            raise ValueError(f"z0 shape {z_t.shape} != ({n},)")
        if np.any(np.abs(z_t) > 1.0 + 1e-10):
            raise ValueError("z0 entries must lie in [-1, 1]")
    else:
        z_t = np.zeros(n, dtype=np.float64)

    tol = 100 * eps
    zero_tol = 10 * eps

    # --- alive (unfrozen) indicator ---
    alive = np.ones(n, dtype=bool)
    for i in range(n):
        if abs(abs(z_t[i]) - 1.0) < tol:
            z_t[i] = np.sign(z_t[i])
            alive[i] = False

    # --- maintain G = sum_{i alive} b_i b_i^T  (m x m) ---
    # updated via rank-1 downdates when a coordinate freezes
    alive_idx = np.flatnonzero(alive)
    G = Q(B[:, alive_idx] @ B[:, alive_idx].T)

    # ------------------------------------------------------------------ #
    #  Main loop — each iteration freezes at least one coordinate
    # ------------------------------------------------------------------ #
    for _iteration in range(n):
        alive_idx = np.flatnonzero(alive)
        k = len(alive_idx)
        if k == 0:
            break

        # 1. Pick a random pivot from the alive set
        p = alive_idx[rng.integers(k)]
        bp = B[:, p]

        # 2. Compute walk direction via the m-space (Woodbury) formulation
        #
        #    Let K = alive \ {pivot}.  Define
        #        C = I_m  +  sum_{j in K} b_j b_j^T  =  I_m + G - b_p b_p^T
        #
        #    The optimal direction (minimising u^T M_S u, M_S = I_k + B_S^T B_S,
        #    subject to u_p = 1) is:
        #        u_p = 1
        #        u_i = -b_i^T C^{-1} b_p     for i in K
        #
        #    Derivation uses the Woodbury identity to move from the k x k
        #    alive-set system to an m x m system.  See Section 3 of [1].
        C = Q(np.eye(m) + G - np.outer(bp, bp))

        # np.linalg.solve requires at least float32; upcast, solve, cast back
        solve_dtype = np.float32 if dtype == np.float16 else np.float64
        C_s = C.astype(solve_dtype)
        bp_s = bp.astype(solve_dtype)
        try:
            w = np.linalg.solve(C_s, bp_s)
        except np.linalg.LinAlgError:
            C_s += np.finfo(solve_dtype).eps * 100 * np.eye(m, dtype=solve_dtype)
            w = np.linalg.solve(C_s, bp_s)
        w = Q(w.astype(np.float64))

        # build local direction vector (length k, indexed like alive_idx)
        u_alive = np.zeros(k)
        pivot_local = np.searchsorted(alive_idx, p)
        u_alive[pivot_local] = 1.0

        non_pivot_mask = np.ones(k, dtype=bool)
        non_pivot_mask[pivot_local] = False
        np_local = np.where(non_pivot_mask)[0]

        if len(np_local) > 0:
            u_alive[np_local] = Q(-(B[:, alive_idx[np_local]].T @ w))

        # 3. Compute step sizes
        #
        #    delta_plus  = max delta >= 0 s.t. z_t + delta u in [-1,1]^n
        #    delta_minus = max delta >= 0 s.t. z_t - delta u in [-1,1]^n
        #
        #    Closed form per coordinate (u_i != 0):
        #      dp_i = (sign(u_i) - z_t_i) / u_i
        #      dm_i = (sign(u_i) + z_t_i) / u_i
        #    then delta_plus = min dp_i, delta_minus = min dm_i.
        z_alive = z_t[alive_idx]
        nz = np.abs(u_alive) > zero_tol

        if not np.any(nz):
            # degenerate direction — freeze pivot by coin flip
            prob_p = (1.0 + z_t[p]) / 2.0
            z_t[p] = 1.0 if rng.random() < prob_p else -1.0
            alive[p] = False
            G = Q(G - np.outer(bp, bp))
            continue

        u_nz = u_alive[nz]
        z_nz = z_alive[nz]

        dp = (np.sign(u_nz) - z_nz) / u_nz
        dm = (np.sign(u_nz) + z_nz) / u_nz
        dp = np.maximum(dp, 0.0)           # floating-point guard
        dm = np.maximum(dm, 0.0)

        delta_plus = np.min(dp)
        delta_minus = np.min(dm)

        if delta_plus + delta_minus < tol:
            break

        # 4. Random step — maintains the martingale E[z_{t+1}] = z_t
        #
        #    P(step = +delta_plus)  = delta_minus / (delta_plus + delta_minus)
        #    P(step = -delta_minus) = delta_plus  / (delta_plus + delta_minus)
        prob_forward = delta_minus / (delta_plus + delta_minus)
        step = delta_plus if rng.random() < prob_forward else -delta_minus

        z_t[alive_idx] = Q(z_t[alive_idx] + step * u_alive)

        # 5. Freeze coordinates that reached +-1
        for i in alive_idx:
            if abs(abs(z_t[i]) - 1.0) < tol:
                z_t[i] = 1.0 if z_t[i] > 0 else -1.0
                alive[i] = False
                G = Q(G - np.outer(B[:, i], B[:, i]))

    # --- snap any stragglers (numerical edge case) ---
    remaining = np.flatnonzero(alive)
    if len(remaining) > 0:
        probs = (1.0 + z_t[remaining]) / 2.0
        z_t[remaining] = np.where(
            rng.random(len(remaining)) < probs, 1.0, -1.0
        )

    return z_t


def discrepancy(B, z_t):
    """
    Compute the signed vector sum (discrepancy) B z_t = sum_i z_i b_i.

    Parameters
    ----------
    B : ndarray (m, n)
    z_t : ndarray (n,)   in {-1, +1}^n

    Returns
    -------
    disc : ndarray (m,)     the discrepancy vector
    l2   : float            its Euclidean norm
    linf : float            its infinity norm
    """
    disc = B @ z_t
    return disc, float(np.linalg.norm(disc)), float(np.max(np.abs(disc)))


# ------------------------------------------------------------------ #
#  Demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(42)

    # --- generate n random unit-norm vectors in R^m ---
    n, m = 500, 10
    B = rng.standard_normal((m, n))
    B /= np.linalg.norm(B, axis=0, keepdims=True)    # project to unit sphere

    print(f"Gram-Schmidt Walk — vector balancing demo")
    print(f"  n = {n} vectors in R^{m},  all of unit norm\n")

    # --- run GSW and compare with random signing ---
    num_trials = 200

    gsw_l2, gsw_linf = [], []
    rand_l2, rand_linf = [], []

    for t in range(num_trials):
        # Gram-Schmidt Walk
        z_gsw = gram_schmidt_walk(B, seed=t)
        _, l2, linf = discrepancy(B, z_gsw)
        gsw_l2.append(l2)
        gsw_linf.append(linf)

        # independent random signing (baseline)
        z_rand = rng.choice([-1.0, 1.0], size=n)
        _, l2r, linfr = discrepancy(B, z_rand)
        rand_l2.append(l2r)
        rand_linf.append(linfr)

    print(f"{'':>18} {'mean L2':>10} {'std L2':>10} "
          f"{'mean Linf':>10} {'std Linf':>10}")
    print("-" * 62)
    print(f"{'GSW':>18} {np.mean(gsw_l2):10.3f} {np.std(gsw_l2):10.3f} "
          f"{np.mean(gsw_linf):10.3f} {np.std(gsw_linf):10.3f}")
    print(f"{'random signing':>18} {np.mean(rand_l2):10.3f} "
          f"{np.std(rand_l2):10.3f} {np.mean(rand_linf):10.3f} "
          f"{np.std(rand_linf):10.3f}")

    # --- martingale check ---
    print(f"\nMartingale check (1000 signings, z0 = 0):")
    zs = np.array([gram_schmidt_walk(B, seed=s) for s in range(1000)])
    mu = zs.mean(axis=0)
    print(f"  mean of E[z_i] = {mu.mean():.5f}   (should be ~0)")
    print(f"  max |E[z_i]|   = {np.max(np.abs(mu)):.5f}")

    # --- starting from a non-zero fractional coloring ---
    print(f"\nRounding a fractional coloring z0:")
    z0 = rng.uniform(-0.8, 0.8, size=n)
    zs = np.array([gram_schmidt_walk(B, z0=z0, seed=s) for s in range(1000)])
    mu = zs.mean(axis=0)
    err = np.max(np.abs(mu - z0))
    print(f"  max |E[z_i] - z0_i| = {err:.5f}   (should be small)")
    print(f"\nDone.")
