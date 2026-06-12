import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class WalkResult:
    assignment: np.ndarray
    # min over iterations of E[delta_t^2] = d_plus * d_minus
    min_second_moment: float
    # weighted sum of input columns by final assignment: B @ assignment
    Bz: np.ndarray
    # Each entry is (z_before_step, u, delta_t, second_moment). Only populated
    # when gram_schmidt_walk is called with record_trajectory=True.
    trajectory: list[tuple[np.ndarray, np.ndarray, float, float]] = field(default_factory=list)


def gram_schmidt_walk(
    B: np.ndarray,
    chop: Callable | None = None,
    noise: Callable[[int], np.ndarray] | None = None,
    record_trajectory: bool = False,
) -> WalkResult:
    """
    Gram-Schmidt Walk (Algorithm 1).

    B:     (m, n) matrix with unit-norm columns
    chop:  optional callable that rounds an array to a reduced-precision format;
           applied to the inputs and output of each lstsq solve
    noise: optional callable(n) -> ndarray; its output is added to z after each
           step, and z is clamped back to [-1, 1]^n before the next iteration.
           Example: lambda n: np.random.normal(0, 0.01, n)
    Returns: WalkResult with assignment vector in {-1, +1}^n and run statistics
    """
    _, n = B.shape
    z = np.zeros(n)
    p = np.random.randint(n)
    min_second_moment = float('inf')
    trajectory = []

    while True:
        active = np.where(np.abs(z) < 1 - 1e-9)[0]
        if len(active) == 0:
            break

        if p not in active:
            p = active[np.random.randint(len(active))]

        # Step direction: argmin_u ||Bu||^2  s.t.  u[p]=1, u[i]=0 for i not in active
        # Substituting u[p]=1, minimise ||B_free @ v + B_p||^2 over free variables v.
        free = active[active != p]
        u = np.zeros(n)
        u[p] = 1.0
        if len(free) > 0:
            B_free = B[:, free]
            b_p    = B[:, p]
            if chop is not None:
                with np.errstate(over='ignore'):
                    B_free = chop(B_free.copy())
                    b_p    = chop(b_p.copy())
            v, _, _, _ = np.linalg.lstsq(B_free, -b_p, rcond=None)
            if chop is not None:
                with np.errstate(over='ignore'):
                    v = chop(v)
                # Overflow at low precision produces inf/nan; zero those entries so
                # the step falls back to the pivot direction for affected coordinates.
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            u[free] = v

        # Feasible step interval Delta = {delta : z + delta*u in [-1,1]^n}
        nz = np.abs(u) > 1e-15
        r1 = (-1 - z[nz]) / u[nz]
        r2 = ( 1 - z[nz]) / u[nz]
        delta_min = np.max(np.minimum(r1, r2))
        delta_max = np.min(np.maximum(r1, r2))

        d_plus  = abs(delta_max)   # |max Delta|
        d_minus = abs(delta_min)   # |min Delta|
        total = d_plus + d_minus
        if total < 1e-15:
            break

        # E[delta_t^2] = d_plus * d_minus  (second moment of the two-point distribution)
        second_moment = d_plus * d_minus
        min_second_moment = min(min_second_moment, second_moment)

        # Martingale-preserving random step: E[delta_t] = 0
        delta_t = d_plus if np.random.random() < d_minus / total else -d_minus
        if record_trajectory:
            trajectory.append((z.copy(), u.copy(), delta_t, second_moment))

        z += delta_t * u
        if noise is not None:
            unfrozen = np.abs(z) < 1 - 1e-9
            z[unfrozen] += noise(unfrozen.sum())
            z = np.clip(z, -1.0, 1.0)
        z = np.where(np.abs(z) > 1 - 1e-9, np.sign(z), z)

    assignment = np.sign(z).astype(int)
    return WalkResult(
        assignment=assignment,
        min_second_moment=min_second_moment,
        Bz=B @ assignment,
        trajectory=trajectory if record_trajectory else [],
    )
