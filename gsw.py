import numpy as np
from dataclasses import dataclass, field
from typing import Callable

import lpla


@dataclass
class WalkResult:
    assignment: np.ndarray
    # weighted sum of input columns by final assignment: B @ assignment
    Bz: np.ndarray
    # Each entry is (z_before_step, u, delta_t). Only populated when
    # gram_schmidt_walk is called with record_trajectory=True.
    trajectory: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)


def gram_schmidt_walk(
    B: np.ndarray,
    chop: Callable | None = None,
    noise: Callable[[int], np.ndarray] | None = None,
    record_trajectory: bool = False,
) -> WalkResult:
    """
    Gram-Schmidt Walk (Algorithm 1).

    B:     (m, n) matrix with unit-norm columns
    chop:  optional callable that rounds an array to a reduced-precision format.
           When given, the least-squares direction is computed by lpla.lstsq, a
           hand-rolled Householder QR/LQ solver in which every arithmetic operation
           is rounded to the target format (not just the solve's inputs/output).
    noise: optional callable(n) -> ndarray; its output is added to z after each
           step, and z is clamped back to [-1, 1]^n before the next iteration.
           Example: lambda n: np.random.normal(0, 0.01, n)
    Returns: WalkResult with assignment vector in {-1, +1}^n and run statistics
    """
    _, n = B.shape
    z = np.zeros(n)
    p = np.random.randint(n)
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
            if chop is not None:
                # Every op in the solve runs at the target precision. Overflow at low
                # precision yields inf/nan, which lpla.lstsq zeros out so the step
                # falls back to the pivot direction for affected coordinates.
                v = lpla.lstsq(B[:, free], -B[:, p], chop)
            else:
                v, _, _, _ = np.linalg.lstsq(B[:, free], -B[:, p], rcond=None)
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

        # Martingale-preserving random step: E[delta_t] = 0
        delta_t = d_plus if np.random.random() < d_minus / total else -d_minus
        if record_trajectory:
            trajectory.append((z.copy(), u.copy(), delta_t))

        z += delta_t * u
        if noise is not None:
            unfrozen = np.abs(z) < 1 - 1e-9
            z[unfrozen] += noise(unfrozen.sum())
            z = np.clip(z, -1.0, 1.0)
        z = np.where(np.abs(z) > 1 - 1e-9, np.sign(z), z)

    assignment = np.sign(z).astype(int)
    return WalkResult(
        assignment=assignment,
        Bz=B @ assignment,
        trajectory=trajectory if record_trajectory else [],
    )
