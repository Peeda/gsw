import numpy as np
import gsw


def main() -> None:
    m, n = 10, 25
    rng = np.random.default_rng()
    B = rng.standard_normal((m, n))
    B /= np.linalg.norm(B, axis=0, keepdims=True)

    result = gsw.gram_schmidt_walk(
        B,
        noise=lambda n: np.random.normal(0, np.sqrt(0.01), n),
        record_trajectory=True,
    )
    T = len(result.trajectory)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"B shape: ({m}, {n})  |  iterations: {T}")
    print(f"min E[delta^2]:  {result.min_second_moment:.6f}")
    print(f"Bz:              {result.Bz}")
    print(f"assignment:      {result.assignment}")
    print()
    print("Press Enter to step through iterations (q + Enter to quit).")
    print()

    for t, (z, u, delta_t, second_moment) in enumerate(result.trajectory):
        raw = input(f"[{t + 1}/{T}] ")
        if raw.strip().lower() == "q":
            break
        print(f"  z (before):    {z}")
        print(f"  u:             {u}")
        print(f"  delta_t:       {delta_t:+.6f}")
        print(f"  E[delta_t^2]:  {second_moment:.6f}")
        print(f"  z (after):     {z + delta_t * u}")
        print()

    print(f"Final assignment: {result.assignment}")
    print(f"Bz:               {result.Bz}")
    print(f"min second moment:{result.min_second_moment}")


if __name__ == "__main__":
    main()
