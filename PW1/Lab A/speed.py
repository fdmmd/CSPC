"""Compare the speed of the pure-Python and NumPy decay simulations."""
import time

from decay import simulate, simulate_loop


def timeit(func, *args, **kwargs):
    """Return the execution time of func in seconds."""
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def main():
    N0 = 200_000
    lam = 0.4

    t_loop = timeit(simulate_loop, N0, lam)
    t_numpy = timeit(simulate, N0, lam)

    speedup = t_loop / t_numpy

    print(f"Pure-Python loop : {t_loop:.4f} s")
    print(f"NumPy vectorised : {t_numpy:.4f} s")
    print(f"Speed-up         : {speedup:.1f}x faster with NumPy")


if __name__ == "__main__":
    main()