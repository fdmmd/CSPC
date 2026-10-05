"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize


# ---------- Helper: gradient descent ----------
def gradient_descent(df, x0, lr=0.1, n_iter=1000, tol=1e-8):
    """Simple gradient descent: x <- x - lr * df(x)."""
    x = x0
    for _ in range(n_iter):
        x_new = x - lr * df(x)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x


# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result
print("=== 2A: f(x) = (x-3)^2 + 1, from x0 = 0 ===")
x0 = 0.0

# (1) gradient descent by hand
x_gd = gradient_descent(df, x0, lr=0.1)
print(f"  Gradient descent: x = {x_gd:.6f}")

# (2) Newton on df(x) = 0
x_newton = newton(df, x0, fprime=d2f)
print(f"  Newton:           x = {x_newton:.6f}")

# (3) SLSQP
res = minimize(f, x0, method="SLSQP")
print(f"  SLSQP:            x = {res.x[0]:.6f}")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2
def run_2b(x0):
    print(f"\n=== 2B: g(x) = x^4 - 3x^2 + x + 5, from x0 = {x0} ===")

    # (1) gradient descent (smaller lr — g has steeper slopes)
    x_gd = gradient_descent(dg, x0, lr=0.01, n_iter=5000)
    print(f"  Gradient descent: x = {x_gd:.6f}, g(x) = {g(x_gd):.4f}")

    # (2) Newton
    x_newton = newton(dg, x0, fprime=d2g)
    curv = d2g(x_newton)
    kind = "minimum" if curv > 0 else "maximum" if curv < 0 else "saddle"
    print(f"  Newton:           x = {x_newton:.6f}, g(x) = {g(x_newton):.4f}, "
          f"g''={curv:+.3f} -> {kind}")

    # (3) SLSQP
    res = minimize(g, x0, method="SLSQP")
    print(f"  SLSQP:            x = {res.x[0]:.6f}, g(x) = {g(res.x[0]):.4f}")


run_2b(0.0)
run_2b(2.0)


# ---------- Bonus info: all stationary points of g ----------
print("\n=== Stationary points of g: roots of dg(x) = 0 ===")
roots = np.roots([4, 0, -6, 1])
for r in sorted(roots):
    curv = d2g(r)
    kind = "minimum" if curv > 0 else "maximum" if curv < 0 else "saddle"
    print(f"  x = {r:+.4f}   g'' = {curv:+.3f}   -> {kind}")