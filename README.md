# CSPC — Lab Reports

## PW1 — Lab A

### Goal
Simulate radioactive decay using a pure-Python loop and a vectorised NumPy
version, verify correctness with pytest, and compare their speed.

### Environment
- Conda env: `cspc`
- Python 3.11, numpy, pytest

### Tests
Ran `pytest -v` — **3 tests passed**:
- `test_starts_at_N0` — initial count is N0
- `test_rejects_negative_rate` — ValueError on negative rate
- `test_matches_law` — average over many seeds matches N0 * exp(-lam * t)

### Speed comparison (`speed.py`)

| Version             | Time (s) |
|---------------------|----------|
| Pure-Python loop    | 2.7031   |
| NumPy vectorised    | 0.0003   |
| **Speed-up**        | **8944.6×** |

### Conclusion
The NumPy version is significantly faster than the pure-Python loop because
it performs the binomial sampling for all atoms in one vectorised operation
instead of iterating over each atom in Python. All three tests pass, which
confirms that both implementations respect the physical law and handle
invalid input correctly.

## PW1 - Lab B: Real Data and a Snakemake Pipeline

**What I built:**
Read a real radioactive-decay dataset (`decay_observed.csv`), compared it to the
analytical law N0·exp(-λt) with a side-by-side plot, and automated the figure
with a small Snakemake pipeline.

**What the data showed:**
The observed points follow the analytical decay curve closely — the decay
matches the exponential law well over the whole time range, with only small
statistical fluctuations around the curve.

**Pipeline:**
The `Snakefile` defines a rule that builds `figure.png` from
`decay_observed.csv` by running `plot.py`. Running
`snakemake --cores 1 figure.png` rebuilds the figure only when an input
changes, so the pipeline stays consistent.

**Conclusion:**
The real data confirms the analytical decay law. Snakemake makes the workflow
reproducible: one command rebuilds exactly what is out of date, nothing more.

## PW2 - Lab A: Motion from Tracking Data

**What I built:**
Analysed noisy free-fall position measurements (`freefall.csv`, 80 samples at
0.1 s) by differentiating once to get velocity and twice to get acceleration,
then integrating the acceleration back up to recover velocity and position.

**Mean acceleration:**
- From numerical differentiation (`np.gradient` twice): **−8.58 m/s²** —
  close to −g, but noisy because the second derivative amplifies measurement
  noise.
- From a quadratic fit (`np.polyfit(t, y, 2)`): **−9.799 m/s²** —
  essentially −g, confirming free fall.

**Why the acceleration is noisy:**
Differentiation is a high-pass operation: it amplifies high-frequency noise.
Applying `np.gradient` twice makes the noise much worse, so the acceleration
panel looks rough even though the underlying physics is constant −g.

**What integrating back showed:**
Integrating the noisy acceleration back to velocity and then to position
recovered the original position to within about a metre. Integration is a
low-pass operation: random noise partly cancels out, the opposite of
differentiation. The recovered position is smooth and matches the measured
data, showing that integration cleans up the noise.
s
**Bonus — 2D trajectory:**
Read `trajectory.csv` (time, x, y), plotted the path (x vs y), and computed
the speed |v| = √(vx² + vy²) using `np.gradient` on each coordinate. The speed
vs time plot is noisy, again showing that numerical differentiation amplifies
measurement noise.

## PW2 - Lab B: Optimization in Chemistry

**What I built:**
Compared three optimisation methods — gradient descent, Newton's method, and
SLSQP — then applied them to three chemistry problems: fitting a reaction rate
constant, finding a chemical equilibrium, and locating a titration's
equivalence point.

### Part 2 — Three routes to a minimum

**2A: easy convex function** `f(x) = (x−3)² + 1`, from `x0 = 0`:

| Method | x |
|---|---|
| Gradient descent | 3.000000 |
| Newton | 3.000000 |
| SLSQP | 3.000000 |

All three reach **x ≈ 3**. This easy case hides the differences between methods.

**2B: harder landscape** `g(x) = x⁴ − 3x² + x + 5`.

`g′(x) = 0` has three stationary points:

| x | g″(x) | Type |
|---|---|---|
| −1.3008 | +14.31 | minimum |
| +0.1699 | −5.65 | **maximum** |
| +1.1309 | +9.35 | minimum |

**From `x0 = 0`:**

| Method | x | g(x) | Notes |
|---|---|---|---|
| Gradient descent | −1.300839 | 1.4861 | local minimum |
| Newton | 0.169938 | 5.0841 | **maximum** (g″ < 0) |
| SLSQP | −1.300857 | 1.4861 | local minimum |

**From `x0 = 2`:**

| Method | x | g(x) | Notes |
|---|---|---|---|
| Gradient descent | 1.130901 | 3.9298 | local minimum |
| Newton | 1.130901 | 3.9298 | local minimum |
| SLSQP | −1.300639 | 1.4861 | **global** minimum |

**Observations:**

- **Do the methods agree?** No — on this landscape they disagree strongly.
  From `x0 = 0`, GD and SLSQP find the global minimum at `x ≈ −1.3`, but
  Newton lands on the **maximum** at `x ≈ 0.17`. From `x0 = 2`, GD and Newton
  get stuck in the local minimum at `x ≈ 1.13`, while SLSQP still finds the
  global minimum at `x ≈ −1.3`.

- **Did Newton find a minimum or another stationary point?**
  Newton solves `g′(x) = 0` but does not check curvature. From `x0 = 0` it
  found the **maximum** (`g″ < 0`). This shows that solving `g′(x) = 0` is
  not enough — you must check the sign of `g″`.

- **How did the starting point change the result?** Dramatically. `x0 = 0`
  and `x0 = 2` led to different stationary points. The starting point decides
  which basin of attraction you fall into.

**Key lesson:** on a simple convex problem all methods agree; on a complicated
landscape the starting point and the algorithm matter.

### Part 3 — Fit a reaction rate

Read `kinetics.csv` (time, concentration), assumed first-order decay
`C(t) = C0·exp(−kt)`, and minimised the squared error with SLSQP
(`bounds=[(0,5)]`, `x0=0.5`).

- **Fitted rate constant:** k ≈ **0.25** (matches the expected value)
- The fitted curve passes through the noisy data (`kinetics.png`)

### Part 4 — Chemical equilibrium

For H2 + I2 ⇌ 2 HI with equilibrium constant K, the equilibrium extent x
satisfies `(2x)² / ((1−x)²) − K = 0`. Solved two ways:

- **Newton** (root-finding on the imbalance)
- **SLSQP** (minimising the squared imbalance)

Both give the same x, as expected.

- **Equilibrium composition** (with K = 54):
  - n_H2 = 1 − x ≈ 0.214 mol
  - n_I2 = 1 − x ≈ 0.214 mol
  - n_HI = 2x ≈ 1.572 mol
- The plot (`equilibrium.png`) shows H2 and I2 falling, HI rising, with the
  equilibrium extent marked.

### Part 5 (bonus) — Titration equivalence point

Read `titration.csv` (volume of base, pH), computed the slope `d(pH)/dV` with
`np.gradient`, and found the volume where the slope is largest
(`np.argmax`). This is the equivalence point.

- **Equivalence point:** V ≈ **50 mL** — exactly where the pH curve jumps.
- The plot (`titration.png`) shows the pH curve on the left and its slope on
  the right, peaking at the equivalence point.

**Conclusion:**
Optimisation and root-finding are the same idea in different clothes. For easy
convex problems all methods agree, but on real (noisy, multi-minimum) chemistry
problems the starting point, the algorithm, and the curvature check all matter.
Fitting, equilibrium, and titration all reduce to "minimise an error" or
"solve f(x) = 0", which `scipy.optimize` handles in a few lines.