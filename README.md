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