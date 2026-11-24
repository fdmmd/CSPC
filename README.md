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

| Gradient descent | 3.000000 |
| Newton | 3.000000 |
| SLSQP | 3.000000 |

All three reach **x ≈ 3**. This easy case hides the differences between methods.

**2B: harder landscape** `g(x) = x⁴ − 3x² + x + 5`.

`g′(x) = 0` has three stationary points:

| x | g″(x) | Type |

| −1.3008 | +14.31 | minimum |
| +0.1699 | −5.65 | **maximum** |
| +1.1309 | +9.35 | minimum |

**From `x0 = 0`:**

| Method | x | g(x) | Notes |

| Gradient descent | −1.300839 | 1.4861 | local minimum |
| Newton | 0.169938 | 5.0841 | **maximum** (g″ < 0) |
| SLSQP | −1.300857 | 1.4861 | local minimum |

**From `x0 = 2`:**

| Method | x | g(x) | Notes |

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
---

## PW3: Data, Distributions, Testing, and Causation

**What I built:**
A complete statistical analysis of two datasets: the UCI Cleveland heart-disease
dataset (303 patients) and a synthetic chemical-exposure dataset (1000 patients).
I characterised distributions, tested normality, compared two groups, measured
correlation, and confronted correlation with causation.

### Session 1 — Characterising the data

**Part 1 — Load:**
`heart.csv` has 303 rows × 14 columns. The four continuous variables are `age`,
`chol`, `trestbps`, `thalach`.

**Part 2 — Distributions:**
The distributions of all four variables are roughly bell-shaped but with
noticeable deviations from normality (skew, outliers).

**Part 3 — Normality verdicts (Shapiro-Wilk, p < 0.05 = not normal):**

| Variable | W | p | Verdict |

| age | 0.9865 | 0.0061 | **NOT normal** |
| chol | 0.9473 | 5.9e-09 | **NOT normal** |
| trestbps | 0.9665 | 1.8e-06 | **NOT normal** |
| thalach | 0.9765 | 7.0e-05 | **NOT normal** |

**Reasoning:** I used both a graphical check (histograms + Q-Q plots) and a
numerical test (Shapiro-Wilk). The Q-Q plots deviate from the diagonal at the
tails, and Shapiro-Wilk gives p < 0.05 for all four variables. So none of the
continuous variables should be treated as normal.

### Session 2 — Drawing conclusions

**Part 4 — Do heart-disease patients reach a different max heart rate?**

Because `thalach` is NOT normal, I used the **Mann-Whitney U test**
(non-parametric) instead of a t-test.

- Healthy: mean thalach ≈ 158 bpm
- Disease: mean thalach ≈ 139 bpm
- **Mann-Whitney U = 16989.5, p = 1.86e-13**

**Conclusion:** the two groups differ **very significantly** (p ≈ 10⁻¹³).
Disease patients reach a **lower** max heart rate. The 95 % CI error bars
(`thalach_comparison.png`) do not overlap, agreeing with the test.

**Part 5 — Do age and thalach move together?**

Because neither is normal, I used **Spearman** correlation (not Pearson).

- **Spearman rho = −0.392, p = 1.5e-12**

**Conclusion:** there is a **moderate, negative, highly significant**
correlation — older patients reach a lower max heart rate. This matches the
known physiological fact that maximum heart rate decreases with age.

### Chemicals and cancer: correlation ≠ causation

**Part 6 — Naive analysis:**

| Chemical | Spearman rho vs malignancy | p |

| benzene | +0.377 | 5e-35 |
| **cadmium** | **+0.885** | ~0 |

Naive conclusion: **cadmium** looks far more strongly associated with
malignancy — taken at face value, it appears to be the cause.

**Part 7 — Account for the confounder:**

The dataset contains `pollution_index`, which is **almost perfectly correlated
with cadmium** (Spearman rho = +0.982) but **not with benzene** (rho = +0.054).

I restricted the data to patients with similar `pollution_index`
(40 < pollution_index < 60, n = 199) and recomputed the correlations:

| Chemical | rho vs malignancy (all) | rho vs malignancy (subset) |

| benzene | +0.377 | **+0.709** (grew!) |
| cadmium | +0.885 | **+0.297** (collapsed!) |

**The reveal:**

- **Cadmium is NOT the cause.** Its apparent effect was a disguised effect of
  `pollution_index`. Once we remove variation in pollution, cadmium's
  correlation with malignancy drops from +0.885 to +0.297.
- **Benzene IS the real cause.** Its correlation with malignancy actually
  **grows** (from +0.377 to +0.709) once pollution noise is removed — because
  in the full data the pollution confounder was masking its effect.

**How I know:** the confounder test is the standard way to distinguish
correlation from causation. If the association of A with Y **survives**
(or strengthens) after controlling for Z, A is likely a genuine cause. If it
**collapses**, A was only a marker of Z.

### Bonus — Entropy of categorical columns

Shannon entropy H measures how hard a category is to guess:

| Column | Categories | H (bits) | H / log₂(k) |

| target | 2 | 0.995 | **0.995** (almost balanced) |
| sex | 2 | 0.905 | 0.905 |
| cp | 4 | 1.737 | 0.869 |
| thal | 3 | 1.246 | 0.786 |

`target` is almost perfectly balanced (46 %/54 %), so a naive classifier has
~54 % accuracy — a useful model must beat that. `thal` is the most lopsided,
with a majority class of 55 %.

### Conclusion

Statistical methods must be chosen to match the data: because none of the
continuous variables is normal, non-parametric methods (Mann-Whitney, Spearman)
were the right choice over t-test and Pearson. And the chemical-mystery
analysis is a textbook demonstration that **correlation does not imply
causation** — cadmium looked like the cause, but controlling for
`pollution_index` showed that benzene is the real driver.

## PW4 - Session A: Supervised Learning (Concrete Strength)

**What I built:**
A full supervised-learning workflow on the UCI Concrete Compressive Strength dataset (1030 tests, 8 inputs, 1 output). I performed cleaning, linear regression, polynomial regression, classification (KNN & Decision Tree), evaluated metrics, and analysed feature importance.

### A1: Variables and Units
| Variable | Unit |
|---|---|
| cement | kg/m³ |
| slag | kg/m³ |
| ash | kg/m³ |
| water | kg/m³ |
| superplastic | kg/m³ |
| coarse_agg | kg/m³ |
| fine_agg | kg/m³ |
| age | day |
| strength (target) | MPa |

### A2: Cleaning decisions
- Found **25 duplicate rows** (1030 → 1005). 
- **Why duplicates are a problem:** if identical rows end up in both train and test sets, the model essentially "memorises" the test data and gives artificially high scores.
- **0 missing values** found.
- `coarse_agg` (up to 1145) and `fine_agg` (up to 992) are on much larger numerical scales than `superplastic` (up to 32), which is why scaling is needed later for KNN.

### A3: Split
- Train: 804 rows, Test: 201 rows (80/20 split, `random_state=42`).

### A4: Linear Regression
**Fitted equation (raw units):**
`strength = -23.39 + 0.117*cement + 0.098*slag + 0.085*ash - 0.131*water + 0.332*superplastic + 0.016*coarse_agg + 0.021*fine_agg + 0.111*age`

**Metrics:**
| Metric | Value |
|---|---|
| Train RMSE | 10.00 MPa |
| Test RMSE | 11.19 MPa |
| Test MAE | 8.90 MPa |
| Train R² | 0.610 |
| Train R² (adj) | 0.606 |
| Test R² | **0.580** |

**Residuals:** The residuals vs `age` plot shows a clear nonlinear pattern (overpredicting at small ages, underpredicting at large ages). This indicates the model is **misspecified** — strength grows nonlinearly (logarithmically) with age.

### A5: Polynomial Regression
| Degree | Features (p) | Train RMSE | Test RMSE | Train R² | Train R² (adj) | Test R² |
|---|---|---|---|---|---|---|
| 1 | 8 | 10.00 | 11.19 | 0.610 | 0.606 | 0.580 |
| 2 | 44 | 6.99 | 8.31 | 0.810 | 0.798 | 0.769 |
| **3** | **164** | **4.40** | **5.95** | **0.925** | **0.905** | **0.881** |
| 4 | 494 | 2.33 | 39.41 | 0.979 | 0.945 | **-4.206** |

**Choice:** I choose **Degree 3** (best test R² = 0.881).
**Degree 4 overfitting:** The number of features explodes to 494, which is more than half of the training rows (804). The model memorises the train data (R²=0.979) but catastrophically fails on the test set (R² = -4.2). Adjusted R² on the train set (0.945) still looks fine — so adjusted R² **cannot** protect you from overfitting on unseen data.

### A6: Classification (KNN & Tree)
Class target: `strength >= 40 MPa` (1 = yes, 0 = no). Proportions: 64% (0) / 36% (1).

**KNN Sweep:** I chose **K = 7** because it gives the highest test accuracy (0.836). K=1 overfits (train 0.998), and large K underfits.

**Decision Tree Sweep:**
- Unlimited tree (`max_depth=None`) gives the best test accuracy (0.881).
- However, it has **80 leaves** and is impossible to read.
- I chose **max_depth=3** (test accuracy 0.796) for interpretability.
- **Root question:** `age <= 42.00`. This makes physical sense: curing time is the most critical factor for concrete strength.

### A7: Metrics & Thresholds
**KNN (K=7) metrics at threshold 0.5:** TP=54, FP=15, FN=18, TN=114. Accuracy=0.836, Precision=0.783, Recall=0.750, F1=0.766.

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.3 | 61 | 23 | 11 | 106 | 0.726 | 0.847 | 0.782 |
| 0.5 | 54 | 15 | 18 | 114 | 0.783 | 0.750 | 0.766 |
| 0.7 | 39 | 11 | 33 | 118 | 0.780 | 0.542 | 0.639 |
| 0.9 | 18 | 1 | 54 | 128 | 0.947 | 0.250 | 0.396 |

**Safety & FP/FN:** On a construction site, **False Positive (FP)** is the most dangerous mistake (predicting concrete is strong when it is actually weak). **Precision** controls FP. To be safer, we should raise the threshold to **0.7** (fewer FPs), but we pay for it with lower Recall (we reject good batches).

### A8: Feature Importance
| Feature | Raw Coef | Standardised Coef | Tree Importance | KNN Perm. Importance |
|---|---|---|---|---|
| cement | 0.117 | **12.12** | 0.285 | 0.165 |
---
| age | 0.111 | 6.89 | **0.320** | **0.176** |
| slag | 0.098 | 8.44 | 0.000 | 0.055 |
| water | -0.131 | -2.83 | 0.046 | 0.068 |
| superplastic | 0.332 | 1.97 | 0.241 | 0.066 |

- **Raw vs standardised:** `superplastic` looks important in raw units, but after scaling, `cement` dominates.
- **Physical interpretation:** All methods agree that **`age` and `cement`** are the most important. This matches engineering knowledge: water has a negative effect, cement and curing time have positive effects. 
- **Does high importance mean causality?** No! (Link to PW3: correlation ≠ causation). Importance only says the model *uses* the feature, not that changing it will *cause* a change.

### Bonus: Physics-based features
By replacing `age` with `log(age)` and `water` with the water-to-binder ratio (`w/b`), I increased the Test R² from **0.580 to 0.806** without adding any features (still 8). This is nearly as good as the polynomial model with 164 features (0.881), proving that **domain knowledge beats brute-force complexity**.

### Session B: Unsupervised Learning (Wine data)

**B1: Load & describe.** The dataset has 178 wines, 13 chemical measurements. `proline` has a massive range (278–1680) compared to `alcohol` (11–15).

**B2: PCA on raw data.** PC1 explains **99.8%** of the variance, but its loading is **0.9998 for `proline`**. This is an artifact: PCA only centers the data, so the variable with the largest numerical range dominates. This is not a real chemical pattern.

**B3: PCA on scaled data.** After scaling, PC1 explains 36.2%, PC2 explains 19.2%. It takes 8 components to reach 90% variance. The PC1 loadings show a **phenolic axis** (`flavanoids` +0.42, `total_phenols` +0.39 vs `nonflavanoid_phenols` −0.30). PC2 is a **color/alcohol axis** (`color_intensity` +0.53, `alcohol` +0.48).

**B4: K-Means.** `n_init=10` stabilizes the inertia (values vary only from 1277.9 to 1282.5 across different seeds). The elbow method clearly shows **k=3** (inertia drops sharply from 2314 → 1659 → 1278, then flattens).

**B5: Cluster interpretation.** The 3 clusters (sizes 65, 51, 62) differ mostly in `proline`, `magnesium`, `color_intensity`, and `alcohol`. Cluster 0 is light/low-alcohol, Cluster 1 is dark/high-color, Cluster 2 is full-bodied/high-proline.

**B6: The reveal.** 
- Scaled K-Means agreement: **96.63%**
- Raw (unscaled) K-Means agreement: 70.22%
- PCA 2D K-Means agreement: 96.63%

The scaled K-Means aligns almost perfectly with the 3 true cultivars. Without scaling, `proline` dominates and ruins the clustering (70%). 

**B7: Snakemake pipeline.**
The `Snakefile` contains 4 rules: `all`, `fetch`, `session_a`, `session_b`. 
- `snakemake -n` (dry run) shows the plan without executing.
- Running from scratch (`rm -rf data results; snakemake --cores 1`) executes fetch → session_a → session_b.
- Running again reports `Nothing to be done` (all outputs are up-to-date).
- Editing a hyperparameter in `sessionA.py` triggers **only `session_a`** to rerun. `fetch` does not rerun because its input script (`fetch_concrete.py`) and its output (`data/concrete.csv`) are unchanged. `session_b` does not rerun because `sessionB.py` hasn't changed.
- **One command to reproduce everything:** `snakemake --cores 1`

**Bonus: Supervised vs Unsupervised.**
- KNN (13 features) test accuracy: **93.33%**
- KNN (2 PCs) test accuracy: **93.33%**
- K-Means (unsupervised) agreement: **96.63%**

The unsupervised K-Means actually outperformed the supervised KNN. The comparison is not completely fair: KNN is evaluated on a smaller unseen test split, while K-Means agreement is measured on all 178 points. Also, KNN struggles with the curse of dimensionality, whereas K-Means finds global cluster centers that separate the 3 chemical clouds perfectly.

## PW5 — Signal Processing: Where Was the Earthquake, and How Strong?

**What I built:**
A full seismological analysis pipeline on two 3-channel broadband records (Z, N, E, 100 Hz, 300 s). I inspected raw waveforms, computed spectra, manually band-passed in the frequency domain, built envelopes, picked P- and S-arrival times, computed the hypocentral distance and origin time, plotted a spectrogram, estimated PGV and magnitude, and wrapped everything into a reproducible Snakemake pipeline. Region velocities: **v_P = 6.2 km/s**, **v_S = 3.6 km/s**.

### 1. Look at the record
- fs = 100.000 Hz, N = 30000 samples, duration = 299.99 s.
- Three channels Z (vertical), N (north–south), E (east–west) in mm/s.
- The record contains **two phases**: a weak first shaking (P) and a strong later shaking (S + surface waves).
- The first weak shaking is **clearest on Z** — P is longitudinal and arrives almost vertically at the surface.
- The later strong shaking is **largest on N and E** — S is transverse; surface waves also move the ground horizontally.
- The first 60 s are **not perfectly quiet**: background noise ~1e-6..1e-5 m/s (microseisms + electronics).
- **Why:** P is longitudinal (near-vertical at the surface), S is transverse (near-horizontal). "Quiet before the earthquake" means "at the noise floor", not "exactly zero".

### 2. The spectrum
- `fft` returns one complex number per sample; `fftfreq` gives the **frequency in Hz** of each element. Without it, the array index means nothing physically.
- The spectrum is **mirrored around 0 Hz** because the input is real: `X(−f) = conj(X(f))`, so |X| is symmetric. Negative frequencies are a mathematical mirror and carry no new information.
- Quiet-part peak: **~0.15 Hz** — this is the **microseism** (ocean waves), recorded by every station on Earth.
- The whole-record amplitude maximum is also at ~0.15 Hz (the earthquake adds most energy at the low end).
- S/N ratio is highest in 0.05–0.5 Hz, but there the earthquake is mixed with microseisms and drift. The useful band for phase picking is **0.5–10 Hz**.

**Energy per band (spectrum_check.py):**

| band | EQ energy | quiet energy | ratio |
|---|---|---|---|
| 0.05–0.5 Hz | 3.65e-07 | 1.31e-13 | 2.79e+06 |
| 0.5–2 Hz | 1.64e-07 | 4.37e-13 | 3.74e+05 |
| 2–5 Hz | 8.61e-08 | 8.74e-13 | 9.85e+04 |
| 5–10 Hz | 2.47e-08 | 1.46e-12 | 1.69e+04 |
| 10–20 Hz | 8.09e-09 | 2.91e-12 | 2.78e+03 |

### 3. Inverse transform and manual band-pass
- `np.allclose(ifft(fft(Z)).real, Z)` → **True**. Filtering in the frequency domain relies on the FFT/IFFT pair being invertible: go to frequencies, zero out the band, come back to time — that is exactly a linear filter.
- Band-pass **a = 0.5 Hz, b = 10 Hz** applied to Z, N, E.
  - Below 0.5 Hz: microseisms and drift → would mask the start of P.
  - Above 10 Hz: network and electronic noise.
  - Both cut-offs come from the spectra in Section 2.
- `Z − Zfilt` is a smooth, low-frequency signal — exactly the microseisms and drift that were removed. In the first 60 s it is pure noise (no earthquake yet); later it also contains slow surface waves.

### 4. The envelope
- Moving average of |x| with window w samples. Horizontal motion = sqrt(N² + E²).
- **Window 0.05 s** — envelope too rough, follows almost every peak; can't see where a phase starts.
- **Window 0.5 s** — compromise: smooth enough, S packet and onset still visible.
- **Window 5 s** — heavily smoothed; start of P is smeared over several seconds, arrival time impossible to pick. Long window introduces a systematic delay.
- Max Z envelope at **109.80 s**; max horizontal envelope at **152.03 s** — horizontal max comes later, as expected (S + surface waves).

### 5. Arrival times — start, not maximum
- **t_P = 108.42 s** — first time Z envelope exceeds 5% of its max.
- **t_S = 144.08 s** — first time (after t_P + 2 s) horizontal envelope exceeds 10% of its max.
- **dt = t_S − t_P = 35.66 s**.
- Horizontal envelope maximum via `find_peaks` → **t_peak = 152.03 s**, i.e. 7.95 s after t_S.
- **Why an arrival is the start, not the maximum:** the maximum is the peak of the wave packet, which arrives *after* the onset (waves build up gradually). If we used t_peak as t_S, dt would be 43.61 s instead of 35.66 s → distance overestimated by **22%**.

### 6. Distance and origin time
**Formula:** R = dt / (1/v_S − 1/v_P), t0 = t_P − R/v_P.

| Quantity | Value |
|---|---|
| dt | 35.66 s |
| 1/v_S − 1/v_P | 0.116487 s/km |
| **R** | **306.1 km** |
| **t0** | **59.04 s** |

- **Uncertainty:** t_S ± 1 s → R changes by **± 8.6 km per second** (R = 297.5 km at t_S−1 s, 314.7 km at t_S+1 s).
- t0 < t_P ✓ — the earthquake happened ~49 s before the P wave reached the station.

### 7. The spectrogram
- `scipy.signal.spectrogram(raw Z, fs=100, nperseg=256)` → f up to 50 Hz, 133 time bins, displayed 0–10 Hz.
- Before t_P (~108 s): almost black, only a thin low-frequency line (~0.15 Hz) — microseisms.
- Around t_P: vertical flash at higher frequencies (1–10 Hz) — the P wave.
- After t_S (~144 s): the whole band brightens, especially lower frequencies — S + surface waves.
- Later (~200+ s): low frequencies dominate, high frequencies fade — surface waves.
- **What the spectrogram adds over the single spectrum (Section 2):** the spectrum shows *which frequencies exist anywhere in the record*; the spectrogram shows *which frequencies exist at which time*. This is what distinguishes P (early, higher) from S and surface waves (later, lower).

### 8. Magnitude
**Formula:** M = (log10(PGV[cm/s]) + log10(R[km]) + 1.0) / 0.55, PGV = max of sqrt(N² + E²) over the whole raw record.

| Quantity | Value |
|---|---|
| PGV | 1.5105 cm/s |
| R | 306.1 km |
| **M** | **6.66** |

- **Why log10(R):** the same earthquake recorded twice as far away would have a smaller PGV. Without log10(R), M would be underestimated. Concretely: ignoring log10(R) gives M = 2.14, i.e. 4.5 units too small for R = 306 km.
- **M+1 at the same distance:** PGV increases by 10^0.55 ≈ **3.55×** (5.359 cm/s vs 1.5105 cm/s).

### 9. The whole system
- `analyse.py` — one function `analyse(csv_file)` does Sections 3–8 and returns t_P, t_S, dt, R, t0, PGV, M. Runs unchanged on both records.
- `report.py` reads `results/events.csv` and writes `results/report.txt` with one block per event plus a comparison.
- Final table (`results/events.csv`):

| file | t_P (s) | t_S (s) | dt (s) | R (km) | t0 (s) | PGV (cm/s) | M |
|---|---|---|---|---|---|---|---|
| seismogram_A.csv | 108.42 | 144.08 | 35.66 | 306.1 | 59.04 | 1.5105 | 6.66 |
| seismogram_B.csv |  84.25 | 102.43 | 18.18 | 156.1 | 59.08 | 0.5711 | 5.36 |

- **Closer: B** (156.1 km < 306.1 km). **Stronger: A** (M 6.66 > M 5.36). A is farther but stronger; B is closer but weaker.

### 10. Snakemake
`Snakefile` with three rules: `analyse` (inputs `analyse.py`, both CSVs → `results/events.csv`), `report` (inputs `report.py`, `results/events.csv` → `results/report.txt`), `all`.

**(a)** `rm -rf results && snakemake --cores 1` → `analyse` runs (`reason: Missing output files`), then `report` (`reason: Input files updated by another job: results/events.csv`), then `all`. 3/3 done.

**(b)** Second run without changes → **`Nothing to be done`**. Snakemake compares mtimes; nothing changed → nothing runs. Idempotent.

**(c)** Changing only the text in `report.py` → `localrule report` runs (`reason: Code has changed`), `analyse` does **not** run. Only the changed step and its downstream consumers re-execute.

**(d)** Changing `B_HIGH = 10.0` → `8.0` in `analyse.py` → `analyse` runs (`reason: Updated input files: analyse.py`), then `report` runs (`reason: Input files updated by another job: results/events.csv`), then `all`. Both downstream steps re-run.

**Why (d) is what a pipeline should do:** Snakemake tracks **data dependencies**, not just command timestamps. Change analysis code → re-run analysis and everything depending on its output. Change only report wording → re-run only the report. Results are always consistent with the code that produced them, without the user having to remember what to re-run. The `B_HIGH` change in (d) was reverted back to **10.0** so the final results correspond to the justified 0.5–10 Hz band-pass.

### Figures index
| figure | Part |
|---|---|
| `step1_full.png`, `step1_zoom60.png` | 1 |
| `step2a_index_vs_freq.png`, `step2b_loglog.png` | 2 |
| `step3_filtered.png`, `step3_removed.png`, `step3_removed_zoom.png` | 3 |
| `step4a_envelope_windows.png`, `step4b_envelopes_Z_H.png` | 4 |
| `step5_arrivals.png`, `step5_arrivals_zoom.png` | 5 |
| `step7_spectrogram.png`, `step7_spectrogram_annot.png` | 7 |
| `analyse_seismogram_A.png`, `analyse_seismogram_B.png` | 9 |

Sections 6 and 8 have no figures — they compute numbers from Sections 5 and 3.