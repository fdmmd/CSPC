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