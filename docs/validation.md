# Validation record

## Checks

- Weighted/unweighted KDE log densities and covariance against SciPy Silverman
  KDE in one, two, and three dimensions; numerical normalization; mixture moments.
- Query gradients against finite differences, finite far-tail log density,
  singular covariance and invalid weight handling.
- JAX SGM against a direct Python transcription of paper Algorithm 1, including
  orders 1-4, residual dimensions, top-K/all seeds, ties, and compiled dynamic scores.
- Small exact set-packing bound and monotonic retained score as seeds increase.
- KDE TC for a correlated Gaussian and nonlinear dependence with low Pearson
  correlation; invariance to coordinate translations/scales.
- Complete Morph partition, residual singleton, full-data fitting after optional
  TC subset selection, independent factor draws, deterministic keys, and gradients.
- Exact-proposal bridge estimates with very large log normalization offsets,
  bounded target support, no overlap, explicit nonconvergence, and IAT adjustment.
- End-to-end beta evidence and correlated Gaussian evidence with known constants,
  disjoint sample splitting, and cached target-evaluation counts.
- Conjugate Gaussian example with an analytically known marginal likelihood.
- SGM benchmark checks identical selections before reporting warm timings.

## Execution

The local Windows runtime initially had NumPy but no JAX, SciPy, or pytest.
Dependency installation was blocked by socket/network restrictions. Local Python
syntax compilation is possible; numerical execution is delegated to the PR's
GitHub Actions workflow on Python 3.10/JAX 0.4.35 and Python 3.12/current JAX.
The first numerical run caught rounding-sensitive covariance singularity and
constant-chain centering behavior across JAX versions. Both were corrected by
stable centering and an explicit correlation-eigenvalue check. SciPy reference
KDEs exclude zero-weight centers, which contribute no mixture mass, to avoid a
reference logpdf NaN issue in SciPy. On commit ee5e10f, all 43 numerical tests,
the conjugate Gaussian example, lint, and the SGM benchmark passed on both
JAX 0.4.35 and 0.11.2. Subsequent checks also exercise both SGM scan strategies.
Run [37265043256](https://github.com/EL-MZ/MAX/actions/runs/37265043256) passed
58 tests on each version, including both SGM scan strategies, plus the Gaussian
example and reference-equivalence benchmark. The final workflow checks committed
formatting without modifying files. Full initial timing metadata is preserved in
[`sgm_cpu_benchmark.json`](sgm_cpu_benchmark.json).

## Initial CPU benchmark

[GitHub Actions run 37264635932](https://github.com/EL-MZ/MAX/actions/runs/37264635932),
commit ee5e10f, d=30, L=2, 435 candidates, K=64, 20 synchronized warm repetitions:

| JAX | Full scan median | Early-stop median | Full / early speed ratio |
| --- | ---: | ---: | ---: |
| 0.4.35 | 8.454 ms | 7.840 ms | 1.078 |
| 0.11.2 | 0.782 ms | 0.991 ms | 0.789 |

Both returned identical selected blocks and seed totals. Early stopping was kept
as an explicit option after this mixed result; full scanning is the default.
These single-run CPU measurements are workload-specific and do not establish
general speedups. The JSON report also records compilation/first-call costs.

The conjugate Gaussian example had exact log Z = -5.458274. The estimates were
-5.456247 (JAX 0.4.35, RE=0.004424) and -5.459431 (JAX 0.11.2, RE=0.004784).
