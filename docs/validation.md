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
Actual run results and any required fixes will be recorded before delivery.
