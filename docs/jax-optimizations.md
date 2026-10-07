# JAX execution enhancements

The pre-change work was committed and pushed to `main` as
`5af3a4e52b5a6c34aa4887d0bffd6372339404f9` before these changes.
This document covers review items **1-5 and 7**. The statistical method,
bandwidth rule, sample selection, SGM search budget, precision defaults and
bridge stopping criterion remain unchanged. Floating-point reductions can
change in their last few bits; bitwise-identical TC scores are not promised.

## 1. Fit marginal KDEs once

`correlation.py` no longer calls the public `fit_kde` once per dimension just
for validation and discards those fits. The compiled TC kernel fits each
marginal once and reuses its evaluation for scoring. Bandwidth and ridge options
are checked separately through a shared helper. A device-side reduction checks
all marginal entropies and candidate scores; only its Boolean result returns to
the host. Invalid or constant unused dimensions are still rejected.

## 2. Reuse target compilation

`evidence._evaluate` is now a module-level JIT function, with the target callable
and chunk size static. `lax.map(..., batch_size=...)` replaces repeated Python
batch dispatch and concatenation. Repeated calls with the same callable, shapes,
dtypes and options reuse compilation. Remainders evaluate only real points;
there are no artificial padded target evaluations. Different shapes or callable
identities can still compile separately. Keep the target callable stable across
workflow calls. The callable must be hashable, as required for a JAX static argument.

## 3. Retain intermediate arrays on the device

- Internal Morph selection uses `_select_validated` after candidate generation
  and TC validation. It avoids downloading the full candidates and scores,
  checking completeness again and uploading them back. The public SGM API still
  validates externally supplied candidate collections on the host; score checks
  reduce to a Boolean on the device.
- TC block-value validation and score validity checks return only small flags.
- Bridge inputs are converted to JAX once. Their validity is reduced to three
  flags; density subtraction runs on the device. Iteration histories and their
  final concatenation remain JAX arrays.
- Cached posterior target densities stay in JAX through slicing and validation.

Input sample/weight validation and discrete selection orchestration still use
the host. The selected block indices are transferred to construct factor groups.
Uncapped bridge control still reads scalar completion/convergence information
between chunks. This is not a completely host-free workflow.

## 4. Compute bridge diagnostics once after averaging

`_bridge_iterate` returns only the final iterate, convergence status, iteration
count and history. The public workflow runs one or more chunks, averages the
requested recent iterates on the device, then calls `_bridge_statistics` once.
This removes repeated overlap, ESS, variance and optional FFT-IAT calculations.
Averaging is compiled separately so varying history lengths do not recompile
the statistics and FFT kernel.
Nonfinite recent values retain the failure instead of hiding it with averaging.

The exported `bridge_kernel` retains its previous contract: it reports statistics
at the final unaveraged iterate. `bridge_sampling` continues to report its recent
iteration average while exposing the actual final iterate and complete history.

## 5. Retain marginal entropies rather than all marginal log densities

Each marginal evaluation is immediately reduced to its weighted entropy:

```text
H_i = -sum_n weight_n * log q_i(sample_n)
TC(block) = sum_n weight_n * log q_block(sample_n) + sum_i_in_block H_i
```

The retained marginal output shrinks from `N * D` values to `D` values. Each
candidate avoids gathering and summing an `N * order` marginal submatrix.
The expectation remains weighted, with the existing zero-weight handling and
negative-score clipping. Reduction reordering is tested against the original
pointwise formula, including samples translated by a large offset.

## 7. Group factor state and bound compiled factor loops

Same-shaped factors share stacked KDE arrays and an array of block indices.
Final fitting groups factors by width and uses bounded batches of at most 16
factors, with a single aggregate covariance-validity check. This removes one
public validation/upload/synchronization cycle per factor.

Density evaluation scans over each group with an accumulated log density,
avoiding an unrolled computation for every factor and avoiding a full
factors-by-queries output matrix. Sampling processes bounded factor batches and
preserves the original per-factor random-key assignment.

Block indices are now dynamic PyTree leaves. Replacing a partition with another
having the same group shapes and factor-order layout reuses compilation, even
inside an outer `jax.jit`. Shapes, dtypes, group layout, query count and static
options can still trigger recompilation.

`MorphState(blocks, factors, dimension)`, `state.blocks`, and `state.factors`
remain available. The tuple views reconstruct the original factor ordering.
Inspect `state.blocks` outside JIT; compiled operations use dynamic array indices.
The internal PyTree representation has changed, so external serialization or
code depending on the old dataclass fields must migrate. Full-state gradients
must account for the new integer index leaves; query gradients remain supported.
The fit-memory benchmark hooks now observe the grouped fitting/selection paths.

## Validation and measurements

Final verification on the implementation:

- Full CPU suite: **131 passed** (33.62 seconds).
- CUDA core suite: **84 passed** (56.78 seconds), covering KDE, Morph, SGM,
  evidence/bridge, averaged and uncapped histories, progress and the new
  optimization regressions. The default device was verified as CUDA.
- `ruff check .`, `ruff format --check .`, and `git diff --check`: passed.
- The checkpoint comparison benchmark ran successfully and verified TC and
  density agreement before reporting timings.


Tests cover weighted KDE equivalence, TC reduction equivalence, singular and
invalid input rejection, partition-independent compilation reuse, target-cache
reuse, query gradients, mixed-precision query/factor promotion, deterministic sampling key assignment, and one final
statistics computation across multiple bridge chunks. Existing tests also cover
SciPy density comparisons, bounded-support targets, evidence, IAT, progress,
SGM ties and averaged/uncapped bridge histories.

A small CPU comparison against the pushed checkpoint used JAX 0.4.35, float64,
seven synchronized warm repetitions and a fixed NumPy random seed of 21:

| Workload | Before, warm median | After, warm median |
| --- | ---: | ---: |
| TC: 96 samples, 12 dimensions, query chunk 32 | 7.71 ms | 6.45 ms |
| Marginal-only fit: 96 samples, 64 dimensions | 13.49 ms | 0.94 ms |
| Density: 32 queries, 64 dimensions | 1.25 ms | 1.01 ms |
| Repeated simple target evaluation: 32 x 64, batch 16 | 18.41 ms | 0.033 ms |

Density first-call time decreased from 964 ms to 59 ms in this run. Grouped
fitting first-call time increased from 58 ms to 167 ms despite its faster warm
execution. First-call figures include compilation and execution; earlier cases
can warm shared primitives. They are not isolated cold-process measurements.
The target result mainly reflects avoiding repeated compilation of a cheap
callback; it is not a prediction for expensive likelihoods or complete workflows.
These small timings are illustrative and do not establish a general speedup.

Reproduce the comparison using the existing project environment:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m benchmarks.jax_optimizations
```

The benchmark reads the checkpoint into a temporary package without changing
the working tree, checks TC/density agreement, and prints device metadata and
timings. For the documented JAX 0.4.x CUDA-wheel layout, retain `CUDA_ROOT` from
[local setup](local-setup.md), including when selecting CPU.

### CUDA progress callbacks

For this JAX 0.4.35 environment, progress callbacks need a local CPU backend even
when numerical work runs on CUDA. Use `JAX_PLATFORMS=cuda,cpu` when testing or
running with progress enabled, and verify `jax.default_backend() == "gpu"`.
An initial CUDA-only run passed 78 tests but failed four progress-related tests
because callbacks could not find a CPU device; enabling both backends resolved
those failures while the default device remained CUDA. No fallback to CPU
computation was used to pass GPU tests.

## Scope

This change does not add training-point self-whitening reuse (item 6), center
tiling (item 8), a new candidate generator (item 9), streamed TC preparation
(item 10), or new user-facing precision/batch defaults (item 11). Those remain
separate optimization opportunities.
