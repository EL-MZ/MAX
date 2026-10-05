# Measured findings

The merged JAX Morph implementation recovered the feasible true block partitions
in this suite. Density accuracy remains limited by Silverman KDE smoothing and
support leakage at 1,000 training vectors. The code under `src/jax_morph` was
not changed for these benchmarks.

## Scope and validation

- Six analytic target families, dimensions 6, 12, 20, and 30, Morph orders 1–3.
- 72 main fits and seven 30D full-scoring comparisons: **79 fitted models**.
- Each training, held-out evaluation, and proposal batch has 1,000 vectors,
  hence 1,000 values per parameter. Batches use separate random streams.
- Input columns are shuffled before fitting. Known true blocks enter only the
  oracle baseline, block diagnostics, and plot display ordering.
- Main scoring uses 256 of the 1,000 training vectors. Final KDE factors use
  all 1,000 vectors; SGM tries up to 64 seeds from the complete candidates.
- 67 tests pass on CPU JAX 0.4.35 and JAX 0.11.2. The benchmark itself uses
  Python 3.12.14, JAX/JAXlib 0.4.35, float64, and a GitHub-hosted AMD EPYC 7763
  runner exposing two logical CPUs.
- 100 images: 72 corner pages, 24 partition adjacency plots, and four summaries.
  Corner pages cover every parameter; axes show true-block order for readability.

## Matching-order results at 30D

Forward KL is in nats. Errors are Monte Carlo standard errors conditional on the
fitted model, from 1,000 independent truth draws; training variability is not
included. Timings are synchronized warm wall times with 256-point TC scoring.

| Target | Order | Exact partition | Forward KL ± MC SE | Warm fit (s) | Proposal ESS / 1,000 | Outside support |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| Independent mixed marginals | 1 | Yes | 2.247 ± 0.278 | 0.014 | 117.6 | 74.8% |
| Strong copula pairs | 2 | Yes | 3.180 ± 0.131 | 0.281 | 42.2 | 72.7% |
| Weak copula pairs | 2 | Yes | 2.676 ± 0.143 | 0.279 | 35.4 | 76.9% |
| Copula triplets | 3 | Yes | 4.267 ± 0.562 | 1.885 | 53.7 | 77.6% |
| Banana pairs | 2 | Yes | 8.800 ± 0.207 | 0.280 | 2.4 | 0.0% |
| Multimodal pairs | 2 | Yes | 2.959 ± 0.062 | 0.280 | 21.9 | 0.0% |

The known-block KDE gives the same KL as each matching-order 30D fit, within
floating-point tolerance. The residual error therefore comes from density
approximation on these examples, rather than incorrect block selection.

### What was recovered

Order 2 recovered all true pairs at every tested dimension in the strong, weak,
banana, and multimodal targets: all 16 paired-target partitions were exact.
Order 3 recovered the triplets exactly at 6D, 12D, and 30D.

The 20D triplet target contains six triplets and a final pair. Order 3 forces the
two leftover coordinates into separate singleton factors, so it retains 18/19
true dependent pairs (94.7%) and cannot reproduce that final two-variable block.
This follows the current fixed-order representation; it is not an SGM selection
error. Orders 2 or 3 also force merges on independent targets. Order 1 is the
appropriate control for that family.

Mismatched orders lose real dependencies. At 30D, order 3 retains only 53.3%–66.7%
of the true pairs in the paired targets and introduces unrelated within-factor
pairs. Its higher-dimensional factors do not automatically improve the fit.

### What remains inaccurate

The banana fit recovers all blocks but has KL 8.800 ± 0.207 nats and proposal
ESS about 2.4 out of 1,000. Its curved factors are strongly oversmoothed by the
current KDE bandwidth; the known-block KDE has the same limitation.

Positive and bounded mixed marginals expose a second limitation. The strong-pair
fit generates 72.7% of proposals outside the exact target support. Gaussian KDE
has positive density there, so reverse KL against these targets is infinite.
The implementation retains these proposals and preserves zero target density;
no likelihood floor or rejection-based renormalization was used.

An exact block partition is therefore useful evidence of selection correctness,
but does not establish an accurate or efficient importance proposal. These
results identify smoothing and support handling as the remaining density issues.

## Speed and scoring tradeoff

At 30D, main warm fits take approximately **0.28 s at order 2** and **1.89 s at
order 3**. Full 1,000-point scoring raises those times to approximately **3.4 s**
and **27.7 s**, respectively: about 12× and 15× the main scoring configuration.

The five order-2 full-scoring fits preserve the same forward KL on the paired
targets; their recovered partitions are unchanged. The triplet order-3 result
also remains exact with the same KL. Order-3 strong-pair selection and the
order-2 triplet cross-block grouping change slightly, but the order mismatch
still dominates their errors. This single-seed suite does not establish that
thinning is equally safe for more difficult targets.

SGM alone takes about 8 ms for pairs and 90 ms for triplets in the main 30D run,
roughly 3%–5% of the corresponding warm complete fit. Most fit time is elsewhere,
predominantly the candidate KDE scoring stage. There are 435 pair candidates and
4,060 triplet candidates. These are CPU measurements, not a GPU speed claim or
an end-to-end comparison with the original MorphZ.

The separate CI SGM comparison tests identical outputs for early stopping and
full scans. Early stopping has speed ratios of 0.991 on JAX 0.4.35 and 0.750 on
JAX 0.11.2, so it gives no CPU speed benefit in those runs. Compare strategies
within each run; do not infer a version speedup from different CI jobs.

## Read the artifacts

- [All metrics, hardware, seeds, and selected blocks](results.json)
- [Scalar metrics](metrics.csv)
- [Exact target parameters](targets.json)
- [SGM CI strategy comparison](sgm_ci_comparison.json)
- [Corner plots and summaries](README.md)
- [Plot reconstruction trace](plot_metadata.json)

The original measurement JSON is preserved. Plot regeneration rebuilds factors
from the saved partitions and seeds, verifies all 72 main held-out KL values,
and fixes histogram colours and nonlinear marginal labels. The regenerated
plots use contour-only overlays for clarity. The independent evaluation and
proposal samples never enter fitting or block selection.
