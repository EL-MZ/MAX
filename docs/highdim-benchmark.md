# Synthetic random-block target and known normalizing constant

The default is now proper conjugate regression; see
[Regression evidence and direct posterior samples](regression-evidence.md).
This document describes the earlier synthetic density stress test, available
through `--family synthetic`. Its normalizing constant is assigned by
construction rather than computed from independently specified regression data.

This benchmark prepares distributions at 3,000, 5,000, and 10,000 dimensions.
The true distribution contains randomly shuffled independent blocks of orders
1--5. Morph uses its existing pair search by default. The scripts call the
existing `fit`, `sample`, `log_prob`, and `bridge_sampling` APIs; they do not
change the Morph mechanism, candidate enumeration, or SGM search.

## Exact target

`benchmarks/highdim_target.py` provides `make_random_target` and `load_target`.
Every block has random coordinate indices, correlation signs and strengths,
locations, scales, and shape parameters. Requested orders appear at least once
when the dimension permits; subsequent block widths are sampled uniformly.
Residual coordinates are singletons. The default shapes are:

- Gaussian with a positive-definite random signed correlation matrix.
- Banana: an invertible quadratic shear of a correlated Gaussian, with unit
  shear Jacobian. Curvature and noise vary between blocks.
- Two-component Gaussian mixture with random means, weights, and correlations.
- Sinh-arcsinh: an invertible skew/tail transform of a correlated Gaussian,
  including its exact Jacobian.

All blocks have support on the full real space. The exact joint density is
`p(x) = product_b p_b(x_b)`. Sampling and density evaluation use the same saved
parameters. Densities retain all Gaussian, mixture, affine, and nonlinear
Jacobian constants; none are estimated with KDE.

The unnormalized target is `f(x) = exp(c) p(x)`, so its exact normalizing
constant is `log Z = c`. The default is `c = -0.1 * dimension`, giving -300,
-500, and -1000 for the three large cases. Use `--log-evidence` to set a
different constant. Keep evidence in log space.

A proper standard-normal prior is provided by `target.log_prior`. The
constructed log likelihood is `log L = log p + c - log prior`. Therefore
`log L + log prior = log f`, and the Bayesian evidence is exactly `exp(c)`.
This is a manufactured normalization benchmark, with exact iid posterior draws,
rather than an observation simulator or a test of posterior sampling.

Optional `--shapes parity` creates noisy parity blocks of orders 3--5 with
independent proper subsets. It is excluded from the default shape mixture.
It can test a pair model's inability to represent pure higher-order dependence.

New command-line runs use `--correlation-levels low medium strong`: absolute
posterior Pearson correlation ranges 0.10–0.30, 0.40–0.65, and 0.80–0.95.
Use `--correlation-levels strong` for pronounced correlation in all eligible
blocks, or `--correlation-levels original` for legacy targets. Blocks receive
an invertible physical affine transform based on their covariance; the exact
density includes its determinant. Gaussian, banana and mixture moments are
analytical; sinh-arcsinh uses deterministic Gaussian quadrature, so its Pearson
calibration is approximate. Pure parity blocks retain pairwise independence,
and singletons are unchanged. The Python factory retains the original default.
Three default corner plots prefer one level each, ranking groups within the
level. Independent blocks stay independent, so cross-block pairs in a 10D
corner still show little correlation. Plot titles report the within-component
mean separately from the mean over all plotted pairs.

## Prepare data without running Morph

Use Python from an environment containing this repository's benchmark extra
(NumPy and SciPy). This module does not import JAX or run Morph.

```bash
python -m benchmarks.highdim_prepare \
  --family synthetic \
  --dimensions 3000 5000 10000 \
  --orders 1 2 3 4 5 \
  --samples 2000 --chunk-size 256 \
  --target-seed 20261006 --sample-seed 20261007 \
  --output benchmark-results/highdim-data
```

In this checkout's existing WSL environment, replace `python` with
`/home/apokrypha/.local/share/max-jax/.venv/bin/python`. For a uv environment,
use `uv run --locked --extra benchmark python`; retain `--extra cuda` if that
environment also contains the project's CUDA dependencies.

Each dimension directory contains:

- `target.json`: complete exact parameters and the known log evidence.
- `posterior_samples.npy`: float64 `(2000, dimension)` exact posterior draws.
- `posterior_log_density.npy`: cached `log f` for each corresponding row.
- `manifest.json`: seeds, chunk size, dimensions, counts, and completion status.

Arrays are written in chunks using NumPy memory maps. A 2,000-by-10,000 posterior
array uses about 153 MiB on disk. The generator refuses to overwrite an existing
dataset. Fix seeds **and chunk size** to reproduce the exact sample stream;
saved target parameters can be loaded independently of RNG versions.

Use the first 1,000 rows for fitting and the remaining 1,000 for bridge posterior
draws, or choose another disjoint split. Do not use the true block partition for
Morph fitting. It is available for diagnostics and plotting only.

```python
from pathlib import Path
import numpy as np
from benchmarks.highdim_target import load_target

directory = Path("benchmark-results/highdim-data/10000d")
target = load_target(directory / "target.json")
posterior = np.load(directory / "posterior_samples.npy", mmap_mode="r")
cached_log_f = np.load(directory / "posterior_log_density.npy", mmap_mode="r")

# Exact evaluations for proposal points, including every normalization constant:
proposal_log_f = target.log_unnormalized(proposal_points)
exact_log_z = target.log_evidence
```

The analytic target implementation is NumPy/SciPy. Evaluate its densities
externally and pass the four log-density vectors to `jax_morph.bridge_sampling`.
The convenience `estimate_evidence` requires a JAX-compatible scalar target,
so the supplied runner uses the public bridge API directly.

## Native Morph workflow runner

`scripts/run-highdim.sh` selects CPU or GPU before importing JAX, enables float64,
and uses the existing local WSL environment if present. `--python /path/to/python`
or `--uv` selects another environment. GPU mode requires a GPU and does not
silently fall back to CPU.

The runner defaults to order 2, Silverman bandwidth, no ridge, all fitting rows
for TC, the top five SGM seeds, and unlimited candidate enumeration. An explicit
`--n-seeds` value overrides the default seed budget; there is no square-root rule.
Options expose the current library controls; no candidates are screened or
silently pruned. `--target-orders` changes the true distribution's widths,
while `--morph-order` independently chooses Morph's approximation order.

Preview large-run candidate counts without fitting:

```bash
bash scripts/run-highdim.sh cpu --dimensions 3000 5000 10000 --dry-run
```

| Dimension | Native pair candidates |
| ---: | ---: |
| 200 | 19,900 |
| 500 | 124,750 |
| 3,000 | 4,498,500 |
| 5,000 | 12,497,500 |
| 10,000 | 49,995,000 |

All native candidates are enumerated without a candidate-count cap.
Choose TC sample count, SGM seeds, and batching yourself.
For prepared data, the runner infers dimension from its manifest:

```bash
bash scripts/run-highdim.sh gpu \
  --dataset benchmark-results/highdim-data/3000d \
  --morph-order 2 \
  --output benchmark-results/highdim-3000
```

Add your desired existing Morph controls such as `--tc-samples`, `--n-seeds`,
`--candidate-batch-size`, `--seed-batch-size`, `--sgm-early-stop`,
`--bandwidth-scale`, and `--regularization`. The example intentionally leaves
those choices at the native defaults. `--help` lists all options.

## Small workflow check

This explicit configuration is a short plumbing check, not an accuracy-tuned
configuration for the high-dimensional runs:

```bash
bash scripts/run-highdim.sh cpu \
  --family synthetic \
  --dimensions 200 500 --morph-order 2 \
  --fit-samples 256 --posterior-samples 256 --proposals 256 \
  --tc-samples 32 --n-seeds 4 \
  --candidate-batch-size 32 --seed-batch-size 4 --sgm-early-stop \
  --query-chunk-size 64 --proposal-chunk-size 128 --plot-samples 256 \
  --output benchmark-results/highdim-smoke-2026-10-06
```

Results include selected pairs, dependency precision/recall, raw forward KL
and its conditional Monte Carlo standard error, MorphZ log evidence and error
against the exact constant, importance ESS, bridge overlap and convergence,
iteration history, synchronized timings, and sampled memory diagnostics.
Training and bridge posterior draws are independent. All proposal draws enter
the estimate. Proposal sampling/evaluation runs in bounded chunks; final KDE
factors still use all fitting rows. Timings include first-call compilation and
transfers; a separate warm density timing is also recorded.

The exact-proposal control feeds the analytic posterior density to the bridge
solver and must recover `c` to 1e-8. This tests bridge arithmetic independently
of Morph density quality. Pair Morph is expected to lose some of the true
3--5-coordinate dependencies. A converged bridge fixed point alone does not
establish an accurate estimate. The runner labels an estimate as poor overlap
when it does not converge, its reported relative error exceeds 0.2, or either
bridge overlap is below 0.01. These diagnostic thresholds are recorded conventions,
not accuracy guarantees. Monte Carlo errors exclude uncertainty across fitted
models; `--seeds` can repeat fits on the same saved target.

Outputs are checkpointed in `metrics.json` per case and combined in `results.json`.
Use `--resume` with identical options to skip completed cases. An interrupted
case is rerun from its recorded seed. Plots now show three disjoint 10-parameter
corners, preferring one configured correlation level each, plus the evidence
iteration history. `--corner-count` and `--corner-dimensions` override these
defaults. True dependency components within each level are ranked by their mean absolute Pearson
correlation in the held-out plotting samples. Whole components are kept together
when they fit; smaller independent components share a panel. Selection computes
only small component correlation matrices, never a full 10kD matrix. Parameters
are not reused across panels; small targets produce fewer or smaller plots.
Files are `corner_01_10d.png`, `corner_02_10d.png`, and `corner_03_10d.png`.
The selected indices and observed mean/maximum absolute correlations are saved
in `metrics.json` under `corner_plots`. Plot selection does not change the target
or Morph fit, and independent components retain their lack of cross-correlation.
The evidence history adds a capped error bar at the final actual iteration,
using the reported relative standard error as an approximate log-Z standard
error (`SE(log Z) ≈ SE(Z)/Z`). That error is evaluated at the reported averaged
estimate; the marker stays at the final solver iterate. Nonfinite/unavailable
errors are omitted.
The evidence plot title also shows the full target dimension and total runtime
in `h:mm:ss`, rounded to the nearest second, matching the terminal summary.
`total_runtime_seconds` measures each case from target generation/loading through
evidence estimation and diagnostics, including first-call compilation; plot
rendering is excluded. Historical metrics use their saved `workflow_seconds`
when the newer total is unavailable.
`--no-plots` skips them. Local results remain in the ignored `benchmark-results/`
directory.

For corner comparisons, evidence plots and a short text summary, add `--plots-only`:

```bash
bash scripts/run-highdim.sh cpu --plots-only \
  --output benchmark-results/highdim-plots
```

Every case writes `summary.txt`, including in `--plots-only` mode. It lists
dimensions, evidence and relative error, KL(Post || Morph) in nats with its
conditional Monte Carlo standard error, runtime stages in `h:mm:ss`, and
survived/total proposals. Survived means finite target density; all proposals
still enter bridge estimation. The report includes importance ESS separately.

The posterior statistics distinguish parameters in known dependent components
from empirical linear correlation. The latter counts parameters with at least
one within-component pair having `|Pearson r| >= 0.30`, using up to `--plot-samples`
held-out exact draws. Positive and negative pair counts are listed separately.
Known nonlinear scale, banana, mixture, transformed and parity dependence types
are listed as applicable. Order counts are disjoint components (pairs,
triplets, etc.), not every subset or pure interaction orders. Coupled regression
also lists the underlying base-group counts; Morph's selected orders are shown
alongside the posterior counts. Historical runs without posterior draws list
only the known structural counts and types.

`--plots-only` suppresses `target.json`, `metrics.json`, `configuration.json`, and
`results.json`. Stage and progress messages remain enabled, controlled by
`--progress-reports`. Candidate scoring, SGM seeds, proposal batches and bridge
iterations use live tqdm bars; `--progress-reports 0` disables those bars.
The terminal also prints each case's evidence estimate,
exact evidence, relative standard error, convergence, total runtime, and the
plot directory. Use a fresh output directory
for a folder containing only plots and `summary.txt`. This mode cannot be combined with `--resume`
or `--no-plots`; checkpointing remains available in the default mode.
