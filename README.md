# MAX - Jax_Morph

Block-distribution benchmarks, KL diagnostics, timings, and corner plots are
described in [docs/block-benchmarks.md](docs/block-benchmarks.md).
Generated benchmark results and plots are saved locally in `benchmark-results/`,
which is excluded from Git.
For configurable TC-only CPU/GPU scaling tests with progress counters, see
[the TC benchmark](docs/tc-benchmark.md).

JAX implementation of the Morph probability density approximation and the
MorphZ post-processing Bayesian evidence estimator. The baseline follows
[Zahraoui et al., Phys. Rev. D 113, 083014 (2026)](https://doi.org/10.1103/1554-y6ns).

Morph fits a product of small, disjoint KDE factors. Blocks are selected using
KDE estimates of total correlation and Seeded Greedy Maximization (SGM). MorphZ
uses independent draws from that product in optimal bridge sampling.

## Installation

For Linux CPU and CUDA environments managed with uv, see
[local CPU and CUDA setup](docs/local-setup.md).

Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
```

The default JAX installation uses CPU. For accelerators, install the appropriate
JAX build using the [official installation guide](https://docs.jax.dev/en/latest/installation.html).
The library does not change JAX's global precision configuration. Enable 64-bit
precision in your application before constructing arrays for evidence work:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## Density approximation

Input arrays use rows for draws: `(n_samples, n_parameters)`.

```python
import jax
from jax_morph import fit, log_prob, sample

# posterior_samples: an existing post-burn-in sample array
fitted = fit(posterior_samples, order=2, n_seeds=None)
state = fitted.state
draws = sample(jax.random.key(1), state, n=1000)
log_densities = log_prob(state, draws)
gradient = jax.grad(lambda point: log_prob(state, point))(draws[0])
print(state.blocks)
print(fitted.diagnostics.retained_total_correlation)
```

`fit(samples, weights=...)` accepts nonnegative weights. Silverman's bandwidth
is the default. `order=2` fits pairs and any residual singleton; `order=1` fits
independent marginals. Factors must have fewer than ten dimensions. All candidate
blocks are scored; `max_candidates` prevents accidental combinatorial allocation.
SGM is a heuristic, not an exact matching algorithm.
`sgm_early_stop=True` enables an equivalent early-exit scan; benchmark it on your
device before enabling it, since it is not consistently faster across JAX versions.

For cheaper block scoring, explicitly set `max_tc_samples=256` and provide a
`key`. This uses a reproducible subset for total correlation only; the selected
KDE factors still use all fitting samples. It can change the selected partition.

## Evidence estimation

```python
from jax_morph import estimate_evidence


def log_density(theta):
    return log_likelihood(theta) + log_prior(theta)


result = estimate_evidence(
    posterior_samples,
    log_density,
    jax.random.key(2),
    order=2,
    n_proposal=2000,
)
print(result.bridge.log_evidence)
print(result.bridge.relative_error)
print(result.bridge.converged)
```

The callback must be JAX-compatible, return a scalar, and include all likelihood
and proper-prior normalization constants. This workflow requires equal-weight
posterior draws. It splits consecutive rows into disjoint fitting and bridge
batches. It estimates autocorrelation from the ordered held-out bridge sequence;
use `posterior_iat=1` for independent draws. For multiple chains, supply a suitable
IAT explicitly rather than treating a concatenated array as one chain.

Provide `posterior_log_density` with one cached value per original row to reduce
additional target evaluations to the number of proposal draws. Bounded-prior
proposals outside the support must return `-inf`; those draws are retained.

For non-JAX likelihoods, evaluate densities externally and call `bridge_sampling`
with the four log-density vectors. See [the numerical method](docs/method.md).

## Validation and implementation records

```bash
pytest -q
ruff check .
ruff format --check .
python examples/gaussian_evidence.py
python benchmarks/sgm.py --dimension 50 --seeds 128
```

- [Method and equations](docs/method.md): mathematical conventions and scope.
- [Implementation record](docs/implementation.md): modules, optimizations, deviations,
  diagnostics, and limitations.
- [Validation record](docs/validation.md): checks and execution results.
- [Changelog](CHANGELOG.md): versioned changes.
