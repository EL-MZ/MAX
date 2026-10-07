# Proper regression evidence and direct posterior samples

The default high-dimensional benchmark is now grouped Bayesian regression,
with a genuine normalized likelihood, proper prior, analytically computed
evidence, and independent exact posterior draws. `src/jax_morph` and its
pair search remain unchanged. The earlier synthetic shape target remains
available explicitly through `--family synthetic` as a density stress test.

## Posterior correlation levels

New command-line runs default to `--correlation-levels low medium strong`.
Groups cycle through population absolute Pearson correlations of 0.10–0.30,
0.40–0.65, and 0.80–0.95, with randomly signed pairs. Every within-group pair
has the group's recorded strength, including coefficient–variance pairs.
Independent groups remain independent. The three default 10D corners each
prefer a different level and rank groups by measured posterior correlation.
The text summary records the level counts alongside empirical correlation counts.

For visibly strong correlations throughout a new target:

```bash
bash scripts/run-highdim.sh cpu --correlation-levels strong --plots-only \
  --output benchmark-results/highdim-strong
```

The same option is available when preparing datasets. Saved datasets retain
their saved correlations; regenerate the dataset to change them. Use a fresh
output directory when changing levels. `--correlation-levels original` restores
the previous target generation and permits resuming previous configurations.
The Python target factory keeps `("original",)` as its default for compatibility.

Calibration changes the coordinates of the proper model: a coefficient basis
change sets the conditional covariance and a determinant-one shear
`u = beta + k * (t - E[t])` correlates each coefficient with `t`, the variance
or log variance. The prior and likelihood both evaluate the inverse shear.
Conditional covariance has signed equicorrelation `rho/(1+rho)` and diagonal
`(1-rho^2)` times its original diagonal; `k` supplies the remaining marginal
variance. Consequently every physical within-group Pearson magnitude is `rho`.
Exact sampling, prior normalization, likelihood normalization, and analytical
evidence are preserved. The conjugate formulas below refer to the base
coefficients before this shear; physical coefficient marginals are transformed.
Optional cross-group coupling can change physical pair correlations further.


## Gaussian evidence: the proposed formula is correct

For fixed `X`, `m0`, positive-definite `Sigma0`, and positive-definite `R`,
`theta ~ N(m0,Sigma0)` and `y | theta ~ N(X theta,R)` imply
`y ~ N(X m0, C)` with `C = R + X Sigma0 X.T`. Hence

\[
\log Z=-\frac12\{n\log(2\pi)+\log|C|
+(y-Xm_0)^\top C^{-1}(y-Xm_0)\}.
\]

The posterior precision is `Sigma0^-1 + X.T R^-1 X` and its mean is
`Vn (Sigma0^-1 m0 + X.T R^-1 y)`. Correlated columns generally induce coefficient
dependence; this is pairwise dependence in a multivariate Gaussian, rather
than a separate pure higher-order interaction. Integrating over unknown
hyperparameters would require a different evidence calculation.

`gaussian_log_evidence` in `benchmarks/regression_target.py` uses observation-space
Cholesky solves and log determinants. Its matrix factorization depends on the
number of observations, not on directly inverting a 10,000-by-10,000 posterior
covariance. Construction of `C` still depends on the size and structure of `X`
and `Sigma0`.

This is the weight-space Gaussian regression/marginal-likelihood calculation in
[Rasmussen & Williams, Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

## Unknown noise: conditional prior is essential

For each group, let `v = sigma^2` and specify

\[
v\sim\operatorname{IG}(a_0,b_0),\quad
\beta\mid v\sim N(m_0,vV_0),\quad
y\mid\beta,v\sim N(X\beta,vI_n).
\]

Our inverse-gamma convention is

\[
p(v)=\frac{b_0^{a_0}}{\Gamma(a_0)}v^{-a_0-1}e^{-b_0/v},\qquad v>0.
\]

`a0 > 0`, `b0 > 0`, and positive-definite `V0` give proper priors. The covariance
of beta is **conditional on v**. An independent fixed Gaussian prior on beta
does not give this normal-inverse-gamma conjugate posterior. This conditional
parameterization agrees with [MathWorks' conjugate regression model](https://www.mathworks.com/help/econ/conjugateblm.html).

The posterior parameters are

\[
V_n=(V_0^{-1}+X^\top X)^{-1},\qquad
m_n=V_n(V_0^{-1}m_0+X^\top y),\qquad a_n=a_0+n/2,
\]
\[
b_n=b_0+\tfrac12\{y^\top y+m_0^\top V_0^{-1}m_0
-m_n^\top V_n^{-1}m_n\}.
\]

The implementation evaluates the equivalent, numerically stable expression
`bn = b0 + (||y-X mn||^2 + (mn-m0).T V0^-1 (mn-m0))/2`.

The exact group evidence is

\[
\boxed{\log Z_g=-\frac n2\log(2\pi)
+\frac12(\log|V_n|-\log|V_0|)
+a_0\log b_0-a_n\log b_n
+\log\Gamma(a_n)-\log\Gamma(a_0).}
\]

Independent group priors **and a likelihood that factorizes across groups** give
`log Z = sum_g log Zg`. At the default width five, a 10,000-dimensional target
has exactly 2,000 groups, each with four coefficients and one variance.

The posterior coefficient marginal is multivariate Student-t with `2 an`
degrees of freedom, mean `mn`, and scale matrix `(bn/an) Vn`. The scale matrix
is not the covariance: when `an > 1`, covariance is `bn/(an-1) Vn`.
The variance posterior is `IG(an,bn)`. Coefficients and variance are dependent
through the conditional coefficient scale even when their linear covariance
is zero. This is relevant to interpreting what a pair Morph approximation
can retain from the five-dimensional group.

## Exact posterior training draws

For every draw and group, independently generate

\[
v=b_n/G,\quad G\sim\operatorname{Gamma}(a_n,\text{scale}=1),
\qquad \beta=m_n+\sqrt v\,L_n z,
\]

where `z ~ N(0,I)` independently and `Ln Ln.T = Vn`. These are direct joint
posterior samples, not prior draws, optimized parameter estimates, or MCMC
samples. All draws have equal weight and posterior IAT=1.

Morph fitting and bridge posterior batches are disjoint. The target generator
shuffles parameter columns once, while preserving the mapping between
coefficients, variance, predictors, and observations. Ground-truth groups enter
only diagnostics; Morph still discovers its own pairs.

`log_unnormalized` evaluates the actual `log_prior + log_likelihood` directly.
It does not define the target by adding log Z to the posterior. The runner
checks independently that `log_prior + log_likelihood - log_posterior` equals
the analytical evidence to 1e-8 on held-out samples.

## Prepare the large cases yourself

```bash
python -m benchmarks.highdim_prepare \
  --family regression --dimensions 3000 5000 10000 --group-orders 5 \
  --variance-coordinate variance --observations-range 8 24 \
  --samples 2000 --chunk-size 256 \
  --target-seed 20261006 --sample-seed 20261007 \
  --output benchmark-results/regression-data
```

Use your benchmark-enabled Python environment. This checkout's existing WSL
Python is `/home/apokrypha/.local/share/max-jax/.venv/bin/python`. This preparation
command imports no JAX and runs no Morph.

The default uses five-parameter groups. To vary dependency block sizes, use
`--group-orders 1 2 3 4 5`: each group has `order-1` coefficients and one variance.
The priors, predictor correlations, number of observations, and realized data
vary randomly between groups and are then fixed and saved. Smaller observation
counts produce heavier posterior tails. There is no free `--log-evidence` value
for regression; evidence is determined by the model and its observed data.

Each directory stores `target.json` with every `Xg`, `yg`, prior hyperparameter,
posterior parameter, shuffled index mapping, and calculated evidence; memory-mapped
`posterior_samples.npy` and `posterior_log_density.npy`; and a completion manifest.
The cache is actual `log likelihood + log prior` at each exact posterior draw.

```python
import numpy as np
from benchmarks.highdim_target import load_target

target = load_target("benchmark-results/regression-data/10000d/target.json")
posterior = target.sample(np.random.default_rng(42), 2000)
training, bridge_posterior = posterior[:1000], posterior[1000:]
exact_log_z = target.log_evidence
cached_log_f = target.log_unnormalized(posterior)
```

Raw variances have positive support. Gaussian KDE proposals with a negative
variance correctly receive `-inf` target density; they are retained in bridge
estimation. `--variance-coordinate log_variance` is an optional target-coordinate
choice. It includes the `sum log(vg)` Jacobian in the transformed prior and
posterior, leaves likelihood values unchanged, and preserves the same evidence.
No KDE or Morph search is modified by either choice. The runner reports support
leakage and records a failed estimate if all proposals miss the support.

The existing runner accepts prepared regression data:

```bash
bash scripts/run-highdim.sh gpu \
  --dataset benchmark-results/regression-data/3000d \
  --morph-order 2 \
  --output benchmark-results/regression-3000
```

Choose your own native TC, SGM, bandwidth, and batching options. All standard
Morph defaults remain unchanged. The larger cases have not been fitted here.

## Mathematical and workflow validation

Tests compare every group evidence to the independently implemented
observation-space Student-t density:
`y ~ t_(2a0)(X m0, (b0/a0)(I + X V0 X.T))`.
They also check the pointwise Bayes identity, saved-model reconstruction,
positive support, and log-variance Jacobians. A 30,000-draw sampler check compares
variance CDFs to inverse gamma, coefficient CDFs to Student-t, and conditional
whitened coefficients to standard normals. The fixed-noise Gaussian formula is
checked separately against SciPy's multivariate normal density.

The initial 200D/500D smoke checks used 256 exact fitting draws, 256 independent posterior
draws, 256 proposals, 32 scoring rows, and four native SGM seeds. These small
settings verify the workflow and do not establish reliable high-D evidence.
The exact-proposal bridge control must recover the analytically computed log Z.
Actual pair-Morph convergence, overlap, KL, support leakage, and evidence errors
are saved and reported separately.

The larger reruns use `--tc-samples 100 --fit-samples 2000
--posterior-samples 20000`, with `--proposals 20000`, `50000`, or `100000`.
Every final factor uses all 2,000 fitting rows. Prepared datasets need at least
22,000 rows for the independent fitting and held-out posterior sets.

Runtime fields separate `morph_seconds` (native search plus final KDE fitting)
from `bridge_sampling_total_seconds` (held-out posterior proposal-density
evaluation, Morph proposal generation, proposal-density and target-density
evaluation, and bridge iteration). The latter sums measured component times.
`bridge_seconds` is only the final bridge iteration, with densities already
computed. `workflow_seconds` also includes diagnostics and bookkeeping.
Target generation, analytical controls and plots are outside these timings;
first-call compilation and synchronization are included.

Add `--plots-only` to save the three corner comparisons, evidence plot and `summary.txt`,
and print an evidence/runtime summary. JSON files are suppressed; progress
reports remain enabled. The evidence plot shows the dimension count and
total case computation time, including target setup and analytical controls,
but excluding plot rendering.

The benchmark now defaults to the top five ranked native SGM seeds, capped
by the number of available candidates. `--n-seeds` explicitly overrides this
budget, including smaller counts. There is no square-root rule or automatic
minimum tied to the candidate count.
Historical reports retain their recorded seed budgets. The seed policy
is a benchmark configuration choice; the Morph implementation is unchanged.

Bridge sampling now defaults to stopping when
`abs(logZ_new - logZ_previous) <= 0.1`, with no iteration cap. `--rtol` sets this
absolute delta-logZ tolerance; its default is 0.1. Use `--max-iterations 10000`
to impose an explicit cap. Converged cases still stop early. Historical reports
retain the tolerance and limits used for their runs.

The public `bridge_sampling` and `estimate_evidence` APIs also default to
`rtol=0.1, max_iterations=None`. Uncapped runs continue compiled iteration
chunks and retain the complete history and global iteration count. Chunk
boundaries do not terminate the solver. Nonfinite iterates retain the existing
failure behavior.

Geometric-mean restarts are now enabled by default after every 1,000
unconverged bridge updates. The new starting log-Z is the midpoint of the last
two values. Set `--bridge-restart-interval N` (at least 2) to change the cycle
length or `--no-geometric-restart` to restore standard iteration. The public
APIs expose `restart_interval=1000` and `geometric_restart=True`. The total
`--max-iterations` cap is unchanged, so a cap no larger than the restart interval
permits no restart. This repeats the remedy published by
[Gronau, Heathcote and Matzke (2020)](https://doi.org/10.3758/s13428-019-01290-6).

The reported log evidence is now the arithmetic mean of the last five computed
log-Z iterates, excluding initialization and values before the latest restart.
If fewer than five updates are available since that restart, only those updates
are averaged. This averages log-Z values, not
Z values. Set `--bridge-average-iterations N` or pass `average_iterations=N`
to `bridge_sampling` or `estimate_evidence`; one returns the final iterate.

The convergence test uses the ordinary bridge update from its input, including
the new seed after a restart; the midpoint jump cannot itself trigger convergence.
History retains all bridge updates and does not insert restart seeds.
`BridgeResult.final_log_evidence` exposes the last actual iterate, and benchmark
metrics also record `final_iteration_log_evidence` and
`log_evidence_averaging_iterations`. Overlap and conditional error diagnostics
are evaluated at the reported average. Averaging does not change a failed
convergence status, and numerical failures are not averaged away.

The runner uses tqdm progress bars for candidate scoring, SGM seeds, proposal
batches (`trange`), and bridge iterations. The bridge bar shows current log
evidence, its absolute change, convergence status, and the reported averaged
estimate. Updates reflect completed batches/iterations during computation.
The first JAX compilation can precede the first computed update. Uncapped
bridge runs show an iteration counter without an estimated total, then finish
at the actual stopping iteration.

`--progress-reports 20` is the default: roughly 20 completion notifications per
compiled stage; tqdm controls display refresh frequency. Set a different count
to change callback frequency or use `--progress-reports 0` to disable the bars.
Candidate and SGM notifications occur after completed batches, so counts may
step past the nominal interval. Proposal bars advance once per completed batch.
Uncapped bridge runs update every 500 iterations by default;
the first iteration and actual stopping iteration are also shown. With an
explicit cap, the periodic interval is based on that cap and progress-reports.
Uncapped verbosity uses 10,000 iterations as a reporting reference, not a limit.
The numerical algorithms and termination rules are unchanged.

Direct API calls stay silent by default and retain text reports when enabled with
`fit(..., candidate_progress_every=6250, seed_progress_every=18)` or
`bridge_sampling(..., max_iterations=10000, progress_every=500)`.

Use `--proposal-counts 5000 10000` to compare two proposal budgets while sharing
one native Morph fit. Every case still uses a separate proposal batch and the
same held-out posterior draws. The second case records `fit_reused: true`,
zero additional `morph_seconds`, and the shared fit's `original_fit_seconds`.
The cache checks training data, search seed, and all fitting/search options.
It exists only in the running process; resuming an incomplete sweep may require
fitting again. Case directories include the proposal count to keep results
separate.

## Controlled cross-group follow-up

`--cross-group-coupling 0.8` pairs neighboring eligible groups and shears their
first coefficient: physical `u = beta_a + 0.8 beta_b`, `w = beta_b`. Variance
coordinates and the other coefficients are unchanged. Prior and likelihood
are evaluated using the inverse `beta_a = u - 0.8 w`. For example, observations
in group a then depend on coefficients in both physical groups. The transformed
conditional prior also contains cross-group coefficient covariance.

The transformation determinant is one, so the integral of actual prior times
likelihood is unchanged: the original sum of analytical log evidences remains
exact. Sampling the conjugate posterior and applying the same forward shear
gives direct exact posterior draws. This is a controlled reparameterization
of the same regression model, with dependence across groups in the tested
coordinates. The saved target records the coupling strength.

At the default base group size five, nonzero coupling creates dependent
components of size ten, intentionally relaxing the five-dimensional block
limit for this follow-up. Diagnostics use these larger components; the three
default 10-parameter corner plots keep coupled components together and choose
those with the strongest mean absolute posterior correlation within each configured level.
Morph still uses its
unchanged pair search. Zero coupling preserves the primary independent-group
benchmark and its block orders up to five.

With zero coupling, the default 10-parameter panels combine two independent
five-parameter groups, ranked by their within-group correlation. Selecting a
panel does not create correlation between these groups. Use `--corner-count 3
--corner-dimensions 10` to set the plot sizes explicitly. Each plot overlays
exact posterior draws (black) and Morph draws (blue), labels the original
parameter indices, and shows its measured maximum and mean absolute Pearson
correlation. Selected indices and correlation summaries are saved in
`metrics.json` under `corner_plots`. The evidence history is also retained.

For general new cross-group models, a simple sum of group evidences requires
factorization or another exact integration argument. A correlated Gaussian
inverse problem retains the observation-space Gaussian formula. A fixed finite
mixture has exact evidence `logsumexp(log mixture_weight + component_log_Z)`;
posterior component probabilities must be proportional to weight times Z.
