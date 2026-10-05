# Implementation record

## Development inputs

The earlier project discussion was recovered through a personal-context summary;
the full chat reader was unavailable. Its recovered decisions were: start with
standard Silverman KDEs, keep factors low dimensional, preserve weights, and
expose fitting, sampling, and log-density operations usable with JAX transforms.
This implementation follows those decisions. No flow or alternative bandwidth
model is substituted. The source paper version is recorded in `method.md`.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `_validation.py` | Host-side shape, weight, and option checks |
| `kde.py` | Weighted Silverman fit, normalized sampling, chunked log density |
| `correlation.py` | KDE total-correlation scores, raw scores, optional subset |
| `sgm.py` | Complete candidates, Algorithm 1, JAX selection kernel |
| `morph.py` | Disjoint partition, singleton factors, PyTree state, fit trace |
| `bridge.py` | Log-space iteration, convergence trace, overlap and error diagnostics |
| `evidence.py` | Consecutive sample split, target evaluation, complete MorphZ workflow |

Fitting orchestration validates on the host. `kde_log_prob`, `sample_kde`,
`log_prob`, `sample`, `sgm_kernel`, and `bridge_kernel` run in JAX. Discrete block
selection is not differentiated. Density evaluation is differentiable in query
points and continuous state leaves; mixture-index draws are discrete and are
not fully pathwise differentiable. The bridge while-loop is not advertised as
reverse-mode differentiable.

## Efficiency changes that preserve the baseline

1. Singleton density evaluations are reused for every candidate block.
2. Kernel centers are whitened once per fitted factor. Query batches bound the
   temporary pairwise kernel array; log-sum-exp preserves far-tail values.
3. KDE resampling uses cumulative mixture weights and inverse-CDF lookup. It
   avoids allocating one categorical Gumbel score per (draw, center) pair.
4. SGM sorts once and terminates each seed scan when its partition is full.
   Bounded seed batches avoid a seeds-by-candidates conflict matrix. This has
   exactly the same selected blocks and totals as the full-scan algorithm under
   the documented tie policy; tests compare to a direct Python reference.
5. Target log densities and proposal log densities are computed once per draw;
   bridge iterations reuse their ratios. Cached posterior targets avoid those
   additional target calls entirely.

`benchmarks/sgm.py` compares warm, synchronized JAX executions of early stopping
and full scanning, checks identical outputs, and records device/version metadata.
Compilation and first execution are reported separately. The benchmark measures
SGM only; it does not establish a GPU or end-to-end MorphZ speedup.

## Explicit changes or extensions

- Weighted KDE and TC support extends the paper's equal-weight workflow.
- Optional `max_tc_samples` uses a uniform subset without replacement, retains
  and renormalizes its input weights, and may alter the scores and partition.
  With weights, a small subset can be unrepresentative; no automatic subset is
  used. Final factors always use the full fitting batch.
- Finite-sample negative TC estimates are clipped for selection, while retained
  raw scores make this choice inspectable.
- Covariance regularization and bandwidth scaling are explicit options; the
  default is unregularized Silverman KDE.
- IAT uses adjacent positive-lag autocorrelation pairs, truncated at the first
  nonpositive pair, instead of an autoregressive spectral fit. Values are clipped
  to [1, N]. A caller can supply an externally estimated IAT.

## Diagnostics and limitations

`MorphFit.diagnostics` contains candidates, raw and clipped scores, selected
candidate indices, retained score, all seed totals, and TC/fit sample counts.
`MorphState.blocks` includes residual singletons. `EvidenceResult` records fit
and bridge indices, proposal draws, and target evaluation counts. `BridgeResult`
contains log Z, relative MC error, IAT, bounded mean bridge overlap terms,
importance-weight ESS, iterations, convergence flag, and padded log-Z history.
The importance ESS is a proposal diagnostic, not a bridge posterior ESS.

The package requires adequate posterior coverage and proper priors. It does not
sample the original posterior, repair missing modes, perform MSS, implement exact
Blossom/ILP optimization, or normalize rejected/truncated KDE proposals. Out-of-
support target evaluations remain -inf and their proposal draws are retained.
Periodic coordinates need an appropriate representation supplied by the caller.
Constant dimensions must be removed; singular blocks need an explicit ridge.
Singularity is checked on the correlation matrix before Cholesky, so numerical
rounding cannot make an exactly collinear block appear valid. Numerically singular
blocks (minimum correlation eigenvalue <= machine epsilon times block size) are
rejected as well. Continuous KDE state arrays use a common promoted dtype.
Candidate enumeration scales as choose(d,L), while full KDE TC scoring costs
roughly O(choose(d,L)*n^2*L). JAX does not eliminate that combinatorial cost.
Order is limited to 1..9, with default 2. The all-seed default can be expensive;
smaller K changes the search budget, not the candidate collection.

For ordered MCMC, use the contiguous held-out batch and inspect chain convergence.
Do not flatten multiple chains and rely on automatic IAT. The equal-weight
evidence API is separate from weighted density fitting; weighted bridge sampling
is not implemented. Tests cover CPU JAX; accelerator performance must be measured
on the intended device.
