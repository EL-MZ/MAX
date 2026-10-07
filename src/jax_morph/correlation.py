"""KDE plug-in estimates of block total correlation (paper Eq. 6)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._progress import progress_map, report_count, validate_progress_every
from ._validation import positive_integer, samples_and_weights
from .kde import _fit_kde, _validate_kde_options, kde_log_prob


class CorrelationResult(NamedTuple):
    raw_scores: jax.Array
    scores: jax.Array
    marginal_entropies: jax.Array
    sample_count: int


@partial(jax.jit, static_argnames=("chunk_size", "candidate_batch_size", "progress_every"))
def _correlation_kernel(
    samples,
    weights,
    blocks,
    bandwidth_scale,
    regularization,
    *,
    chunk_size,
    candidate_batch_size,
    progress_every=0,
):
    def marginal_entropy(index):
        values = samples[:, index, None]
        state = _fit_kde(values, weights, bandwidth_scale, regularization)
        return -(weights @ kde_log_prob(state, values, chunk_size=chunk_size))

    # Reduce immediately: retain D entropies, not an N-by-D log-density matrix.
    entropies = jax.lax.map(marginal_entropy, jnp.arange(samples.shape[1]))

    def total_correlation(block):
        values = samples[:, block]
        state = _fit_kde(values, weights, bandwidth_scale, regularization)
        joint_logs = kde_log_prob(state, values, chunk_size=chunk_size)
        return weights @ joint_logs + jnp.sum(entropies[block])

    raw_scores = progress_map(
        total_correlation,
        blocks,
        batch_size=min(candidate_batch_size, blocks.shape[0]),
        progress_every=progress_every,
        label="candidate scores",
    )
    return raw_scores, entropies


@jax.jit
def _valid_blocks(blocks, dimension):
    return (
        jnp.all(blocks >= 0)
        & jnp.all(blocks < dimension)
        & jnp.all(jnp.diff(jnp.sort(blocks, axis=1), axis=1) != 0)
    )


@jax.jit
def _finite_correlations(raw, entropies):
    return jnp.all(jnp.isfinite(raw)) & jnp.all(jnp.isfinite(entropies))


def estimate_total_correlations(
    samples,
    blocks,
    weights=None,
    *,
    bandwidth_scale=1.0,
    regularization=0.0,
    max_samples=None,
    key=None,
    chunk_size=128,
    candidate_batch_size=1,
    progress_every=0,
):
    """Estimate E[log KDE(block) - sum(log KDE(singleton))] on fit data.

    Small negative plug-in estimates are clipped to zero for SGM; raw values
    are also returned. Optional ``max_samples`` thins only this scoring stage,
    using a uniform subset without replacement, then renormalizes retained
    weights. The final Morph factors are still fitted on all fitting data.
    """
    samples, weights = samples_and_weights(samples, weights)
    positive_integer(chunk_size, "chunk_size")
    positive_integer(candidate_batch_size, "candidate_batch_size")
    validate_progress_every(progress_every)
    blocks = jnp.asarray(blocks)
    if (
        blocks.ndim != 2
        or min(blocks.shape) < 1
        or not np.issubdtype(blocks.dtype, np.integer)
        or not bool(_valid_blocks(blocks, samples.shape[1]))
    ):
        raise ValueError("blocks must contain distinct, valid dimension indices")
    if max_samples is not None:
        positive_integer(max_samples, "max_samples")
        if max_samples < len(samples):
            if key is None:
                raise ValueError("a PRNG key is required when thinning TC samples")
            # Uniform subset, retaining original weights, avoids double-weighting.
            indices = np.asarray(jax.random.permutation(key, len(samples)))[:max_samples]
            samples, weights = samples_and_weights(samples[indices], weights[indices])
    # The compiled marginal fits validate every dimension, including unused ones.
    _validate_kde_options(bandwidth_scale, regularization)
    if np.count_nonzero(weights) <= blocks.shape[1]:
        raise ValueError("TC scoring needs more positive-weight samples than block dimensions")
    if progress_every:
        report_count("candidate scores", 0, len(blocks))
    raw, entropies = _correlation_kernel(
        jnp.asarray(samples),
        jnp.asarray(weights),
        jnp.asarray(blocks),
        bandwidth_scale,
        regularization,
        chunk_size=chunk_size,
        candidate_batch_size=candidate_batch_size,
        progress_every=progress_every,
    )
    if not bool(_finite_correlations(raw, entropies)):
        raise ValueError("nonfinite TC scores: check block covariance or set regularization > 0")
    return CorrelationResult(raw, jnp.maximum(raw, 0), entropies, len(samples))
