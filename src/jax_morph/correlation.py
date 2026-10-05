"""KDE plug-in estimates of block total correlation (paper Eq. 6)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._validation import positive_integer, samples_and_weights
from .kde import _fit_kde, fit_kde, kde_log_prob


class CorrelationResult(NamedTuple):
    raw_scores: jax.Array
    scores: jax.Array
    marginal_entropies: jax.Array
    sample_count: int


@partial(jax.jit, static_argnames=("chunk_size", "candidate_batch_size"))
def _correlation_kernel(
    samples,
    weights,
    blocks,
    bandwidth_scale,
    regularization,
    *,
    chunk_size,
    candidate_batch_size,
):
    def marginal_log_density(index):
        values = samples[:, index, None]
        state = _fit_kde(values, weights, bandwidth_scale, regularization)
        return kde_log_prob(state, values, chunk_size=chunk_size)

    # Each singleton KDE is evaluated once, not once per candidate block.
    marginal_logs = jax.lax.map(marginal_log_density, jnp.arange(samples.shape[1])).T

    def total_correlation(block):
        values = samples[:, block]
        state = _fit_kde(values, weights, bandwidth_scale, regularization)
        joint_logs = kde_log_prob(state, values, chunk_size=chunk_size)
        return weights @ (joint_logs - jnp.sum(marginal_logs[:, block], axis=1))

    raw_scores = jax.lax.map(
        total_correlation,
        blocks,
        batch_size=min(candidate_batch_size, blocks.shape[0]),
    )
    return raw_scores, -(weights @ marginal_logs)


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
    blocks = np.asarray(blocks)
    if (
        blocks.ndim != 2
        or min(blocks.shape) < 1
        or not np.issubdtype(blocks.dtype, np.integer)
        or np.any(blocks < 0)
        or np.any(blocks >= samples.shape[1])
        or np.any(np.diff(np.sort(blocks, axis=1), axis=1) == 0)
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
    # Validate bandwidth/ridge and marginal variance before entering the core.
    for index in range(samples.shape[1]):
        fit_kde(
            samples[:, index, None],
            weights,
            bandwidth_scale=bandwidth_scale,
            regularization=regularization,
        )
    if np.count_nonzero(weights) <= blocks.shape[1]:
        raise ValueError("TC scoring needs more positive-weight samples than block dimensions")
    raw, entropies = _correlation_kernel(
        jnp.asarray(samples),
        jnp.asarray(weights),
        jnp.asarray(blocks),
        bandwidth_scale,
        regularization,
        chunk_size=chunk_size,
        candidate_batch_size=candidate_batch_size,
    )
    if not np.isfinite(np.asarray(raw)).all():
        raise ValueError("nonfinite TC scores: check block covariance or set regularization > 0")
    return CorrelationResult(raw, jnp.maximum(raw, 0), entropies, len(samples))
