"""Weighted Gaussian KDEs with standard Silverman bandwidths.

These kernels implement the same covariance and bandwidth convention as
``jax.scipy.stats.gaussian_kde(..., bw_method='silverman')``. Evaluation uses
cached whitened centers and log-sum-exp, with bounded query batches.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

from ._validation import positive_integer, samples_and_weights


class KDEState(NamedTuple):
    centers: jax.Array
    whitened_centers: jax.Array
    location: jax.Array
    cholesky: jax.Array
    log_weights: jax.Array
    log_normalizer: jax.Array
    bandwidth: jax.Array
    effective_sample_size: jax.Array


@jax.jit
def _fit_kde(samples, weights, bandwidth_scale, regularization):
    """Unchecked numerical kernel; weights must already sum to one."""
    dtype = jnp.result_type(samples, weights)
    samples = samples.astype(dtype)
    weights = weights.astype(dtype)
    dimension = samples.shape[1]
    offsets = samples - samples[0]
    mean_offset = weights @ offsets
    location = samples[0] + mean_offset
    centered = offsets - mean_offset
    sum_squared_weights = jnp.sum(weights**2)
    effective_size = 1.0 / sum_squared_weights
    covariance = (centered.T * weights) @ centered / (1.0 - sum_squared_weights)
    # An opt-in diagonal ridge retains the original parameter units.
    covariance += regularization * jnp.diag(jnp.diag(covariance))
    deviations = jnp.sqrt(jnp.diag(covariance))
    correlation = covariance / (deviations[:, None] * deviations[None, :])
    nonsingular = jnp.linalg.eigvalsh(correlation)[0] > jnp.finfo(dtype).eps * dimension
    bandwidth = bandwidth_scale * (effective_size * (dimension + 2) / 4.0) ** (
        -1.0 / (dimension + 4)
    )
    cholesky = jnp.linalg.cholesky(covariance) * bandwidth
    cholesky = jnp.where(nonsingular, cholesky, jnp.nan)
    whitened = solve_triangular(cholesky, centered.T, lower=True).T
    log_normalizer = 0.5 * dimension * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(jnp.diag(cholesky)))
    return KDEState(
        samples,
        whitened,
        location,
        cholesky,
        jnp.log(weights),
        log_normalizer,
        bandwidth,
        effective_size,
    )


def fit_kde(samples, weights=None, *, bandwidth_scale=1.0, regularization=0.0):
    """Fit a KDE to rows of samples. Fitting validation runs on the host.

    ``bandwidth_scale=1`` is Silverman's rule. ``regularization=0`` preserves
    the standard KDE exactly and rejects singular covariance matrices.
    """
    samples, weights = samples_and_weights(samples, weights)
    if not np.isfinite(bandwidth_scale) or bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if np.count_nonzero(weights) <= samples.shape[1]:
        raise ValueError("a KDE needs more positive-weight samples than dimensions")
    state = _fit_kde(jnp.asarray(samples), jnp.asarray(weights), bandwidth_scale, regularization)
    if not np.isfinite(np.asarray(state.cholesky)).all():
        raise ValueError(
            "KDE covariance is singular; remove constant dimensions or explicitly "
            "set regularization > 0 for collinear dimensions"
        )
    return state


def _log_prob_batch(state, points):
    whitened = solve_triangular(state.cholesky, (points - state.location).T, lower=True).T
    difference = whitened[:, None, :] - state.whitened_centers[None, :, :]
    log_kernels = -0.5 * jnp.sum(difference**2, axis=-1)
    return logsumexp(log_kernels + state.log_weights, axis=-1) - state.log_normalizer


@partial(jax.jit, static_argnames=("chunk_size",))
def kde_log_prob(state, points, *, chunk_size=128):
    """Evaluate one point ``(d,)`` or a batch ``(n, d)`` in log space.

    Differentiable in evaluation points and numerical KDE state. ``chunk_size``
    is static under JIT; peak kernel storage is O(chunk_size * n_centers * d).
    """
    positive_integer(chunk_size, "chunk_size")
    points = jnp.asarray(points)
    single = points.ndim == 1
    if points.ndim not in (1, 2) or points.shape[-1] != state.centers.shape[1]:
        raise ValueError("points must have shape (d,) or (n, d)")
    points = jnp.atleast_2d(points)
    count = points.shape[0]
    if count == 0:
        return jnp.empty((0,), dtype=state.centers.dtype)
    size = min(chunk_size, count)
    padding = (-count) % size
    batches = jnp.pad(points, ((0, padding), (0, 0))).reshape(-1, size, points.shape[1])
    values = jax.lax.map(lambda batch: _log_prob_batch(state, batch), batches)
    values = values.reshape(-1)[:count]
    return values[0] if single else values


@partial(jax.jit, static_argnames=("n",))
def sample_kde(key, state, n):
    """Draw from the *same* normalized Gaussian mixture used by log_prob."""
    positive_integer(n, "n")
    component_key, noise_key = jax.random.split(key)
    cumulative_weights = jnp.cumsum(jnp.exp(state.log_weights)).at[-1].set(1.0)
    uniforms = jax.random.uniform(component_key, (n,), dtype=state.centers.dtype)
    indices = jnp.searchsorted(cumulative_weights, uniforms, side="right")
    noise = jax.random.normal(noise_key, (n, state.centers.shape[1]), dtype=state.centers.dtype)
    return state.centers[indices] + noise @ state.cholesky.T
