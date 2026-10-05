"""Normalized product of disjoint low-dimensional Gaussian KDE factors."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._validation import positive_integer, samples_and_weights
from .correlation import estimate_total_correlations
from .kde import KDEState, fit_kde, kde_log_prob, sample_kde
from .sgm import candidate_blocks, seeded_greedy_maximization


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MorphState:
    """Immutable PyTree; block structure is static, KDE parameters are leaves."""

    blocks: tuple[tuple[int, ...], ...]
    factors: tuple[KDEState, ...]
    dimension: int

    def tree_flatten(self):
        return self.factors, (self.blocks, self.dimension)

    @classmethod
    def tree_unflatten(cls, metadata, factors):
        blocks, dimension = metadata
        return cls(blocks, tuple(factors), dimension)


class FitDiagnostics(NamedTuple):
    candidates: jax.Array
    raw_total_correlations: jax.Array
    total_correlations: jax.Array
    selected_candidate_indices: jax.Array
    retained_total_correlation: jax.Array
    seed_totals: jax.Array
    tc_sample_count: int
    fit_sample_count: int


class MorphFit(NamedTuple):
    state: MorphState
    diagnostics: FitDiagnostics


def fit(
    samples,
    weights=None,
    *,
    order=2,
    n_seeds=None,
    bandwidth_scale=1.0,
    regularization=0.0,
    max_tc_samples=None,
    key=None,
    chunk_size=128,
    candidate_batch_size=1,
    seed_batch_size=16,
    max_candidates=100_000,
):
    """Fit a Morph approximation; return numerical state and selection trace.

    Order L selects floor(d/L) length-L blocks; every remaining dimension gets
    its own singleton KDE, exactly as in Eq. 5. Orders >= 10 are rejected to
    keep factors low dimensional. Use order=1 for a product of marginals.
    Fitting/selection is host-orchestrated; evaluation and sampling are jittable.
    """
    samples, weights = samples_and_weights(samples, weights)
    dimension = samples.shape[1]
    order = positive_integer(order, "order")
    if order > dimension or order >= 10:
        raise ValueError("order must satisfy 1 <= order <= dimension and order < 10")
    candidates = candidate_blocks(dimension, order, max_candidates=max_candidates)
    if order == 1:
        blocks = tuple((index,) for index in range(dimension))
        raw_scores = scores = jnp.zeros((dimension,))
        selected = jnp.arange(dimension)
        total = jnp.asarray(0.0)
        seed_totals = jnp.empty((0,))
        tc_sample_count = 0
    else:
        correlations = estimate_total_correlations(
            samples,
            candidates,
            weights,
            bandwidth_scale=bandwidth_scale,
            regularization=regularization,
            max_samples=max_tc_samples,
            key=key,
            chunk_size=chunk_size,
            candidate_batch_size=candidate_batch_size,
        )
        selection = seeded_greedy_maximization(
            candidates,
            correlations.scores,
            dimension=dimension,
            n_seeds=n_seeds,
            seed_batch_size=seed_batch_size,
        )
        chosen = tuple(tuple(map(int, block)) for block in np.asarray(selection.blocks))
        used = {index for block in chosen for index in block}
        singletons = tuple((index,) for index in range(dimension) if index not in used)
        blocks = chosen + singletons
        raw_scores, scores = correlations.raw_scores, correlations.scores
        selected, total, seed_totals = (
            selection.candidate_indices,
            selection.total_score,
            selection.seed_totals,
        )
        tc_sample_count = correlations.sample_count
    factors = tuple(
        fit_kde(
            samples[:, block],
            weights,
            bandwidth_scale=bandwidth_scale,
            regularization=regularization,
        )
        for block in blocks
    )
    state = MorphState(blocks, factors, dimension)
    diagnostics = FitDiagnostics(
        candidates,
        raw_scores,
        scores,
        selected,
        total,
        seed_totals,
        tc_sample_count,
        len(samples),
    )
    return MorphFit(state, diagnostics)


@partial(jax.jit, static_argnames=("chunk_size",))
def log_prob(state, points, *, chunk_size=128):
    """Sum factor log densities for a point (d,) or rows of points (n, d)."""
    points = jnp.asarray(points)
    if points.ndim not in (1, 2) or points.shape[-1] != state.dimension:
        raise ValueError("points must have shape (d,) or (n, d)")
    result = jnp.zeros(points.shape[:-1], dtype=state.factors[0].centers.dtype)
    for block, factor in zip(state.blocks, state.factors):
        result += kde_log_prob(factor, points[..., jnp.asarray(block)], chunk_size=chunk_size)
    return result


@partial(jax.jit, static_argnames=("n",))
def sample(key, state, n):
    """Independent joint proposal draws, using a separate key per factor."""
    positive_integer(n, "n")
    keys = jax.random.split(key, len(state.factors))
    result = jnp.zeros((n, state.dimension), dtype=state.factors[0].centers.dtype)
    for block, factor, factor_key in zip(state.blocks, state.factors, keys):
        result = result.at[:, jnp.asarray(block)].set(sample_kde(factor_key, factor, n))
    return result
