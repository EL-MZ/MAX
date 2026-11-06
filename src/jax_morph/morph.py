"""Normalized product of disjoint low-dimensional Gaussian KDE factors."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._validation import positive_integer, samples_and_weights
from .correlation import estimate_total_correlations
from .kde import KDEState, _fit_kde, _validate_kde_options, kde_log_prob, sample_kde
from .sgm import _select_validated, candidate_blocks


class _FactorGroup(NamedTuple):
    indices: jax.Array
    factors: KDEState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MorphState:
    """Grouped KDE PyTree with dynamic block indices.

    The constructor and eager ``blocks``/``factors`` views retain the original
    API. Only group shapes, factor ordering and dimension are static under JIT.
    Inspect ``blocks`` outside JIT; numerical kernels use the array indices.
    """

    _groups: tuple[_FactorGroup, ...]
    dimension: int
    _layout: tuple[tuple[int, ...], ...]

    def __init__(self, blocks, factors, dimension):
        groups = {}
        for position, (block, factor) in enumerate(zip(blocks, factors)):
            signature = tuple((leaf.shape, leaf.dtype) for leaf in factor)
            groups.setdefault(signature, []).append((position, block, factor))
        packed, layout = [], []
        for members in groups.values():
            positions, indices, states = zip(*members)
            packed.append(
                _FactorGroup(
                    jnp.asarray(indices, dtype=jnp.int32),
                    jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states),
                )
            )
            layout.append(positions)
        object.__setattr__(self, "_groups", tuple(packed))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_layout", tuple(layout))

    @property
    def blocks(self):
        blocks = [None] * sum(map(len, self._layout))
        for group, positions in zip(self._groups, self._layout):
            for position, block in zip(positions, np.asarray(group.indices)):
                blocks[position] = tuple(map(int, block))
        return tuple(blocks)

    @property
    def factors(self):
        factors = [None] * sum(map(len, self._layout))
        for group, positions in zip(self._groups, self._layout):
            for index, position in enumerate(positions):
                factors[position] = jax.tree_util.tree_map(lambda x: x[index], group.factors)
        return tuple(factors)

    def tree_flatten(self):
        return self._groups, (self.dimension, self._layout)

    @classmethod
    def tree_unflatten(cls, metadata, groups):
        dimension, layout = metadata
        state = object.__new__(cls)
        object.__setattr__(state, "_groups", tuple(groups))
        object.__setattr__(state, "dimension", dimension)
        object.__setattr__(state, "_layout", layout)
        return state


@jax.jit
def _fit_factor_group(samples, weights, indices, bandwidth_scale, regularization):
    return jax.lax.map(
        lambda block: _fit_kde(samples[:, block], weights, bandwidth_scale, regularization),
        indices,
        batch_size=min(16, len(indices)),
    )


@jax.jit
def _valid_factor_groups(groups):
    valid = jnp.asarray(True)
    for group in groups:
        valid = valid & jnp.all(jnp.isfinite(group.factors.cholesky))
    return valid


def _fit_factors(samples, weights, blocks, bandwidth_scale, regularization):
    """Fit bounded batches, transferring only one aggregate validity flag."""
    _validate_kde_options(bandwidth_scale, regularization)
    if np.count_nonzero(weights) <= max(map(len, blocks)):
        raise ValueError("a KDE needs more positive-weight samples than dimensions")
    by_width = {}
    for position, block in enumerate(blocks):
        by_width.setdefault(len(block), []).append((position, block))
    samples, weights = jnp.asarray(samples), jnp.asarray(weights)
    groups, layout = [], []
    for members in by_width.values():
        positions, indices = zip(*members)
        indices = jnp.asarray(indices, dtype=jnp.int32)
        groups.append(
            _FactorGroup(
                indices,
                _fit_factor_group(
                    samples,
                    weights,
                    indices,
                    bandwidth_scale,
                    regularization,
                ),
            )
        )
        layout.append(positions)
    if not bool(_valid_factor_groups(tuple(groups))):
        raise ValueError(
            "KDE covariance is singular; remove constant dimensions or explicitly "
            "set regularization > 0 for collinear dimensions"
        )
    return MorphState.tree_unflatten((samples.shape[1], tuple(layout)), groups)


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
    n_seeds=5,
    bandwidth_scale=1.0,
    regularization=0.0,
    max_tc_samples=None,
    key=None,
    chunk_size=128,
    candidate_batch_size=1,
    seed_batch_size=16,
    sgm_early_stop=False,
    candidate_progress_every=0,
    seed_progress_every=0,
):
    """Fit a Morph approximation; return numerical state and selection trace.

    Order L selects floor(d/L) length-L blocks; every remaining dimension gets
    its own singleton KDE, exactly as in Eq. 5. Orders >= 10 are rejected to
    keep factors low dimensional. Use order=1 for a product of marginals.
    Fitting/selection is host-orchestrated; evaluation and sampling are jittable.
    SGM tries five seeds by default; set n_seeds=None to try every candidate.
    """
    samples, weights = samples_and_weights(samples, weights)
    dimension = samples.shape[1]
    order = positive_integer(order, "order")
    if order > dimension or order >= 10:
        raise ValueError("order must satisfy 1 <= order <= dimension and order < 10")
    candidates = candidate_blocks(dimension, order)
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
            progress_every=candidate_progress_every,
        )
        selection = _select_validated(
            candidates,
            correlations.scores,
            dimension=dimension,
            n_seeds=n_seeds,
            seed_batch_size=seed_batch_size,
            early_stop=sgm_early_stop,
            progress_every=seed_progress_every,
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
    state = _fit_factors(samples, weights, blocks, bandwidth_scale, regularization)
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
    # A scan carry cannot change dtype when queries or later groups are wider.
    dtype = jnp.result_type(points, *(group.factors.centers for group in state._groups))
    result = jnp.zeros(points.shape[:-1], dtype=dtype)
    for group in state._groups:

        def add_factor(total, item):
            block, factor = item
            values = kde_log_prob(factor, points[..., block], chunk_size=chunk_size)
            return total + values, None

        result, _ = jax.lax.scan(add_factor, result, group)
    return result


@partial(jax.jit, static_argnames=("n",))
def sample(key, state, n):
    """Independent joint proposal draws, using a separate key per factor."""
    positive_integer(n, "n")
    keys = jax.random.split(key, sum(map(len, state._layout)))
    result = jnp.zeros((n, state.dimension), dtype=state._groups[0].factors.centers.dtype)
    for group, positions in zip(state._groups, state._layout):
        draws = jax.lax.map(
            lambda item: sample_kde(item[0], item[1], n),
            (keys[jnp.asarray(positions)], group.factors),
            batch_size=min(16, len(positions)),
        )
        values = draws.transpose(1, 0, 2).reshape(n, -1)
        result = result.at[:, group.indices.reshape(-1)].set(values)
    return result
