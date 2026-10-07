"""Numerical and compilation regressions for the JAX execution refactor."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morph import MorphState, estimate_total_correlations, fit, fit_kde, log_prob, sample
from jax_morph.evidence import _evaluate
from jax_morph.kde import kde_log_prob, sample_kde


def test_target_evaluator_reuses_trace_and_handles_remainder():
    traces = []

    def target(point):
        traces.append(point.shape)
        return -jnp.sum(point**2)

    points = jnp.arange(48.0).reshape(16, 3)
    np.testing.assert_allclose(_evaluate(target, points, 4), -np.sum(points**2, axis=1))
    initial_traces = len(traces)
    assert initial_traces > 0
    # JAX versions can trace a separate (even empty) lax.map remainder branch.
    for _ in range(3):
        np.testing.assert_allclose(_evaluate(target, points, 4), -np.sum(points**2, axis=1))
        assert len(traces) == initial_traces
    np.testing.assert_allclose(_evaluate(target, points[:11], 4), -np.sum(points[:11] ** 2, axis=1))


@pytest.mark.parametrize("shift", [0.0, 1e8])
def test_entropy_reduction_matches_pointwise_weighted_tc(shift):
    rng = np.random.default_rng(22)
    data = rng.normal(size=(73, 5)) + shift
    weights = rng.uniform(size=len(data))
    weights[:4] = 0
    normalized = weights / weights.sum()
    blocks = jnp.array([[0, 2], [1, 3], [3, 4]])
    actual = estimate_total_correlations(
        data, blocks, weights, chunk_size=17, candidate_batch_size=2
    )
    marginals = np.column_stack(
        [
            kde_log_prob(fit_kde(data[:, i, None], weights), data[:, i, None], chunk_size=17)
            for i in range(data.shape[1])
        ]
    )
    expected = []
    for block in np.asarray(blocks):
        values = data[:, block]
        joint = kde_log_prob(fit_kde(values, weights), values, chunk_size=17)
        expected.append(normalized @ (joint - marginals[:, block].sum(axis=1)))
    np.testing.assert_allclose(actual.raw_scores, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual.marginal_entropies, -(normalized @ marginals), atol=1e-12)


def test_tc_still_rejects_singular_unused_marginal_and_invalid_options():
    data = np.random.default_rng(1).normal(size=(20, 3))
    data[:, 2] = 1
    with pytest.raises(ValueError, match="nonfinite"):
        estimate_total_correlations(data, [[0, 1]])
    for option in ({"bandwidth_scale": 0}, {"regularization": -1}):
        with pytest.raises(ValueError):
            estimate_total_correlations(data[:, :2], [[0, 1]], **option)


def test_dynamic_partition_reuses_outer_jit_and_preserves_views():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(40, 5))
    partitions = (((0, 1), (2,), (3, 4)), ((0, 3), (1,), (2, 4)))
    states = [
        MorphState(blocks, tuple(fit_kde(data[:, b]) for b in blocks), 5) for blocks in partitions
    ]
    traces = []

    @jax.jit
    def evaluate(state, points):
        traces.append(1)
        return log_prob(state, points, chunk_size=7)

    points = jnp.asarray(data[:11])
    for state, blocks in zip(states, partitions):
        assert state.blocks == blocks
        expected = sum(
            kde_log_prob(factor, points[:, jnp.array(block)], chunk_size=7)
            for block, factor in zip(blocks, state.factors)
        )
        np.testing.assert_allclose(evaluate(state, points), expected, atol=1e-12)
        leaves, structure = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(structure, leaves)
        assert restored.blocks == blocks
        np.testing.assert_allclose(log_prob(restored, points), expected, atol=1e-12)
    assert len(traces) == 1
    grad = jax.grad(lambda x: log_prob(states[0], x))(points[0])
    h = 1e-5
    fd = [
        (log_prob(states[0], points[0].at[i].add(h)) - log_prob(states[0], points[0].at[i].add(-h)))
        / (2 * h)
        for i in range(5)
    ]
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)


def test_grouped_sampling_keeps_original_factor_key_order():
    rng = np.random.default_rng(44)
    data = rng.normal(size=(35, 5))
    blocks = ((0, 4), (2,), (1, 3))
    factors = tuple(fit_kde(data[:, b]) for b in blocks)
    state = MorphState(blocks, factors, 5)
    key = jax.random.key(81)
    expected = jnp.zeros((23, 5))
    for block, factor, subkey in zip(blocks, factors, jax.random.split(key, len(blocks))):
        expected = expected.at[:, jnp.array(block)].set(sample_kde(subkey, factor, 23))
    np.testing.assert_allclose(sample(key, state, 23), expected, rtol=1e-12, atol=1e-12)


def test_grouped_weighted_fits_match_individual_fits():
    rng = np.random.default_rng(24)
    data = rng.normal(size=(55, 7))
    weights = rng.uniform(size=len(data))
    fitted = fit(data, weights, n_seeds=3, candidate_batch_size=4)
    for block, factor in zip(fitted.state.blocks, fitted.state.factors):
        expected = fit_kde(data[:, block], weights)
        for actual_leaf, expected_leaf in zip(factor, expected):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-12, atol=1e-12)


def test_uncapped_bridge_computes_statistics_only_once(monkeypatch):
    bridge = importlib.import_module("jax_morph.bridge")
    calls = []
    original = bridge._bridge_statistics

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "_bridge_statistics", counted)
    monkeypatch.setattr(bridge, "_BRIDGE_CHUNK_ITERATIONS", 3)
    zeros = jnp.zeros(8)
    result = bridge.bridge_sampling(
        zeros + 2, zeros, zeros - 2, zeros, initial_log_evidence=10.0, rtol=1e-8
    )
    jax.block_until_ready(result)
    assert int(result.iterations) > 3
    assert calls == [1]


def test_grouped_density_promotes_mixed_factor_and_query_dtypes():
    data = np.random.default_rng(8).normal(size=(30, 2))
    narrow = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), fit_kde(data[:, :1]))
    wide = fit_kde(data[:, 1:])
    points = jnp.asarray(data[:9], dtype=jnp.float64)
    for factors in ((narrow, narrow), (narrow, wide)):
        state = MorphState(((0,), (1,)), factors, 2)
        expected = kde_log_prob(factors[0], points[:, :1]) + kde_log_prob(factors[1], points[:, 1:])
        np.testing.assert_allclose(log_prob(state, points), expected, rtol=1e-12, atol=1e-12)
        assert log_prob(state, points).dtype == jnp.float64


def test_bridge_statistics_cache_is_independent_of_history_length():
    bridge = importlib.import_module("jax_morph.bridge")
    bridge._bridge_statistics.clear_cache()
    ratios = jnp.linspace(-1.0, 1.0, 16)
    for count in (3, 7):
        result = bridge._BridgeIteration(
            jnp.asarray(0.0), jnp.asarray(count), jnp.asarray(True), jnp.zeros(count + 1)
        )
        actual = bridge._finalize_bridge(
            result, ratios, ratios, average_iterations=5, posterior_iat=1.0, estimate_iat=True
        )
        jax.block_until_ready(actual)
        assert bridge._bridge_statistics._cache_size() == 1
    bridge._bridge_statistics.clear_cache()
