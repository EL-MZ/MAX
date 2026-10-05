import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morph import (
    candidate_blocks,
    estimate_total_correlations,
    fit,
    kde_log_prob,
    log_prob,
    sample,
)


def test_total_correlation_detects_nonlinear_dependence():
    rng = np.random.default_rng(9)
    x = rng.uniform(-2, 2, 350)
    y = x**2 + rng.normal(scale=0.1, size=len(x))
    samples = np.column_stack((x, y, rng.normal(size=len(x))))
    scores = estimate_total_correlations(samples, candidate_blocks(3, 2), chunk_size=37)
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.2
    assert scores.scores[0] > 0.4
    assert scores.scores[0] > max(scores.scores[1:]) + 0.3


def test_gaussian_total_correlation_and_affine_units():
    rng = np.random.default_rng(18)
    rho = 0.8
    samples = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=700)
    blocks = candidate_blocks(2, 2)
    score = estimate_total_correlations(samples, blocks).scores[0]
    assert abs(score + 0.5 * np.log(1 - rho**2)) < 0.12
    transformed = samples * [7, 0.2] + [11, -4]
    other = estimate_total_correlations(transformed, blocks).scores[0]
    np.testing.assert_allclose(score, other, atol=1e-10)


def test_partition_singletons_product_density_and_independent_draws():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(240, 5))
    samples[:, 1] = samples[:, 0] + rng.normal(scale=0.1, size=240)
    fitted = fit(samples, order=2, max_tc_samples=80, key=jax.random.key(11))
    state = fitted.state
    flattened = [i for block in state.blocks for i in block]
    assert sorted(flattened) == list(range(5))
    assert [len(block) for block in state.blocks] == [2, 2, 1]
    assert fitted.diagnostics.tc_sample_count == 80
    assert all(len(factor.centers) == 240 for factor in state.factors)
    points = jnp.asarray(rng.normal(size=(20, 5)))
    expected = sum(
        kde_log_prob(factor, points[:, jnp.asarray(block)])
        for block, factor in zip(state.blocks, state.factors)
    )
    np.testing.assert_allclose(log_prob(state, points), expected)
    assert jax.grad(lambda point: log_prob(state, point))(points[0]).shape == (5,)
    draws = np.asarray(jax.jit(lambda key: sample(key, state, 10_000))(jax.random.key(2)))
    a, b = state.blocks[0][0], state.blocks[-1][0]
    assert abs(np.corrcoef(draws[:, a], draws[:, b])[0, 1]) < 0.05
    np.testing.assert_array_equal(
        sample(jax.random.key(7), state, 8), sample(jax.random.key(7), state, 8)
    )


def test_weighted_fit_and_order_one():
    samples = np.random.default_rng(8).normal(size=(70, 3))
    weights = np.linspace(1, 3, len(samples))
    result = fit(samples, weights, order=1)
    assert result.state.blocks == ((0,), (1,), (2,))
    assert result.diagnostics.tc_sample_count == 0
    assert result.state.factors[0].effective_sample_size < len(samples)


def test_invalid_factor_order_and_thinning_key():
    samples = np.random.default_rng(9).normal(size=(20, 10))
    with pytest.raises(ValueError, match="order"):
        fit(samples, order=10)
    with pytest.raises(ValueError, match="key"):
        fit(samples, max_tc_samples=10)
