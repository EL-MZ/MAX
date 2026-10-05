import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

from jax_morph import fit_kde, kde_log_prob, sample_kde


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_silverman_matches_scipy(dimension, weighted):
    rng = np.random.default_rng(32)
    samples = rng.normal(size=(91, dimension))
    samples[:, 0] *= 3
    weights = rng.uniform(size=91) if weighted else None
    if weighted:
        weights[:3] = 0
    points = rng.normal(size=(37, dimension))
    state = fit_kde(samples, weights)
    reference = gaussian_kde(samples.T, weights=weights, bw_method="silverman")
    np.testing.assert_allclose(state.cholesky @ state.cholesky.T, reference.covariance)
    np.testing.assert_allclose(kde_log_prob(state, points, chunk_size=7), reference.logpdf(points.T))
    np.testing.assert_allclose(kde_log_prob(state, points[0]), reference.logpdf(points[0])[0])
    assert kde_log_prob(state, np.empty((0, dimension))).shape == (0,)


def test_density_normalization_and_sampling_moments():
    values = np.linspace(-2, 2, 100)[:, None]
    weights = np.linspace(1, 4, 100)
    state = fit_kde(values, weights)
    grid = np.linspace(-8, 8, 5000)
    integral = trapezoid(np.exp(kde_log_prob(state, grid[:, None])), grid)
    assert abs(integral - 1) < 1e-6
    draws = np.asarray(sample_kde(jax.random.key(45), state, 30_000))[:, 0]
    normalized = weights / weights.sum()
    expected_mean = normalized @ values[:, 0]
    expected_variance = normalized @ (values[:, 0] - expected_mean)**2 + state.cholesky[0, 0]**2
    assert abs(draws.mean() - expected_mean) < 0.025
    assert abs(draws.var() - expected_variance) < 0.04


def test_log_density_gradient_and_tail_stability():
    state = fit_kde(np.linspace(-1, 1, 31)[:, None])
    point = jnp.array([0.37])
    gradient = jax.grad(lambda x: kde_log_prob(state, x))(point)
    step = 1e-5
    finite_difference = (kde_log_prob(state, point + step) - kde_log_prob(state, point - step)) / (
        2 * step
    )
    np.testing.assert_allclose(gradient[0], finite_difference, rtol=1e-6)
    assert np.isfinite(kde_log_prob(state, jnp.array([1e4])))


def test_singular_covariance_is_explicit():
    values = np.linspace(-1, 1, 20)
    samples = np.column_stack((values, values))
    with pytest.raises(ValueError, match="singular"):
        fit_kde(samples)
    assert np.isfinite(fit_kde(samples, regularization=1e-6).cholesky).all()
    with pytest.raises(ValueError, match="singular"):
        fit_kde(np.ones((20, 1)), regularization=1e-6)


@pytest.mark.parametrize("weights", [np.zeros(10), np.ones(9), -np.ones(10)])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        fit_kde(np.arange(10)[:, None], weights)
