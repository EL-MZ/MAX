"""Check analytic targets independently of the Morph estimator."""

import numpy as np
import pytest
from scipy import stats

from benchmarks.distributions import SCENARIOS, block_metrics, make_target


@pytest.mark.parametrize("name", SCENARIOS)
def test_reproducible_shuffled_target_and_finite_density(name):
    target = make_target(name, 7, seed=13)
    samples = target.sample(np.random.default_rng(4), 1000)
    np.testing.assert_array_equal(samples, target.sample(np.random.default_rng(4), 1000))
    assert samples.shape == (1000, 7)
    assert np.isfinite(target.logpdf(samples)).all()
    assert sorted(i for b in target.blocks for i in b) == list(range(7))
    np.testing.assert_allclose(target.logpdf(samples), target.block_logpdf(samples).sum(axis=1))


def test_copula_density_matches_independent_marginals_and_support():
    target = make_target("independent_mixed", 7, seed=2)
    samples = target.sample(np.random.default_rng(3), 90)
    canonical = samples[:, np.argsort(target.permutation)]
    expected = sum(dist.logpdf(canonical[:, i]) for i, dist in enumerate(target.marginals))
    np.testing.assert_allclose(target.logpdf(samples), expected, atol=1e-12)
    beta_column = int(np.flatnonzero(target.permutation == 2)[0])
    samples[0, beta_column] = -1
    assert np.isneginf(target.logpdf(samples)[0])


def test_banana_density_jacobian_and_zero_linear_correlation():
    target = make_target("banana_pairs", 2)
    canonical = np.array([[0.3, -0.4], [-0.6, 0.2]])
    physical = canonical[:, target.permutation]
    residual = (canonical[:, 1] - 0.7 * (canonical[:, 0] ** 2 - 1)) / 0.2
    expected = stats.norm.logpdf(canonical[:, 0]) + stats.norm.logpdf(residual) - np.log(0.2)
    np.testing.assert_allclose(target.logpdf(physical), expected)
    draws = target.sample(np.random.default_rng(21), 1000)
    assert abs(np.corrcoef(draws.T)[0, 1]) < 0.15


def test_pair_metrics_distinguish_splits_and_mergers():
    truth = ((0, 1, 2), (3,))
    result = block_metrics(truth, ((0, 1), (2, 3)))
    assert result["pair_recall"] == 1 / 3
    assert result["pair_precision"] == 1 / 2
    assert not result["exact_partition"]
    assert block_metrics(((0,), (1,)), ((0, 1),))["pair_recall"] is None
    assert block_metrics(truth, ((2, 1, 0), (3,)))["exact_partition"]
