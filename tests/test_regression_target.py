"""Evidence checked against SciPy observation-space densities; exact posterior draws checked."""

import json

import numpy as np
import pytest
from scipy import stats

from benchmarks.highdim_target import load_target
from benchmarks.regression_target import gaussian_log_evidence, make_regression_target


def test_fixed_noise_gaussian_evidence():
    rng = np.random.default_rng(4)
    design = rng.normal(size=(7, 20))
    mean = rng.normal(size=20)
    prior_covariance = np.diag(rng.uniform(0.3, 2, 20))
    noise = rng.normal(size=(7, 7))
    noise_covariance = noise @ noise.T + np.eye(7)
    y = rng.normal(size=7)
    expected = stats.multivariate_normal.logpdf(
        y, design @ mean, noise_covariance + design @ prior_covariance @ design.T
    )
    np.testing.assert_allclose(
        gaussian_log_evidence(design, y, mean, prior_covariance, noise_covariance),
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("orders", [[1], [2], [3], [4], [5], [1, 2, 3, 4, 5]])
def test_group_evidence_matches_multivariate_student_t_marginal(orders):
    target = make_regression_target(20, seed=2, group_orders=orders)
    expected = 0.0
    for group in target.factors:
        n = len(group.observations)
        scale = (
            group.b0
            / group.a0
            * (np.eye(n) + group.design @ group.prior_covariance @ group.design.T)
        )
        expected += stats.multivariate_t.logpdf(
            group.observations,
            loc=group.design @ group.prior_mean,
            shape=scale,
            df=2 * group.a0,
        )
    np.testing.assert_allclose(target.log_evidence, expected, atol=1e-11)


@pytest.mark.parametrize("coordinate", ["variance", "log_variance"])
def test_actual_prior_likelihood_and_posterior_bayes_identity(coordinate, tmp_path):
    target = make_regression_target(
        200,
        seed=7,
        group_orders=[1, 2, 3, 4, 5],
        variance_coordinate=coordinate,
    )
    points = target.sample(np.random.default_rng(4), 100)
    log_f = target.log_prior(points) + target.log_likelihood(points)
    np.testing.assert_allclose(log_f - target.logpdf(points), target.log_evidence, atol=1e-10)
    assert sorted(i for group in target.blocks for i in group) == list(range(200))
    assert set(map(len, target.blocks)) == {1, 2, 3, 4, 5}
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target.description()))
    loaded = load_target(path)
    np.testing.assert_array_equal(loaded.log_unnormalized(points), target.log_unnormalized(points))


def test_direct_posterior_sampler_marginals_and_conditional_gaussian():
    target = make_regression_target(5, seed=9)
    group = target.factors[0]
    draws = target.sample(np.random.default_rng(18), 30000)[:, group.indices]
    beta, variance = draws[:, :-1], draws[:, -1]
    assert stats.kstest(variance, stats.invgamma(group.an, scale=group.bn).cdf).statistic < 0.015
    for index in range(4):
        scale = np.sqrt(group.bn / group.an * group.posterior_covariance[index, index])
        distribution = stats.t(2 * group.an, loc=group.posterior_mean[index], scale=scale)
        assert stats.kstest(beta[:, index], distribution.cdf).statistic < 0.015
    white = ((beta - group.posterior_mean) / np.sqrt(variance[:, None])) @ np.linalg.inv(
        np.linalg.cholesky(group.posterior_covariance)
    ).T
    np.testing.assert_allclose(white.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose(np.cov(white.T), np.eye(4), atol=0.025)
    assert abs(np.corrcoef(variance[:-1], variance[1:])[0, 1]) < 0.025


def test_variance_support_and_log_transform_jacobian():
    raw = make_regression_target(10, seed=4)
    transformed = make_regression_target(10, seed=4, variance_coordinate="log_variance")
    points = raw.sample(np.random.default_rng(9), 40)
    log_points = points.copy()
    variance_indices = [group.indices[-1] for group in raw.factors]
    jacobian = np.log(points[:, variance_indices]).sum(axis=1)
    log_points[:, variance_indices] = np.log(points[:, variance_indices])
    np.testing.assert_allclose(
        transformed.log_unnormalized(log_points),
        raw.log_unnormalized(points) + jacobian,
        atol=1e-12,
    )
    np.testing.assert_allclose(transformed.log_likelihood(log_points), raw.log_likelihood(points))
    assert transformed.log_evidence == raw.log_evidence
    points[0, variance_indices[0]] = -1
    assert np.isneginf(raw.log_unnormalized(points)[0])


def test_default_group_count_at_200_dimensions():
    target = make_regression_target(200)
    assert len(target.factors) == 40
    assert all(len(group.indices) == 5 for group in target.factors)


def test_cross_group_regression_preserves_evidence_and_exact_sampler(tmp_path):
    independent = make_regression_target(10, seed=4)
    coupled = make_regression_target(10, seed=4, cross_group_coupling=0.8)
    first, second = coupled.factors
    rng_seed = 12
    base = independent.sample(np.random.default_rng(rng_seed), 50)
    draws = coupled.sample(np.random.default_rng(rng_seed), 50)
    expected = base.copy()
    expected[:, first.indices[0]] += 0.8 * expected[:, second.indices[0]]
    np.testing.assert_array_equal(draws, expected)
    assert len(coupled.blocks) == 1 and len(coupled.blocks[0]) == 10
    assert coupled.log_evidence == independent.log_evidence
    np.testing.assert_allclose(coupled.logpdf(draws), independent.logpdf(base), atol=1e-11)
    np.testing.assert_allclose(
        coupled.log_unnormalized(draws) - coupled.logpdf(draws),
        coupled.log_evidence,
        atol=1e-11,
    )
    # Independent full conditional Gaussian reference under the coefficient shear.
    transform = np.eye(8)
    transform[0, 4] = 0.8
    coefficients = draws[:, first.indices[:-1] + second.indices[:-1]]
    means = transform @ np.concatenate([first.posterior_mean, second.posterior_mean])
    expected_logs = []
    for row, point in enumerate(draws):
        v1, v2 = point[first.indices[-1]], point[second.indices[-1]]
        covariance = np.zeros((8, 8))
        covariance[:4, :4] = v1 * first.posterior_covariance
        covariance[4:, 4:] = v2 * second.posterior_covariance
        covariance = transform @ covariance @ transform.T
        expected_logs.append(
            stats.multivariate_normal.logpdf(coefficients[row], means, covariance)
            + stats.invgamma.logpdf(v1, first.an, scale=first.bn)
            + stats.invgamma.logpdf(v2, second.an, scale=second.bn)
        )
    np.testing.assert_allclose(coupled.logpdf(draws), expected_logs, atol=1e-10)
    path = tmp_path / "coupled.json"
    path.write_text(json.dumps(coupled.description()))
    np.testing.assert_allclose(
        load_target(path).log_unnormalized(draws), coupled.log_unnormalized(draws)
    )
