import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import betaln
from scipy.stats import beta, norm

from jax_morph import bridge_sampling, estimate_evidence
from jax_morph.bridge import integrated_autocorrelation_time


@pytest.mark.parametrize("offset", [-1e5, 0.0, 1e5])
def test_exact_proposal_and_large_log_normalizer(offset):
    log_g_p = norm.logpdf(np.linspace(-2, 2, 41))
    log_g_g = norm.logpdf(np.linspace(-3, 3, 57))
    result = bridge_sampling(log_g_p + offset, log_g_p, log_g_g + offset, log_g_g)
    assert result.converged
    assert abs(result.log_evidence - offset) < 1e-9
    assert result.relative_error < 1e-10
    np.testing.assert_allclose(result.importance_ess, 57)


def test_bounded_target_keeps_proposals_outside_support():
    rng = np.random.default_rng(12)
    posterior = rng.beta(3, 6, 6000)
    proposal = rng.normal(0.35, 0.2, 6000)
    log_z = betaln(3, 6)
    result = bridge_sampling(
        beta.logpdf(posterior, 3, 6) + log_z,
        norm.logpdf(posterior, 0.35, 0.2),
        beta.logpdf(proposal, 3, 6) + log_z,
        norm.logpdf(proposal, 0.35, 0.2),
        posterior_iat=1,
    )
    assert np.isneginf(beta.logpdf(proposal, 3, 6)).any()
    assert result.converged
    assert abs(result.log_evidence - log_z) < 5 * result.relative_error + 0.01
    assert 0 < result.proposal_overlap < 1
    assert 0 < result.posterior_overlap < 1


def test_iat_adjusts_error_for_correlated_chain():
    rng = np.random.default_rng(7)
    chain = np.zeros(8192)
    noise = rng.normal(size=len(chain))
    for i in range(1, len(chain)):
        chain[i] = 0.8 * chain[i - 1] + noise[i]
    tau = integrated_autocorrelation_time(jnp.asarray(chain))
    assert 6 < tau < 13  # Analytic AR(1) IAT is (1 + 0.8) / (1 - 0.8) = 9.
    np.testing.assert_allclose(integrated_autocorrelation_time(jnp.ones(100)), 1)
    target_p = norm.logpdf(chain)
    proposal_p = norm.logpdf(chain, scale=2)
    points = rng.normal(scale=2, size=1000)
    target_g, proposal_g = norm.logpdf(points), norm.logpdf(points, scale=2)
    iid = bridge_sampling(target_p, proposal_p, target_g, proposal_g, posterior_iat=1)
    correlated = bridge_sampling(target_p, proposal_p, target_g, proposal_g, posterior_iat=9)
    assert correlated.relative_error > iid.relative_error
    np.testing.assert_allclose(iid.log_evidence, correlated.log_evidence)


def test_no_overlap_nonconvergence_and_invalid_values():
    zeros = np.zeros(20)
    with pytest.raises(ValueError, match="overlap"):
        bridge_sampling(zeros, zeros, np.full(20, -np.inf), zeros)
    with pytest.raises(ValueError, match="NaN"):
        bridge_sampling(zeros, zeros, np.full(20, np.nan), zeros)
    points = np.linspace(-2, 2, 20)
    result = bridge_sampling(
        norm.logpdf(points),
        norm.logpdf(points, scale=2),
        norm.logpdf(2 * points),
        norm.logpdf(2 * points, scale=2),
        initial_log_evidence=10,
        max_iterations=1,
    )
    assert not result.converged
    assert np.isnan(result.relative_error)


@pytest.mark.parametrize("geometric_restart", [True, False])
def test_end_to_end_beta_evidence_with_cached_target(geometric_restart):
    rng = np.random.default_rng(24)
    samples = rng.beta(3, 6, size=(1400, 1))

    def log_density(point):
        theta = point[0]
        return jnp.where(
            (theta > 0) & (theta < 1), 2 * jnp.log(theta) + 5 * jnp.log1p(-theta), -jnp.inf
        )

    cached = 2 * np.log(samples[:, 0]) + 5 * np.log1p(-samples[:, 0])
    result = estimate_evidence(
        samples,
        log_density,
        jax.random.key(41),
        order=1,
        n_proposal=1200,
        posterior_log_density=cached,
        posterior_iat=1,
        geometric_restart=geometric_restart,
        restart_interval=2,
    )
    assert result.bridge.converged
    assert abs(result.bridge.log_evidence - betaln(3, 6)) < 0.04
    assert not set(np.asarray(result.fit_indices)).intersection(result.bridge_indices.tolist())
    assert result.target_evaluations == 1200


def test_end_to_end_correlated_gaussian():
    covariance = jnp.array([[1.0, 0.85, 0.0], [0.85, 1.0, 0.0], [0.0, 0.0, 2.0]])
    samples = jax.random.multivariate_normal(jax.random.key(1), jnp.zeros(3), covariance, (1600,))
    precision = jnp.linalg.inv(covariance)
    log_normalizer = 0.5 * (3 * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(covariance)[1])

    def log_density(point):
        return -0.5 * point @ precision @ point - log_normalizer + 2.3

    result = estimate_evidence(
        samples,
        log_density,
        jax.random.key(2),
        max_tc_samples=160,
        posterior_iat=1,
    )
    assert result.bridge.converged
    assert abs(result.bridge.log_evidence - 2.3) < 0.05
    assert {0, 1} == set(result.morph.state.blocks[0])
