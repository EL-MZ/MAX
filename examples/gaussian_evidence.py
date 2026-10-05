"""A conjugate Gaussian model with exact posterior draws and known evidence."""

import jax
import jax.numpy as jnp

from jax_morph import estimate_evidence

jax.config.update("jax_enable_x64", True)


def main():
    # Prior theta ~ N(0, C); likelihood y | theta ~ N(theta, I).
    prior_covariance = jnp.array([
        [2.0, 1.7, 0.0, 0.0], [1.7, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 0.7, 1.0],
    ])
    observation = jnp.array([0.3, -0.2, 0.8, -0.1])
    dimension = observation.size
    prior_precision = jnp.linalg.inv(prior_covariance)
    posterior_covariance = jnp.linalg.inv(prior_precision + jnp.eye(dimension))
    posterior_mean = posterior_covariance @ observation
    prior_log_constant = -0.5 * (
        dimension * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(prior_covariance)[1]
    )
    likelihood_log_constant = -0.5 * dimension * jnp.log(2 * jnp.pi)

    def log_density(theta):
        residual = observation - theta
        return (
            prior_log_constant - 0.5 * theta @ prior_precision @ theta
            + likelihood_log_constant - 0.5 * residual @ residual
        )

    samples = jax.random.multivariate_normal(
        jax.random.key(10), posterior_mean, posterior_covariance, (2400,)
    )
    result = estimate_evidence(
        samples, log_density, jax.random.key(20), order=2, max_tc_samples=256,
        n_proposal=1500, posterior_iat=1,
    )
    marginal_covariance = prior_covariance + jnp.eye(dimension)
    exact = -0.5 * (
        dimension * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(marginal_covariance)[1]
        + observation @ jnp.linalg.solve(marginal_covariance, observation)
    )
    print(f"Exact log Z: {float(exact):.6f}")
    print(f"MorphZ log Z: {float(result.bridge.log_evidence):.6f}")
    print(f"Relative MC error: {float(result.bridge.relative_error):.6f}")
    print(f"Selected blocks: {result.morph.state.blocks}")
    print(f"Target evaluations: {result.target_evaluations}")
    assert result.bridge.converged
    assert abs(float(result.bridge.log_evidence - exact)) < 0.1


if __name__ == "__main__":
    main()
