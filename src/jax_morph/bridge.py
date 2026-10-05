"""Optimal bridge sampling, evaluated entirely in log space (paper Eqs. 9-10)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ._validation import positive_integer


class BridgeResult(NamedTuple):
    log_evidence: jax.Array
    relative_error: jax.Array
    posterior_iat: jax.Array
    posterior_overlap: jax.Array
    proposal_overlap: jax.Array
    importance_ess: jax.Array
    iterations: jax.Array
    converged: jax.Array
    log_evidence_history: jax.Array


def _log_mean_exp(values):
    return logsumexp(values) - jnp.log(values.size)


@jax.jit
def integrated_autocorrelation_time(values):
    """FFT initial-positive-sequence IAT for one ordered chain.

    Pair adjacent positive-lag autocorrelations; stop at the first nonpositive
    pair. Constant sequences return one. This replaces the AR spectral-density
    fit used by Gronau et al.; it is an explicit implementation choice.
    """
    count = values.size
    centered = values - jnp.mean(values)
    fft_size = 1 << (2 * count - 1).bit_length()
    transform = jnp.fft.rfft(centered, n=fft_size)
    covariance = jnp.fft.irfft(transform * jnp.conj(transform), n=fft_size)[:count] / count
    denominator = jnp.where(covariance[0] > 0, covariance[0], 1.0)
    correlations = covariance / denominator
    pair_count = (count - 1) // 2
    pairs = correlations[1:1 + 2 * pair_count].reshape(-1, 2).sum(axis=1)
    retained = jnp.cumprod((pairs > 0).astype(values.dtype))
    tau = 1 + 2 * jnp.sum(pairs * retained)
    return jnp.where(covariance[0] > 0, jnp.clip(tau, 1, count), 1.0)


@partial(jax.jit, static_argnames=("max_iterations", "estimate_iat"))
def bridge_kernel(
    posterior_log_ratios, proposal_log_ratios, *, initial_log_evidence,
    rtol=1e-6, max_iterations=1000, posterior_iat=1.0, estimate_iat=False,
):
    """Unchecked, jittable solver. Ratios are log(unnormalized target/proposal).

    The posterior ratios must be finite; proposal ratios may be -inf outside
    target support. No support rejection or truncated-proposal normalization is
    introduced. The history has static length; unused entries remain NaN.
    """
    n_posterior = posterior_log_ratios.size
    n_proposal = proposal_log_ratios.size
    log_fraction_p = jnp.log(n_posterior / (n_posterior + n_proposal))
    log_fraction_g = jnp.log(n_proposal / (n_posterior + n_proposal))
    history = jnp.full((max_iterations + 1,), jnp.nan).at[0].set(initial_log_evidence)
    initial = (jnp.int32(0), initial_log_evidence, jnp.asarray(jnp.inf), history)

    def unfinished(carry):
        iteration, log_z, difference, _ = carry
        return (iteration < max_iterations) & (difference > jnp.log1p(rtol)) & jnp.isfinite(log_z)

    def update(carry):
        iteration, log_z, _, history = carry
        # Subtract log_z first: both expectations are bounded bridge terms.
        normalized_g = proposal_log_ratios - log_z
        normalized_p = posterior_log_ratios - log_z
        numerator = _log_mean_exp(
            normalized_g - jnp.logaddexp(log_fraction_p + normalized_g, log_fraction_g)
        )
        denominator = _log_mean_exp(
            -jnp.logaddexp(log_fraction_p + normalized_p, log_fraction_g)
        )
        new_log_z = log_z + numerator - denominator
        history = history.at[iteration + 1].set(new_log_z)
        return iteration + 1, new_log_z, jnp.abs(new_log_z - log_z), history

    iterations, log_z, difference, history = jax.lax.while_loop(unfinished, update, initial)
    converged = jnp.isfinite(log_z) & (difference <= jnp.log1p(rtol))
    normalized_g, normalized_p = proposal_log_ratios - log_z, posterior_log_ratios - log_z
    f1 = jnp.exp(
        normalized_g - jnp.logaddexp(log_fraction_p + normalized_g, log_fraction_g)
    )
    f2 = jnp.exp(-jnp.logaddexp(log_fraction_p + normalized_p, log_fraction_g))
    tau = integrated_autocorrelation_time(f2) if estimate_iat else jnp.asarray(posterior_iat)
    relative_variance = (
        jnp.var(f1, ddof=1) / (n_proposal * jnp.mean(f1)**2)
        + tau * jnp.var(f2, ddof=1) / (n_posterior * jnp.mean(f2)**2)
    )
    relative_error = jnp.where(converged, jnp.sqrt(relative_variance), jnp.nan)
    importance_ess = jnp.exp(
        2 * logsumexp(proposal_log_ratios) - logsumexp(2 * proposal_log_ratios)
    )
    return BridgeResult(
        log_z, relative_error, tau, jnp.exp(log_fraction_g) * jnp.mean(f2),
        jnp.exp(log_fraction_p) * jnp.mean(f1), importance_ess, iterations, converged, history,
    )


def bridge_sampling(
    log_target_posterior, log_proposal_posterior, log_target_proposal, log_proposal_proposal,
    *, initial_log_evidence=None, rtol=1e-6, max_iterations=1000, posterior_iat=None,
):
    """Estimate log Z from equal-weight posterior and independent proposal draws.

    Supply four vectors of evaluated log densities; the target is likelihood
    times a proper prior, including every normalization constant. A supplied
    posterior_iat must be >= 1; None estimates it from the ordered bridge f2
    sequence. Relative error is a conditional Monte Carlo diagnostic, not a
    guarantee of posterior coverage or an assessment of fitting uncertainty.
    """
    arrays = tuple(np.asarray(value) for value in (
        log_target_posterior, log_proposal_posterior, log_target_proposal, log_proposal_proposal
    ))
    if any(value.ndim != 1 or value.size < 2 for value in arrays):
        raise ValueError("log-density arrays must be one-dimensional with at least two draws")
    target_p, proposal_p, target_g, proposal_g = arrays
    if target_p.shape != proposal_p.shape or target_g.shape != proposal_g.shape:
        raise ValueError("target and proposal arrays must match within each sample group")
    if not all(np.isfinite(value).all() for value in (target_p, proposal_p, proposal_g)):
        raise ValueError("posterior target values and all proposal density values must be finite")
    if np.any(np.isnan(target_g)) or np.any(np.isposinf(target_g)):
        raise ValueError("proposal target values may be finite or -inf, never NaN or +inf")
    if not np.isfinite(target_g).any():
        raise ValueError("no proposal draws overlap target support; evidence is not identifiable")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    positive_integer(max_iterations, "max_iterations")
    if posterior_iat is not None and (not np.isfinite(posterior_iat) or posterior_iat < 1):
        raise ValueError("posterior_iat must be finite and >= 1")
    posterior_ratios, proposal_ratios = jnp.asarray(target_p - proposal_p), jnp.asarray(
        target_g - proposal_g
    )
    initial = _log_mean_exp(proposal_ratios) if initial_log_evidence is None else jnp.asarray(
        initial_log_evidence
    )
    if not np.isfinite(np.asarray(initial)).all() or initial.ndim != 0:
        raise ValueError("initial_log_evidence must be a finite scalar")
    return bridge_kernel(
        posterior_ratios, proposal_ratios, initial_log_evidence=initial,
        rtol=rtol, max_iterations=max_iterations,
        posterior_iat=1.0 if posterior_iat is None else posterior_iat,
        estimate_iat=posterior_iat is None,
    )
