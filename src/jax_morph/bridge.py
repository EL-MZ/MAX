"""Optimal bridge sampling, evaluated entirely in log space (paper Eqs. 9-10)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ._progress import validate_progress_every
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

    @property
    def final_log_evidence(self):
        """Actual final solver iterate, before recent-iteration averaging."""
        return self.log_evidence_history[self.iterations]


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
    offsets = values - values[0]
    centered = offsets - jnp.mean(offsets)
    fft_size = 1 << (2 * count - 1).bit_length()
    transform = jnp.fft.rfft(centered, n=fft_size)
    covariance = jnp.fft.irfft(transform * jnp.conj(transform), n=fft_size)[:count] / count
    denominator = jnp.where(covariance[0] > 0, covariance[0], 1.0)
    correlations = covariance / denominator
    pair_count = (count - 1) // 2
    pairs = correlations[1 : 1 + 2 * pair_count].reshape(-1, 2).sum(axis=1)
    retained = jnp.cumprod((pairs > 0).astype(values.dtype))
    tau = 1 + 2 * jnp.sum(pairs * retained)
    return jnp.where(covariance[0] > 0, jnp.clip(tau, 1, count), 1.0)


@partial(jax.jit, static_argnames=("estimate_iat",))
def _bridge_statistics(
    log_z, posterior_ratios, proposal_ratios, converged, posterior_iat, estimate_iat
):
    n_posterior, n_proposal = posterior_ratios.size, proposal_ratios.size
    log_fraction_p = jnp.log(n_posterior / (n_posterior + n_proposal))
    log_fraction_g = jnp.log(n_proposal / (n_posterior + n_proposal))
    normalized_g, normalized_p = proposal_ratios - log_z, posterior_ratios - log_z
    f1 = jnp.exp(normalized_g - jnp.logaddexp(log_fraction_p + normalized_g, log_fraction_g))
    f2 = jnp.exp(-jnp.logaddexp(log_fraction_p + normalized_p, log_fraction_g))
    tau = integrated_autocorrelation_time(f2) if estimate_iat else jnp.asarray(posterior_iat)
    relative_variance = jnp.var(f1, ddof=1) / (n_proposal * jnp.mean(f1) ** 2) + tau * jnp.var(
        f2, ddof=1
    ) / (n_posterior * jnp.mean(f2) ** 2)
    relative_error = jnp.where(converged, jnp.sqrt(relative_variance), jnp.nan)
    importance_ess = jnp.exp(2 * logsumexp(proposal_ratios) - logsumexp(2 * proposal_ratios))
    return (
        log_z,
        relative_error,
        tau,
        jnp.exp(log_fraction_g) * jnp.mean(f2),
        jnp.exp(log_fraction_p) * jnp.mean(f1),
        importance_ess,
    )


@partial(
    jax.jit,
    static_argnames=("max_iterations", "estimate_iat", "progress_every", "report_limit"),
)
def bridge_kernel(
    posterior_log_ratios,
    proposal_log_ratios,
    *,
    initial_log_evidence,
    rtol=0.1,
    max_iterations=1000,
    posterior_iat=1.0,
    estimate_iat=False,
    progress_every=0,
    iteration_offset=0,
    report_limit=True,
):
    """Unchecked, jittable solver. Ratios are log(unnormalized target/proposal).

    The posterior ratios must be finite; proposal ratios may be -inf outside
    target support. No support rejection or truncated-proposal normalization is
    introduced. This is a bounded compiled chunk; the public bridge_sampling
    wrapper continues chunks without a cap when max_iterations is None.
    The history has static length; unused entries remain NaN.
    """
    n_posterior = posterior_log_ratios.size
    n_proposal = proposal_log_ratios.size
    log_fraction_p = jnp.log(n_posterior / (n_posterior + n_proposal))
    log_fraction_g = jnp.log(n_proposal / (n_posterior + n_proposal))
    history = jnp.full((max_iterations + 1,), jnp.nan).at[0].set(initial_log_evidence)
    initial = (jnp.int32(0), initial_log_evidence, jnp.asarray(jnp.inf), history)

    def unfinished(carry):
        iteration, log_z, difference, _ = carry
        return (iteration < max_iterations) & (difference > rtol) & jnp.isfinite(log_z)

    def update(carry):
        iteration, log_z, _, history = carry
        # Subtract log_z first: both expectations are bounded bridge terms.
        normalized_g = proposal_log_ratios - log_z
        normalized_p = posterior_log_ratios - log_z
        numerator = _log_mean_exp(
            normalized_g - jnp.logaddexp(log_fraction_p + normalized_g, log_fraction_g)
        )
        denominator = _log_mean_exp(-jnp.logaddexp(log_fraction_p + normalized_p, log_fraction_g))
        new_log_z = log_z + numerator - denominator
        history = history.at[iteration + 1].set(new_log_z)
        difference = jnp.abs(new_log_z - log_z)
        if progress_every:
            completed = iteration_offset + iteration + 1
            show = (
                (completed == 1)
                | (completed % progress_every == 0)
                | (report_limit & (iteration + 1 == max_iterations))
                | (difference <= rtol)
                | ~jnp.isfinite(new_log_z)
            )

            def report(_):
                def output(index, value, change):
                    limit = f"/{max_iterations:,}" if report_limit else " (uncapped)"
                    print(
                        f"    bridge iteration {int(index):,}{limit}: "
                        f"logZ={float(value):.8f}, delta_logZ={float(change):.3g}",
                        flush=True,
                    )

                jax.debug.callback(output, completed, new_log_z, difference, ordered=True)

            jax.lax.cond(show, report, lambda _: None, None)
        return iteration + 1, new_log_z, difference, history

    iterations, log_z, difference, history = jax.lax.while_loop(unfinished, update, initial)
    converged = jnp.isfinite(log_z) & (difference <= rtol)
    if progress_every:

        def finish(index, success):
            status = "converged" if bool(success) else "stopped without convergence"
            print(f"    bridge finished at iteration {int(index):,}: {status}", flush=True)

        def report_finish(_):
            jax.debug.callback(finish, iteration_offset + iterations, converged, ordered=True)

        jax.lax.cond(
            report_limit | converged | ~jnp.isfinite(log_z) | (iterations < max_iterations),
            report_finish,
            lambda _: None,
            None,
        )
    statistics = _bridge_statistics(
        log_z,
        posterior_log_ratios,
        proposal_log_ratios,
        converged,
        posterior_iat,
        estimate_iat=estimate_iat,
    )
    return BridgeResult(
        *statistics,
        iterations,
        converged,
        history,
    )


_BRIDGE_CHUNK_ITERATIONS = 1000


def _average_bridge_result(
    result, posterior_ratios, proposal_ratios, *, average_iterations, posterior_iat, progress_every
):
    count = int(result.iterations)
    window = min(count, average_iterations)
    if not window:
        return result
    values = np.asarray(result.log_evidence_history)[count - window + 1 : count + 1]
    # Keep numerical failures visible instead of averaging them away.
    if not np.isfinite(values).all():
        return result
    log_z = jnp.asarray(values.mean(), dtype=result.log_evidence.dtype)
    statistics = _bridge_statistics(
        log_z,
        posterior_ratios,
        proposal_ratios,
        result.converged,
        1.0 if posterior_iat is None else posterior_iat,
        estimate_iat=posterior_iat is None,
    )
    averaged = BridgeResult(
        *statistics, result.iterations, result.converged, result.log_evidence_history
    )
    if progress_every:
        jax.effects_barrier()
        print(
            f"    bridge reported logZ={float(log_z):.8f}: mean of last {window} iterations; "
            f"final iterate={float(result.final_log_evidence):.8f}",
            flush=True,
        )
    return averaged


def bridge_sampling(
    log_target_posterior,
    log_proposal_posterior,
    log_target_proposal,
    log_proposal_proposal,
    *,
    initial_log_evidence=None,
    rtol=0.1,
    max_iterations=None,
    posterior_iat=None,
    progress_every=0,
    average_iterations=5,
):
    """Estimate log Z from equal-weight posterior and independent proposal draws.

    Supply four vectors of evaluated log densities; the target is likelihood
    times a proper prior, including every normalization constant. A supplied
    posterior_iat must be >= 1; None estimates it from the ordered bridge f2
    sequence. Relative error is a conditional Monte Carlo diagnostic, not a
    guarantee of posterior coverage or an assessment of fitting uncertainty.
    Stop when abs(log_Z_new - log_Z_previous) <= rtol. By default there is no
    iteration cap; supply max_iterations to impose one. Uncapped runs use
    compiled chunks and preserve the full history across chunk boundaries.
    Report the arithmetic mean of the last average_iterations computed log-Z
    values (default 5), excluding the initialization. Convergence still uses
    consecutive solver iterates. final_log_evidence retains the final iterate.
    """
    arrays = tuple(
        np.asarray(value)
        for value in (
            log_target_posterior,
            log_proposal_posterior,
            log_target_proposal,
            log_proposal_proposal,
        )
    )
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
    if max_iterations is not None:
        positive_integer(max_iterations, "max_iterations")
    validate_progress_every(progress_every)
    positive_integer(average_iterations, "average_iterations")
    if posterior_iat is not None and (not np.isfinite(posterior_iat) or posterior_iat < 1):
        raise ValueError("posterior_iat must be finite and >= 1")
    posterior_ratios, proposal_ratios = (
        jnp.asarray(target_p - proposal_p),
        jnp.asarray(target_g - proposal_g),
    )
    initial = (
        _log_mean_exp(proposal_ratios)
        if initial_log_evidence is None
        else jnp.asarray(initial_log_evidence)
    )
    if not np.isfinite(np.asarray(initial)).all() or initial.ndim != 0:
        raise ValueError("initial_log_evidence must be a finite scalar")
    if progress_every:
        limit = f"/{max_iterations:,}" if max_iterations is not None else " (uncapped)"
        print(
            f"    bridge iteration 0{limit}: logZ={float(initial):.8f}",
            flush=True,
        )
    options = dict(
        rtol=rtol,
        posterior_iat=1.0 if posterior_iat is None else posterior_iat,
        estimate_iat=posterior_iat is None,
        progress_every=progress_every,
    )
    if max_iterations is not None:
        result = bridge_kernel(
            posterior_ratios,
            proposal_ratios,
            initial_log_evidence=initial,
            max_iterations=max_iterations,
            **options,
        )
        return _average_bridge_result(
            result,
            posterior_ratios,
            proposal_ratios,
            average_iterations=average_iterations,
            posterior_iat=posterior_iat,
            progress_every=progress_every,
        )
    histories = [np.asarray(initial).reshape(1)]
    iterations = 0
    while True:
        result = bridge_kernel(
            posterior_ratios,
            proposal_ratios,
            initial_log_evidence=initial,
            max_iterations=_BRIDGE_CHUNK_ITERATIONS,
            iteration_offset=iterations,
            report_limit=False,
            **options,
        )
        completed = int(result.iterations)
        histories.append(np.asarray(result.log_evidence_history)[1 : completed + 1])
        iterations += completed
        if (
            bool(result.converged)
            or not np.isfinite(float(result.log_evidence))
            or completed < _BRIDGE_CHUNK_ITERATIONS
        ):
            result = result._replace(
                iterations=jnp.asarray(iterations),
                log_evidence_history=jnp.asarray(np.concatenate(histories)),
            )
            return _average_bridge_result(
                result,
                posterior_ratios,
                proposal_ratios,
                average_iterations=average_iterations,
                posterior_iat=posterior_iat,
                progress_every=progress_every,
            )
        initial = result.log_evidence
