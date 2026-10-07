"""Optimal bridge sampling, evaluated entirely in log space (paper Eqs. 9-10)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ._progress import emit_progress, validate_progress_every
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


class _BridgeIteration(NamedTuple):
    log_evidence: jax.Array
    iterations: jax.Array
    converged: jax.Array
    log_evidence_history: jax.Array


@partial(
    jax.jit,
    static_argnames=(
        "max_iterations",
        "progress_every",
        "report_limit",
        "geometric_restart",
        "restart_interval",
    ),
)
def _bridge_iterate(
    posterior_log_ratios,
    proposal_log_ratios,
    *,
    initial_log_evidence,
    rtol=0.1,
    max_iterations=1000,
    progress_every=0,
    iteration_offset=0,
    report_limit=True,
    geometric_restart=True,
    restart_interval=1000,
    previous_log_evidence=None,
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
    dtype = jnp.result_type(posterior_log_ratios, proposal_log_ratios, initial_log_evidence, 0.0)
    initial_log_evidence = jnp.asarray(initial_log_evidence, dtype=dtype)
    log_fraction_p = jnp.log(n_posterior / (n_posterior + n_proposal))
    log_fraction_g = jnp.log(n_proposal / (n_posterior + n_proposal))
    history = jnp.full((max_iterations + 1,), jnp.nan, dtype=dtype).at[0].set(initial_log_evidence)
    previous = initial_log_evidence if previous_log_evidence is None else previous_log_evidence
    previous = jnp.asarray(previous, dtype=dtype)
    initial = (jnp.int32(0), initial_log_evidence, jnp.asarray(jnp.inf), history, previous)

    def unfinished(carry):
        iteration, log_z, difference, _, _ = carry
        return (iteration < max_iterations) & (difference > rtol) & jnp.isfinite(log_z)

    def update(carry):
        iteration, log_z, _, history, previous = carry
        old_log_z = log_z
        if geometric_restart:
            completed = iteration_offset + iteration
            restart = (completed >= 2) & (completed % restart_interval == 0)
            # Gronau, Heathcote & Matzke (2020), Behav Res Methods 52, 918-937,
            # Results after Fig. 5: restart oscillating bridge iterations at
            # sqrt(Z_previous * Z_current). DOI: 10.3758/s13428-019-01290-6
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC7148283/
            # In log space this is their arithmetic midpoint, without exponentiation.
            # Repeating this fallback every restart_interval updates is our
            # extension; the tolerance and total iteration cap stay unchanged.
            log_z = jnp.where(restart, 0.5 * previous + 0.5 * log_z, log_z)
            if progress_every:

                def report_restart(_):
                    def output(index, value):
                        if not emit_progress(
                            "Bridge iterations", index, restart_log_z=float(value)
                        ):
                            print(
                                f"    bridge geometric restart after {int(index):,} updates: "
                                f"logZ={float(value):.8f}",
                                flush=True,
                            )

                    jax.debug.callback(output, completed, log_z, ordered=True)

                jax.lax.cond(restart, report_restart, lambda _: None, None)
        # Subtract log_z first: both expectations are bounded bridge terms.
        normalized_g = proposal_log_ratios - log_z
        normalized_p = posterior_log_ratios - log_z
        numerator = _log_mean_exp(
            normalized_g - jnp.logaddexp(log_fraction_p + normalized_g, log_fraction_g)
        )
        denominator = _log_mean_exp(-jnp.logaddexp(log_fraction_p + normalized_p, log_fraction_g))
        new_log_z = log_z + numerator - denominator
        history = history.at[iteration + 1].set(new_log_z)
        # On a restart, check the original bridge update at the new starting
        # value, never the size of the midpoint jump itself.
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
                    if emit_progress(
                        "Bridge iterations",
                        index,
                        max_iterations if report_limit else None,
                        log_z=float(value),
                        delta=float(change),
                    ):
                        return
                    limit = f"/{max_iterations:,}" if report_limit else " (uncapped)"
                    print(
                        f"    bridge iteration {int(index):,}{limit}: "
                        f"logZ={float(value):.8f}, delta_logZ={float(change):.3g}",
                        flush=True,
                    )

                jax.debug.callback(output, completed, new_log_z, difference, ordered=True)

            jax.lax.cond(show, report, lambda _: None, None)
        return iteration + 1, new_log_z, difference, history, old_log_z

    iterations, log_z, difference, history, _ = jax.lax.while_loop(unfinished, update, initial)
    converged = jnp.isfinite(log_z) & (difference <= rtol)
    if progress_every:

        def finish(index, success):
            status = "converged" if bool(success) else "stopped without convergence"
            if emit_progress("Bridge iterations", index, finished=True, status=status):
                return
            print(f"    bridge finished at iteration {int(index):,}: {status}", flush=True)

        def report_finish(_):
            jax.debug.callback(finish, iteration_offset + iterations, converged, ordered=True)

        jax.lax.cond(
            report_limit | converged | ~jnp.isfinite(log_z) | (iterations < max_iterations),
            report_finish,
            lambda _: None,
            None,
        )
    return _BridgeIteration(log_z, iterations, converged, history)


@partial(
    jax.jit,
    static_argnames=(
        "max_iterations",
        "estimate_iat",
        "progress_every",
        "report_limit",
        "geometric_restart",
        "restart_interval",
    ),
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
    geometric_restart=True,
    restart_interval=1000,
    previous_log_evidence=None,
):
    """Public jittable solver with diagnostics at the final (unaveraged) iterate."""
    result = _bridge_iterate(
        posterior_log_ratios,
        proposal_log_ratios,
        initial_log_evidence=initial_log_evidence,
        rtol=rtol,
        max_iterations=max_iterations,
        progress_every=progress_every,
        iteration_offset=iteration_offset,
        report_limit=report_limit,
        geometric_restart=geometric_restart,
        restart_interval=restart_interval,
        previous_log_evidence=previous_log_evidence,
    )
    statistics = _bridge_statistics(
        result.log_evidence,
        posterior_log_ratios,
        proposal_log_ratios,
        result.converged,
        posterior_iat,
        estimate_iat=estimate_iat,
    )
    return BridgeResult(
        *statistics,
        result.iterations,
        result.converged,
        result.log_evidence_history,
    )


_BRIDGE_CHUNK_ITERATIONS = 1000


@partial(jax.jit, static_argnames=("average_iterations",))
def _average_log_evidence(result, *, average_iterations):
    count = result.iterations
    window = jnp.minimum(count, average_iterations)
    indices = jnp.arange(result.log_evidence_history.size)
    retained = (indices > count - window) & (indices <= count)
    values = jnp.where(retained, result.log_evidence_history, 0.0)
    mean = jnp.sum(values) / jnp.maximum(window, 1)
    # Preserve numerical failures rather than averaging them away.
    return jnp.where((window > 0) & jnp.all(jnp.isfinite(values)), mean, result.log_evidence)


def _finalize_bridge(
    result, posterior_ratios, proposal_ratios, *, average_iterations, posterior_iat, estimate_iat
):
    # History length varies in uncapped runs. Keep it out of the statistics JIT
    # so changing the convergence count does not recompile the FFT/reductions.
    log_z = _average_log_evidence(result, average_iterations=average_iterations)
    statistics = _bridge_statistics(
        log_z,
        posterior_ratios,
        proposal_ratios,
        result.converged,
        posterior_iat,
        estimate_iat=estimate_iat,
    )
    return BridgeResult(
        *statistics,
        result.iterations,
        result.converged,
        result.log_evidence_history,
    )


def _average_bridge_result(
    result,
    posterior_ratios,
    proposal_ratios,
    *,
    average_iterations,
    posterior_iat,
    progress_every,
    geometric_restart=True,
    restart_interval=1000,
):
    if geometric_restart:
        # A successful midpoint restart may converge in one update. Do not
        # contaminate its reported estimate with the old oscillating branches.
        count = int(result.iterations)
        since_restart = (max(count, 1) - 1) % restart_interval + 1
        average_iterations = min(average_iterations, since_restart)
    averaged = _finalize_bridge(
        result,
        posterior_ratios,
        proposal_ratios,
        average_iterations=average_iterations,
        posterior_iat=1.0 if posterior_iat is None else posterior_iat,
        estimate_iat=posterior_iat is None,
    )
    if progress_every:
        jax.effects_barrier()
        window = min(int(result.iterations), average_iterations)
        if not emit_progress(
            "Bridge iterations",
            int(result.iterations),
            reported_log_z=float(averaged.log_evidence),
            averaging_window=window,
            final_log_z=float(result.log_evidence),
        ):
            print(
                f"    bridge reported logZ={float(averaged.log_evidence):.8f}: "
                f"mean of last {window} iterations; "
                f"final iterate={float(result.log_evidence):.8f}",
                flush=True,
            )
    return averaged


@jax.jit
def _density_validity(target_p, proposal_p, target_g, proposal_g):
    return jnp.stack(
        (
            jnp.all(jnp.isfinite(target_p))
            & jnp.all(jnp.isfinite(proposal_p))
            & jnp.all(jnp.isfinite(proposal_g)),
            jnp.any(jnp.isnan(target_g) | jnp.isposinf(target_g)),
            jnp.any(jnp.isfinite(target_g)),
        )
    )


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
    geometric_restart=True,
    restart_interval=1000,
):
    """Estimate log Z from equal-weight posterior and independent proposal draws.

    Supply four vectors of evaluated log densities; the target is likelihood
    times a proper prior, including every normalization constant. A supplied
    posterior_iat must be >= 1; None estimates it from the ordered bridge f2
    sequence. Relative error is a conditional Monte Carlo diagnostic, not a
    guarantee of posterior coverage or an assessment of fitting uncertainty.
    By default, after each restart_interval (1000) unconverged updates, restart
    at the mean of the last two log-Z values (the geometric mean on the Z scale).
    Set geometric_restart=False for uninterrupted standard bridge iteration.
    restart_interval must be at least two. This repeats the fallback published
    by Gronau, Heathcote & Matzke (2020), doi:10.3758/s13428-019-01290-6.
    Stop when abs(F(log_Z) - log_Z) <= rtol for an ordinary bridge update F,
    including the first update from a restart. A jump cannot itself converge.
    By default there is no iteration cap; max_iterations caps the total number
    of bridge updates across all restarts. Uncapped runs use
    compiled chunks and preserve the full history across chunk boundaries.
    Report the arithmetic mean of the last average_iterations computed log-Z
    values (default 5), excluding initialization and values before the latest
    restart. History retains every bridge update, without inserting midpoint
    seeds; iterations counts updates only. final_log_evidence is the last update.
    """
    arrays = tuple(
        jnp.asarray(value)
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
    finite_required, invalid_target, overlap = np.asarray(_density_validity(*arrays))
    if not finite_required:
        raise ValueError("posterior target values and all proposal density values must be finite")
    if invalid_target:
        raise ValueError("proposal target values may be finite or -inf, never NaN or +inf")
    if not overlap:
        raise ValueError("no proposal draws overlap target support; evidence is not identifiable")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    if max_iterations is not None:
        positive_integer(max_iterations, "max_iterations")
    validate_progress_every(progress_every)
    positive_integer(average_iterations, "average_iterations")
    if not isinstance(geometric_restart, (bool, np.bool_)):
        raise ValueError("geometric_restart must be a boolean")
    positive_integer(restart_interval, "restart_interval")
    if restart_interval < 2:
        raise ValueError("restart_interval must be at least two")
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
        if not emit_progress("Bridge iterations", 0, max_iterations, log_z=float(initial)):
            print(
                f"    bridge iteration 0{limit}: logZ={float(initial):.8f}",
                flush=True,
            )
    options = dict(
        rtol=rtol,
        progress_every=progress_every,
        geometric_restart=bool(geometric_restart),
        restart_interval=restart_interval,
    )
    if max_iterations is not None:
        result = _bridge_iterate(
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
            geometric_restart=geometric_restart,
            restart_interval=restart_interval,
        )
    histories = [initial.reshape(1)]
    iterations = 0
    previous = initial
    while True:
        result = _bridge_iterate(
            posterior_ratios,
            proposal_ratios,
            initial_log_evidence=initial,
            max_iterations=_BRIDGE_CHUNK_ITERATIONS,
            iteration_offset=iterations,
            report_limit=False,
            previous_log_evidence=previous,
            **options,
        )
        completed = int(result.iterations)
        histories.append(result.log_evidence_history[1 : completed + 1])
        iterations += completed
        if (
            bool(result.converged)
            or not np.isfinite(float(result.log_evidence))
            or completed < _BRIDGE_CHUNK_ITERATIONS
        ):
            result = result._replace(
                iterations=jnp.asarray(iterations),
                log_evidence_history=jnp.concatenate(histories),
            )
            return _average_bridge_result(
                result,
                posterior_ratios,
                proposal_ratios,
                average_iterations=average_iterations,
                posterior_iat=posterior_iat,
                progress_every=progress_every,
                geometric_restart=geometric_restart,
                restart_interval=restart_interval,
            )
        previous = result.log_evidence_history[completed - 1]
        initial = result.log_evidence
