"""Sample-split MorphZ workflow: fit, propose, evaluate, bridge."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._validation import positive_integer, samples_and_weights
from .bridge import BridgeResult, bridge_sampling
from .morph import MorphFit, fit, log_prob, sample


class EvidenceResult(NamedTuple):
    bridge: BridgeResult
    morph: MorphFit
    proposal_samples: jax.Array
    fit_indices: jax.Array
    bridge_indices: jax.Array
    target_evaluations: int


def _evaluate(log_density, points, chunk_size):
    evaluate_batch = jax.jit(jax.vmap(log_density))
    results = [
        evaluate_batch(points[start : start + chunk_size])
        for start in (range(0, len(points), chunk_size))
    ]
    return jnp.concatenate(results)


def estimate_evidence(
    samples,
    log_density,
    key,
    *,
    n_proposal=None,
    fit_fraction=0.5,
    posterior_log_density=None,
    posterior_iat=None,
    rtol=1e-6,
    max_iterations=1000,
    evaluation_chunk_size=128,
    **fit_options,
):
    """MorphZ using equal-weight, post-burn-in posterior draws (n, d).

    ``log_density(point)`` must be JAX-compatible and return scalar log L+log pi.
    Split consecutive rows to preserve bridge-chain order for IAT estimation.
    Samples used to fit the proposal are never reused as bridge posterior draws.
    Weighted KDE fitting is supported by ``fit``; this evidence convenience API
    deliberately requires equal-weight posterior draws, e.g. resampled NS output.
    Cached ``posterior_log_density`` avoids evaluating the held-out posterior.
    """
    samples, _ = samples_and_weights(samples)
    if not np.isfinite(fit_fraction) or not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    split = int(len(samples) * fit_fraction)
    if split < 2 or len(samples) - split < 2:
        raise ValueError("both the fit and bridge posterior batches need at least two samples")
    positive_integer(evaluation_chunk_size, "evaluation_chunk_size")
    if "weights" in fit_options or "key" in fit_options:
        raise ValueError("pass equal-weight samples; the workflow manages its own PRNG keys")
    fit_key, proposal_key = jax.random.split(key)
    fitted = fit(samples[:split], key=fit_key, **fit_options)
    held_out = jnp.asarray(samples[split:])
    n_proposal = len(held_out) if n_proposal is None else positive_integer(n_proposal, "n_proposal")
    if n_proposal < 2:
        raise ValueError("at least two proposal draws are required")
    proposals = sample(proposal_key, fitted.state, n_proposal)
    if posterior_log_density is None:
        posterior_target = _evaluate(log_density, held_out, evaluation_chunk_size)
        evaluations = len(held_out) + n_proposal
    else:
        cached = np.asarray(posterior_log_density)
        if cached.shape != (len(samples),) or not np.isfinite(cached).all():
            raise ValueError("posterior_log_density must contain one finite value per sample")
        posterior_target = cached[split:]
        evaluations = n_proposal
    proposal_target = _evaluate(log_density, proposals, evaluation_chunk_size)
    posterior_proposal = log_prob(fitted.state, held_out, chunk_size=evaluation_chunk_size)
    proposal_proposal = log_prob(fitted.state, proposals, chunk_size=evaluation_chunk_size)
    bridge = bridge_sampling(
        posterior_target,
        posterior_proposal,
        proposal_target,
        proposal_proposal,
        posterior_iat=posterior_iat,
        rtol=rtol,
        max_iterations=max_iterations,
    )
    return EvidenceResult(
        bridge, fitted, proposals, jnp.arange(split), jnp.arange(split, len(samples)), evaluations
    )
