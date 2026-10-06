"""Progress reports preserve native numerical results and bound output volume."""

import builtins

import jax
import numpy as np
import pytest
from scipy.stats import norm

from jax_morph import bridge_sampling, fit, log_prob


def test_candidate_and_seed_progress_preserves_fit_and_flushes(capsys, monkeypatch):
    rng = np.random.default_rng(31)
    data = rng.normal(size=(48, 7))
    data[:, 1] += data[:, 0]
    options = dict(
        order=2,
        n_seeds=5,
        max_tc_samples=32,
        key=jax.random.key(2),
        candidate_batch_size=4,
        seed_batch_size=2,
        sgm_early_stop=True,
    )
    silent = fit(data, **options)
    jax.block_until_ready(silent)
    jax.effects_barrier()
    assert not capsys.readouterr().out
    messages = []
    original_print = builtins.print

    def record_print(*args, **kwargs):
        messages.append((args, kwargs))
        return original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", record_print)
    reported = fit(data, **options, candidate_progress_every=5, seed_progress_every=2)
    jax.block_until_ready(reported)
    jax.effects_barrier()
    assert reported.state.blocks == silent.state.blocks
    for actual, expected in zip(reported.diagnostics, silent.diagnostics):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(log_prob(reported.state, data), log_prob(silent.state, data))
    output = capsys.readouterr().out
    assert "candidate scores: 21/21" in output
    assert "SGM seeds: 5/5" in output
    assert len(output.splitlines()) <= 10
    assert messages and all(kwargs.get("flush") is True for _, kwargs in messages)


@pytest.mark.parametrize("exact", [False, True])
def test_bridge_progress_preserves_iteration_history_and_reports_finish(capsys, exact):
    points = np.linspace(-2, 2, 20)
    target_p, proposal_p = norm.logpdf(points), norm.logpdf(points, scale=2)
    target_g, proposal_g = norm.logpdf(2 * points), norm.logpdf(2 * points, scale=2)
    if exact:
        proposal_p, proposal_g = target_p, target_g
        target_p, target_g = target_p + 2.0, target_g + 2.0
    options = dict(max_iterations=7, rtol=1e-14, posterior_iat=1.0)
    if not exact:
        options["initial_log_evidence"] = 10.0
    silent = bridge_sampling(target_p, proposal_p, target_g, proposal_g, **options)
    jax.block_until_ready(silent)
    jax.effects_barrier()
    assert not capsys.readouterr().out
    reported = bridge_sampling(
        target_p, proposal_p, target_g, proposal_g, **options, progress_every=3
    )
    jax.block_until_ready(reported)
    jax.effects_barrier()
    for actual, expected in zip(reported, silent):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    output = capsys.readouterr().out
    iterations = int(reported.iterations)
    assert f"bridge iteration {iterations}/7" in output
    assert f"bridge finished at iteration {iterations}:" in output
    assert len(output.splitlines()) <= 7
    assert bool(reported.converged) == exact
