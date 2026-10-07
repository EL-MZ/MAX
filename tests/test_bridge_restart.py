"""Geometric restarts resolve oscillation without changing the bridge equation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from benchmarks.highdim_suite import bridge_record
from jax_morph import bridge as bridge_module
from jax_morph import bridge_sampling


def _update(value, posterior, proposal):
    """Independent ordinary bridge update, using unnormalized log ratios."""
    fraction = len(posterior) / (len(posterior) + len(proposal))
    numerator = proposal - np.logaddexp(np.log(fraction) + proposal, np.log1p(-fraction) + value)
    denominator = -np.logaddexp(np.log(fraction) + posterior, np.log1p(-fraction) + value)
    return (
        logsumexp(numerator)
        - np.log(len(proposal))
        - (logsumexp(denominator) - np.log(len(posterior)))
    )


@pytest.mark.parametrize("offset", [-34000.0, 100000.0])
def test_default_restart_resolves_two_cycle_and_excludes_old_average(offset):
    zeros = np.zeros(8)
    options = dict(
        initial_log_evidence=offset + 10, max_iterations=1001, posterior_iat=1.0, rtol=1e-9
    )
    standard = bridge_sampling(
        zeros + offset + 100, zeros, zeros + offset - 100, zeros, geometric_restart=False, **options
    )
    restarted = bridge_sampling(zeros + offset + 100, zeros, zeros + offset - 100, zeros, **options)
    assert not standard.converged
    assert restarted.converged
    assert int(restarted.iterations) == 1001
    np.testing.assert_array_equal(
        restarted.log_evidence_history[:1001], standard.log_evidence_history[:1001]
    )
    assert float(restarted.final_log_evidence) == pytest.approx(offset, abs=1e-9)
    assert float(restarted.log_evidence) == pytest.approx(offset, abs=1e-9)
    assert bridge_record(restarted, offset)["log_evidence_averaging_iterations"] == 1
    assert (
        bridge_record(standard, offset, geometric_restart=False)[
            "log_evidence_averaging_iterations"
        ]
        == 5
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7])
def test_repeated_restarts_are_independent_of_compiled_chunks(chunk_size, monkeypatch):
    posterior, proposal = np.full(8, 2.0), np.full(11, -2.0)
    zeros_p, zeros_g = np.zeros_like(posterior), np.zeros_like(proposal)
    options = dict(initial_log_evidence=10.0, restart_interval=4, rtol=1e-10, posterior_iat=1.0)
    reference = bridge_sampling(
        posterior, zeros_p, proposal, zeros_g, max_iterations=200, **options
    )
    monkeypatch.setattr(bridge_module, "_BRIDGE_CHUNK_ITERATIONS", chunk_size)
    actual = bridge_sampling(posterior, zeros_p, proposal, zeros_g, **options)
    assert actual.converged and reference.converged
    count = int(actual.iterations)
    assert count > 8  # Exercise more than one restart.
    assert count == int(reference.iterations)
    np.testing.assert_allclose(
        actual.log_evidence_history, reference.log_evidence_history[: count + 1], atol=1e-12
    )
    expected = [10.0]
    for index in range(count):
        seed = np.mean(expected[-2:]) if index and index % 4 == 0 else expected[-1]
        expected.append(_update(seed, posterior, proposal))
    np.testing.assert_allclose(actual.log_evidence_history, expected, atol=1e-12)
    last_restart = ((count - 1) // 4) * 4
    window = min(5, count - last_restart)
    assert float(actual.log_evidence) == pytest.approx(np.mean(expected[-window:]), abs=1e-12)
    assert (
        abs(
            _update(float(actual.final_log_evidence), posterior, proposal)
            - float(actual.final_log_evidence)
        )
        < 1e-10
    )


def test_total_cap_and_restart_do_not_claim_false_convergence(capsys):
    zeros = np.zeros(8)
    options = dict(initial_log_evidence=10.0, restart_interval=2, rtol=1e-12, posterior_iat=1.0)
    capped = bridge_sampling(
        zeros + 2, zeros, zeros - 2, zeros, max_iterations=3, progress_every=1, **options
    )
    jax.effects_barrier()
    assert int(capped.iterations) == 3
    assert not capped.converged
    assert np.isnan(float(capped.relative_error))
    assert float(capped.log_evidence) == float(capped.final_log_evidence)
    history = np.asarray(capped.log_evidence_history)
    seed = history[1:3].mean()
    assert history[3] == pytest.approx(_update(seed, zeros + 2, zeros - 2), abs=1e-12)
    assert abs(history[3] - seed) > options["rtol"]
    output = capsys.readouterr().out
    assert "geometric restart after 2 updates" in output
    assert output.count("bridge finished") == 1
    before_restart = bridge_sampling(
        zeros + 2, zeros, zeros - 2, zeros, max_iterations=2, **options
    )
    np.testing.assert_array_equal(before_restart.log_evidence_history, history[:3])


def test_standard_opt_out_matches_independent_iteration():
    posterior, proposal = np.full(8, 2.0), np.full(11, -2.0)
    result = bridge_sampling(
        posterior,
        np.zeros_like(posterior),
        proposal,
        np.zeros_like(proposal),
        initial_log_evidence=10.0,
        geometric_restart=False,
        restart_interval=2,
        max_iterations=12,
        rtol=1e-12,
        posterior_iat=1.0,
    )
    expected = [10.0]
    for _ in range(12):
        expected.append(_update(expected[-1], posterior, proposal))
    np.testing.assert_allclose(result.log_evidence_history, expected, atol=1e-12)
    assert float(result.log_evidence) == pytest.approx(np.mean(expected[-5:]), abs=1e-12)


@pytest.mark.parametrize("geometric_restart", [True, False])
def test_jittable_kernel_supports_restart_and_standard_modes(geometric_restart):
    result = bridge_module.bridge_kernel(
        jnp.full(8, 100.0),
        jnp.full(8, -100.0),
        initial_log_evidence=jnp.asarray(10.0),
        geometric_restart=geometric_restart,
        restart_interval=2,
        max_iterations=3,
        rtol=1e-9,
    )
    assert bool(result.converged) == geometric_restart
    if geometric_restart:
        assert float(result.log_evidence) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("interval", [0, 1, -1, True, 2.5])
def test_invalid_restart_interval(interval):
    zeros = np.zeros(4)
    with pytest.raises(ValueError, match="restart_interval"):
        bridge_sampling(zeros, zeros, zeros, zeros, restart_interval=interval)


@pytest.mark.parametrize("enabled", [0, 1, "false", None])
def test_invalid_restart_switch(enabled):
    zeros = np.zeros(4)
    with pytest.raises(ValueError, match="geometric_restart"):
        bridge_sampling(zeros, zeros, zeros, zeros, geometric_restart=enabled)
