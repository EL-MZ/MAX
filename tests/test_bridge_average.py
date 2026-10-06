"""Recent-iteration reporting preserves solver state and uses log-space means."""

import numpy as np
import pytest

from jax_morph import bridge as bridge_module
from jax_morph import bridge_sampling


@pytest.mark.parametrize("cap", [1, 3, 7, None])
def test_reported_mean_uses_last_computed_iterates_without_initialization(cap, monkeypatch):
    zeros = np.zeros(8)
    options = dict(initial_log_evidence=10.0, rtol=1e-6, posterior_iat=1.0, max_iterations=cap)
    # A small chunk size also tests averaging across chunk boundaries.
    monkeypatch.setattr(bridge_module, "_BRIDGE_CHUNK_ITERATIONS", 3)
    final = bridge_sampling(zeros + 2, zeros, zeros - 2, zeros, average_iterations=1, **options)
    averaged = bridge_sampling(zeros + 2, zeros, zeros - 2, zeros, **options)
    count = int(averaged.iterations)
    history = np.asarray(averaged.log_evidence_history)
    computed = history[1 : count + 1]
    assert float(averaged.log_evidence) == pytest.approx(computed[-5:].mean(), abs=1e-12)
    assert float(averaged.final_log_evidence) == pytest.approx(float(final.log_evidence), abs=1e-12)
    assert count == int(final.iterations)
    assert bool(averaged.converged) == bool(final.converged)
    np.testing.assert_array_equal(averaged.log_evidence_history, final.log_evidence_history)
    if cap is not None and count > 1:
        assert not np.isclose(float(averaged.log_evidence), float(final.log_evidence))
    # Verify diagnostics use the reported mean, rather than silently retaining
    # diagnostics at the final iterate. A zero-step kernel evaluates only these.
    reference = bridge_module.bridge_kernel(
        zeros + 2,
        zeros - 2,
        initial_log_evidence=averaged.log_evidence,
        max_iterations=0,
        posterior_iat=1.0,
    )
    np.testing.assert_allclose(averaged.posterior_overlap, reference.posterior_overlap)
    np.testing.assert_allclose(averaged.proposal_overlap, reference.proposal_overlap)
    np.testing.assert_allclose(averaged.importance_ess, final.importance_ess)
    if not averaged.converged:
        assert np.isnan(float(averaged.relative_error))


def test_explicit_averaging_window_override():
    zeros = np.zeros(8)
    result = bridge_sampling(
        zeros + 2,
        zeros,
        zeros - 2,
        zeros,
        initial_log_evidence=10.0,
        max_iterations=7,
        rtol=1e-12,
        posterior_iat=1.0,
        average_iterations=2,
    )
    np.testing.assert_allclose(
        result.log_evidence, np.asarray(result.log_evidence_history)[-2:].mean()
    )


@pytest.mark.parametrize("window", [0, -1, False])
def test_averaging_window_must_be_positive(window):
    zeros = np.zeros(4)
    with pytest.raises(ValueError, match="average_iterations"):
        bridge_sampling(zeros, zeros, zeros, zeros, average_iterations=window)
