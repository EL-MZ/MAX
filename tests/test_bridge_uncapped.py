"""Uncapped bridge continuation retains stopping rules, history and progress."""

import re

import jax
import numpy as np
import pytest

from jax_morph import bridge as bridge_module
from jax_morph import bridge_sampling


def test_uncapped_continues_chunks_and_preserves_complete_history(monkeypatch, capsys):
    # Deliberately slow fixed-point convergence; force several compiled chunks.
    zeros = np.zeros(8)
    posterior = np.full(8, 2.0)
    proposal = np.full(8, -2.0)
    options = dict(initial_log_evidence=10.0, rtol=1e-8, posterior_iat=1.0)
    reference = bridge_sampling(posterior, zeros, proposal, zeros, max_iterations=1000, **options)
    capped = bridge_sampling(posterior, zeros, proposal, zeros, max_iterations=3, **options)
    assert reference.converged and int(reference.iterations) > 3
    assert not capped.converged and int(capped.iterations) == 3
    monkeypatch.setattr(bridge_module, "_BRIDGE_CHUNK_ITERATIONS", 3)
    actual = bridge_sampling(posterior, zeros, proposal, zeros, progress_every=7, **options)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert actual.converged
    assert int(actual.iterations) == int(reference.iterations)
    assert len(actual.log_evidence_history) == int(actual.iterations) + 1
    np.testing.assert_allclose(
        actual.log_evidence_history,
        reference.log_evidence_history[: int(reference.iterations) + 1],
        rtol=1e-12,
        atol=1e-12,
    )
    for field in reference._fields[:-1]:
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), rtol=1e-12)
    history = np.asarray(actual.log_evidence_history)
    assert abs(history[-1] - history[-2]) <= options["rtol"]
    assert abs(history[-2] - history[-3]) > options["rtol"]
    output = capsys.readouterr().out
    assert "(uncapped)" in output
    assert "stopped without convergence" not in output
    indices = [
        int(value.replace(",", "")) for value in re.findall(r"bridge iteration ([\d,]+)", output)
    ]
    assert indices == sorted(set(indices))
    assert indices[0] == 0 and indices[-1] == int(actual.iterations)
    assert output.count("bridge finished") == 1


def test_default_tolerance_is_point_one_absolute_change_in_log_evidence():
    zeros = np.zeros(4)
    result = bridge_sampling(
        zeros + 2.0,
        zeros,
        zeros + 2.0,
        zeros,
        initial_log_evidence=2.098,
        posterior_iat=1.0,
    )
    assert result.converged and int(result.iterations) == 1
    assert float(result.log_evidence) == pytest.approx(2.0)


@pytest.mark.parametrize("limit", [0, -1, False])
def test_explicit_iteration_cap_must_be_positive(limit):
    zeros = np.zeros(4)
    with pytest.raises(ValueError, match="max_iterations"):
        bridge_sampling(zeros, zeros, zeros, zeros, max_iterations=limit)
