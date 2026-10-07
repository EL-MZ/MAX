"""Live tqdm rendering keeps the native computation and callback lifetimes intact."""

import jax
import numpy as np
import pytest

from benchmarks.highdim_progress import tqdm_progress
from jax_morph import bridge_sampling, fit
from jax_morph._progress import emit_progress, progress_renderer


def test_tqdm_fit_progress_uses_current_renderer_on_cached_kernels(capsys):
    data = np.random.default_rng(31).normal(size=(32, 7))
    options = dict(
        n_seeds=3,
        candidate_batch_size=4,
        seed_batch_size=2,
        candidate_progress_every=5,
        seed_progress_every=2,
    )
    results = []
    for _ in range(2):
        with tqdm_progress():
            results.append(fit(data, **options))
        captured = capsys.readouterr()
        assert not captured.out
        assert "candidate scores" in captured.err and "21/21" in captured.err
        assert "SGM seeds" in captured.err and "3/3" in captured.err
    assert results[0].state.blocks == results[1].state.blocks
    for actual, expected in zip(results[0].diagnostics, results[1].diagnostics):
        np.testing.assert_allclose(actual, expected)
    assert not emit_progress("after scope", 1, 1)


@pytest.mark.parametrize("limit", [7, None])
def test_bridge_bar_ends_at_actual_convergence_iteration(capsys, limit):
    zeros = np.zeros(4)
    options = dict(initial_log_evidence=5.0, max_iterations=limit, posterior_iat=1.0)
    reference = bridge_sampling(zeros + 2, zeros, zeros + 2, zeros, **options)
    with tqdm_progress():
        actual = bridge_sampling(zeros + 2, zeros, zeros + 2, zeros, progress_every=3, **options)
    for received, expected in zip(actual, reference):
        np.testing.assert_allclose(received, expected, equal_nan=True)
    captured = capsys.readouterr()
    assert not captured.out
    assert f"{int(actual.iterations)}/{int(actual.iterations)}" in captured.err
    assert "converged" in captured.err and "mean logZ" in captured.err


def test_progress_renderer_drains_callbacks_and_restores_on_exception():
    events = []
    with pytest.raises(RuntimeError, match="interrupted"):
        with progress_renderer(lambda *event: events.append(event)):
            jax.debug.callback(lambda: emit_progress("test", 1, 1), ordered=True)
            raise RuntimeError("interrupted")
    assert len(events) == 1 and events[0][:3] == ("test", 1, 1)
    assert not emit_progress("after scope", 1, 1)
