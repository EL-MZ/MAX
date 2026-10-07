"""Corner selection favors strong dependence without a full-target correlation matrix."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.highdim_plots import (
    absolute_correlation,
    plot_case,
    plot_evidence_history,
    select_corner_parameters,
)
from benchmarks.regression_target import make_regression_target


def test_three_disjoint_ten_parameter_corners_rank_positive_and_negative_correlations():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(1000, 40))
    blocks = [tuple(range(start, start + 10)) for start in range(0, 40, 10)]
    for block, noise in zip(blocks[1:], (0.02, 0.08, 0.2)):
        common = rng.normal(size=(len(truth), 1))
        signs = np.where(np.arange(10) % 2, -1, 1)
        truth[:, block] = common * signs + noise * rng.normal(size=(len(truth), 10))
    target = SimpleNamespace(blocks=blocks)
    selections = select_corner_parameters(target, truth)
    assert [set(indices) for indices in selections] == [set(block) for block in blocks[1:]]
    assert all(len(indices) == 10 for indices in selections)
    assert len(set(i for indices in selections for i in indices)) == 30


def test_coupled_regression_corners_keep_full_ten_parameter_components():
    target = make_regression_target(40, seed=2, cross_group_coupling=0.8)
    truth = target.sample(np.random.default_rng(3), 500)
    selections = select_corner_parameters(target, truth)
    assert len(selections) == 3
    assert all(set(indices) in [set(block) for block in target.blocks] for indices in selections)


def test_smaller_independent_groups_fill_panels_and_small_targets_do_not_repeat():
    target = SimpleNamespace(blocks=[tuple(range(i, i + 5)) for i in range(0, 35, 5)])
    truth = np.random.default_rng(5).normal(size=(100, 35))
    selections = select_corner_parameters(target, truth)
    assert list(map(len, selections)) == [10, 10, 10]
    for indices in selections:
        assert sum(set(block).issubset(indices) for block in target.blocks) == 2
    small = SimpleNamespace(blocks=[(0, 1, 2), (3, 4)])
    assert set(select_corner_parameters(small, truth[:, :5])[0]) == set(range(5))
    assert list(map(len, select_corner_parameters(small, truth[:, :5], dimensions=3))) == [3, 2]


def test_constant_columns_and_single_sample_have_finite_correlations():
    for points in (np.ones((10, 4)), np.ones((1, 4))):
        np.testing.assert_array_equal(absolute_correlation(points), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="positive"):
        select_corner_parameters(SimpleNamespace(blocks=[(0,)]), np.ones((5, 1)), count=0)


def test_plot_case_writes_three_full_corners_and_records_coordinates(tmp_path):
    target = make_regression_target(40, seed=2, cross_group_coupling=0.8)
    rng = np.random.default_rng(3)
    truth, proposals = target.sample(rng, 200), target.sample(rng, 200)
    metrics = dict(morph_order=2, history=[target.log_evidence], converged=True)
    plots = plot_case(target, truth, proposals, metrics, tmp_path)
    assert len(plots) == 3
    for number, record in enumerate(plots, 1):
        assert record["file"] == f"corner_{number:02d}_10d.png"
        assert (tmp_path / record["file"]).stat().st_size > 1000
        assert len(record["indices"]) == 10
        assert 0 <= record["mean_absolute_correlation"] <= record["max_absolute_correlation"] <= 1
    assert (tmp_path / "evidence_history.png").is_file()


@pytest.mark.parametrize("restart_enabled", [True, False])
def test_thinned_history_keeps_both_cycle_branches_and_overlays_all_restarts(
    tmp_path, monkeypatch, restart_enabled
):
    from matplotlib.axes import Axes

    updates, jumps = [], []
    original_plot, original_scatter = Axes.plot, Axes.scatter

    def capture_plot(self, x, y, **kwargs):
        if kwargs.get("label", "").startswith("MorphZ iteration"):
            updates.append((np.asarray(x), np.asarray(y), kwargs))
        return original_plot(self, x, y, **kwargs)

    def capture_scatter(self, x, y, **kwargs):
        jumps.append((np.asarray(x), np.asarray(y), kwargs))
        return original_scatter(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    history = np.where(np.arange(1002) % 2, -20.0, -10.0)
    metrics = dict(
        history=history, converged=False, geometric_restart=restart_enabled,
        bridge_restart_interval=100,
    )
    plot_evidence_history(SimpleNamespace(dimension=3000, log_evidence=-15.0), metrics, tmp_path)
    indices, values, style = updates[0]
    assert len(indices) == 200
    assert indices[0] == 0 and indices[-1] == len(history) - 1
    assert np.sum(indices % 2 == 0) == np.sum(indices % 2 == 1) == 100
    np.testing.assert_array_equal(values, history[indices])
    assert style["linestyle"] == "none"
    if restart_enabled:
        jump_indices, jump_values, jump_style = jumps[0]
        np.testing.assert_array_equal(jump_indices, np.arange(101, 1002, 100))
        np.testing.assert_array_equal(jump_values, history[jump_indices])
        assert jump_style["color"] == "red"
        assert jump_style["zorder"] > style["zorder"]
    else:
        assert jumps == []


@pytest.mark.parametrize("relative_error", [0.12, float("nan"), None])
def test_evidence_error_bar_is_only_at_the_last_actual_iteration(
    tmp_path, monkeypatch, relative_error
):
    from matplotlib.axes import Axes

    calls = []
    titles = []
    original = Axes.errorbar
    original_set_title = Axes.set_title

    def capture(self, x, y, **kwargs):
        calls.append((x, y, kwargs["yerr"]))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)

    def capture_title(self, title, **kwargs):
        titles.append(title)
        return original_set_title(self, title, **kwargs)

    monkeypatch.setattr(Axes, "set_title", capture_title)
    metrics = dict(
        history=[-12.0, -10.5, -10.1],
        converged=True,
        relative_error=relative_error,
        total_runtime_seconds=12.34,
    )
    plot_evidence_history(SimpleNamespace(dimension=1000, log_evidence=-10.0), metrics, tmp_path)
    assert calls == ([(2, -10.1, 0.12)] if relative_error == 0.12 else [])
    assert any("1,000 dimensions | Total runtime: 0:00:12" in title for title in titles)
    assert (tmp_path / "evidence_history.png").is_file()
