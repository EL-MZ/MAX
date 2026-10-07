"""Physical posterior correlation calibration preserves densities and evidence."""

import json

import numpy as np
import pytest
from scipy.special import polygamma

from benchmarks.correlation_levels import CORRELATION_RANGES, DEFAULT_CORRELATION_LEVELS
from benchmarks.highdim_plots import select_corner_parameters
from benchmarks.highdim_summary import posterior_dependence
from benchmarks.highdim_target import load_target, make_random_target
from benchmarks.regression_target import make_regression_target


@pytest.mark.parametrize("coordinate", ["variance", "log_variance"])
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_regression_population_correlations_and_bayes_identity(coordinate, order, tmp_path):
    original = make_regression_target(6 * order, seed=7, group_orders=[order])
    target = make_regression_target(
        6 * order,
        seed=7,
        group_orders=[order],
        variance_coordinate=coordinate,
        correlation_levels=DEFAULT_CORRELATION_LEVELS,
    )
    np.testing.assert_allclose(target.log_evidence, original.log_evidence, atol=1e-11, rtol=0)
    assert [g.correlation_level for g in target.factors] == DEFAULT_CORRELATION_LEVELS * 2
    for group in target.factors:
        ev = group.bn / (group.an - 1)
        vt = (
            group.bn**2 / ((group.an - 1) ** 2 * (group.an - 2))
            if coordinate == "variance"
            else float(polygamma(1, group.an))
        )
        covariance = np.empty((order, order))
        covariance[:-1, :-1] = ev * group.posterior_covariance + vt * np.outer(
            group.variance_shear, group.variance_shear
        )
        covariance[:-1, -1] = covariance[-1, :-1] = vt * group.variance_shear
        covariance[-1, -1] = vt
        std = np.sqrt(np.diag(covariance))
        correlations = covariance / std[:, None] / std[None, :]
        np.testing.assert_allclose(
            np.abs(correlations[np.triu_indices(order, 1)]),
            group.correlation_strength,
            atol=2e-12,
            rtol=0,
        )
    draws = target.sample(np.random.default_rng(5), 500)
    np.testing.assert_allclose(
        target.log_unnormalized(draws) - target.logpdf(draws),
        target.log_evidence,
        atol=1e-10,
    )
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target.description()))
    loaded = load_target(path)
    np.testing.assert_allclose(
        loaded.sample(np.random.default_rng(5), 500), draws, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        loaded.log_unnormalized(draws),
        target.log_unnormalized(draws),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("coordinate", ["variance", "log_variance"])
def test_regression_sampled_correlations_and_level_corners(coordinate):
    target = make_regression_target(
        60,
        seed=9,
        variance_coordinate=coordinate,
        correlation_levels=DEFAULT_CORRELATION_LEVELS,
    )
    truth = target.sample(np.random.default_rng(12), 30000)
    levels_by_index = {}
    for group in target.factors:
        correlations = np.corrcoef(truth[:, group.indices], rowvar=False)
        np.testing.assert_allclose(
            np.abs(correlations[np.triu_indices(5, 1)]),
            group.correlation_strength,
            atol=0.025,
        )
        levels_by_index.update({i: group.correlation_level for i in group.indices})
    selections = select_corner_parameters(target, truth)
    assert list(map(len, selections)) == [10, 10, 10]
    for indices, level in zip(selections, DEFAULT_CORRELATION_LEVELS):
        assert {levels_by_index[i] for i in indices} == {level}
    record = posterior_dependence(target, truth[:5000])
    assert record["correlation_level_parameter_counts"] == dict(low=20, medium=20, strong=20)
    assert record["linear_correlated_parameters"] >= 40


@pytest.mark.parametrize("coordinate", ["variance", "log_variance"])
def test_calibrated_cross_group_shear_preserves_evidence(coordinate):
    target = make_regression_target(
        60,
        seed=4,
        variance_coordinate=coordinate,
        cross_group_coupling=0.8,
        correlation_levels=DEFAULT_CORRELATION_LEVELS,
    )
    for a, b in target.coupling_pairs:
        assert target.factors[a].correlation_level == target.factors[b].correlation_level
    draws = target.sample(np.random.default_rng(12), 100)
    np.testing.assert_allclose(
        target.log_unnormalized(draws) - target.logpdf(draws),
        target.log_evidence,
        atol=1e-10,
    )


@pytest.mark.parametrize("shape", ["gaussian", "banana", "mixture", "sinh_arcsinh"])
def test_synthetic_physical_correlation_levels_and_affine_density(shape, tmp_path):
    original = make_random_target(15, seed=4, orders=[5], shapes=[shape])
    target = make_random_target(
        15,
        seed=4,
        orders=[5],
        shapes=[shape],
        correlation_levels=DEFAULT_CORRELATION_LEVELS,
    )
    rng_seed = 8
    draws = target.sample(np.random.default_rng(rng_seed), 30000)
    base = original.sample(np.random.default_rng(rng_seed), 30000)
    for block, base_block in zip(target.factors, original.factors):
        low, high = CORRELATION_RANGES[block.correlation_level]
        assert low <= block.correlation_strength <= high
        correlations = np.corrcoef(draws[:, block.indices], rowvar=False)
        np.testing.assert_allclose(
            np.abs(correlations[np.triu_indices(5, 1)]),
            block.correlation_strength,
            atol=0.025,
        )
        np.testing.assert_allclose(
            block.logpdf(draws[:50, block.indices]),
            base_block.logpdf(base[:50, block.indices])
            - np.linalg.slogdet(block.physical_transform)[1],
            atol=1e-10,
        )
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target.description()))
    loaded = load_target(path)
    np.testing.assert_array_equal(loaded.logpdf(draws[:100]), target.logpdf(draws[:100]))
    assert target.log_evidence == original.log_evidence


def test_parity_retains_pairwise_independence_and_singletons_are_not_calibrated():
    original = make_random_target(20, seed=4, orders=[1, 2, 5], shapes=["parity"])
    target = make_random_target(
        20,
        seed=4,
        orders=[1, 2, 5],
        shapes=["parity"],
        correlation_levels=["strong"],
    )
    for base, block in zip(original.factors, target.factors):
        if block.shape == "parity" or len(block.indices) == 1:
            assert block.physical_transform is None
            np.testing.assert_array_equal(block.correlation, base.correlation)


def test_invalid_correlation_levels_are_rejected():
    for factory in (make_regression_target, make_random_target):
        with pytest.raises(ValueError, match="correlation levels"):
            factory(20, correlation_levels=[])
        with pytest.raises(ValueError, match="correlation levels"):
            factory(20, correlation_levels=["extreme"])


@pytest.mark.parametrize("levels", [None, ["strong"], ["original"]])
def test_runner_correlation_levels_default_and_override(levels, tmp_path, monkeypatch):
    from benchmarks import highdim_suite

    cases = []

    def capture_case(args, dimension, seed, directory, fit_cache=None):
        cases.append(args.correlation_levels)
        return dict(complete=True)

    argv = ["highdim_suite", "--dimensions", "30", "--output", str(tmp_path)]
    if levels is not None:
        argv.extend(["--correlation-levels", *levels])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(highdim_suite, "run_case", capture_case)
    highdim_suite.main()
    assert cases == [levels or DEFAULT_CORRELATION_LEVELS]


def test_prepare_cli_stores_calibrated_levels_and_valid_densities(tmp_path, monkeypatch):
    from benchmarks import highdim_prepare

    monkeypatch.setattr(
        "sys.argv",
        [
            "highdim_prepare",
            "--dimensions",
            "30",
            "--samples",
            "16",
            "--chunk-size",
            "7",
            "--output",
            str(tmp_path),
        ],
    )
    highdim_prepare.main()
    directory = tmp_path / "30d"
    target = load_target(directory / "target.json")
    assert list(target.correlation_levels) == DEFAULT_CORRELATION_LEVELS
    draws = np.load(directory / "posterior_samples.npy")
    np.testing.assert_allclose(
        target.log_unnormalized(draws),
        np.load(directory / "posterior_log_density.npy"),
        atol=1e-11,
    )
