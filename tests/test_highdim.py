"""Independent density/Jacobian checks and native MorphZ workflow checks."""

import json
from argparse import Namespace

import jax
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from benchmarks.highdim_prepare import prepare_dataset
from benchmarks.highdim_suite import run_case, sgm_seed_count
from benchmarks.highdim_target import SHAPES, load_target, make_random_target


@pytest.mark.parametrize("shape", SHAPES)
def test_exact_density_against_independent_change_of_variables(shape):
    target = make_random_target(5, seed=4, orders=[5], shapes=[shape])
    block = target.factors[0]
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(31, 5))
    transformed = latent.copy()
    jacobian = np.full(len(latent), np.log(block.scale).sum())
    if shape == "banana":
        transformed[:, 1:] += block.curvature[1:] * (latent[:, :1] ** 2 - 1)
    elif shape == "sinh_arcsinh":
        argument = (np.arcsinh(latent) + block.skew) / block.tail
        transformed = np.sinh(argument)
        jacobian += (
            np.log(np.cosh(argument)) - np.log(block.tail) - 0.5 * np.log1p(latent**2)
        ).sum(axis=1)
    physical = block.location + block.scale * transformed
    if shape in ("mixture", "parity"):
        expected = (
            logsumexp(
                np.column_stack(
                    [
                        stats.multivariate_normal.logpdf(latent, mean, 0.45**2 * block.correlation)
                        + np.log(probability)
                        for mean, probability in zip(block.means, block.probabilities)
                    ]
                ),
                axis=1,
            )
            - jacobian
        )
    else:
        expected = (
            stats.multivariate_normal.logpdf(latent, np.zeros(5), block.correlation) - jacobian
        )
    np.testing.assert_allclose(block.logpdf(physical), expected, atol=1e-10)


def test_target_covers_orders_reproducibility_prior_and_saved_roundtrip(tmp_path):
    target = make_random_target(200, seed=13)
    assert set(map(len, target.blocks)) == {1, 2, 3, 4, 5}
    assert sorted(i for b in target.blocks for i in b) == list(range(200))
    points = target.sample(np.random.default_rng(4), 40)
    np.testing.assert_array_equal(points, target.sample(np.random.default_rng(4), 40))
    np.testing.assert_allclose(target.logpdf(points), target.block_logpdf(points).sum(axis=1))
    np.testing.assert_allclose(
        target.log_likelihood(points) + target.log_prior(points),
        target.log_unnormalized(points),
        atol=1e-10,
    )
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target.description()))
    loaded = load_target(path)
    np.testing.assert_array_equal(loaded.logpdf(points), target.logpdf(points))


def test_parity_has_independent_proper_subsets():
    target = make_random_target(5, seed=7, orders=[5], shapes=["parity"])
    block = target.factors[0]
    # Marginalize the discrete mixture over the fifth coordinate. Four binary
    # signs are independent and uniformly weighted; component noise is diagonal.
    means = block.means[:, :4]
    assert len(np.unique(means, axis=0)) == 16
    x = np.array([0.2, -1.0, 0.7, 1.9])
    joint = logsumexp(
        [stats.norm.logpdf(x, mean, 0.45).sum() - np.log(len(means)) for mean in means]
    )
    product_logs = sum(
        logsumexp(stats.norm.logpdf(x[i], [-1.5, 1.5], 0.45)) - np.log(2) for i in range(4)
    )
    np.testing.assert_allclose(joint, product_logs, atol=1e-12)


def test_prepared_memory_mapped_data_has_correct_cached_density(tmp_path):
    target = make_random_target(20)
    manifest = prepare_dataset(target, tmp_path, samples=16, seed=8, chunk_size=7)
    assert manifest["complete"]
    points = np.load(tmp_path / "posterior_samples.npy", mmap_mode="r")
    cached = np.load(tmp_path / "posterior_log_density.npy")
    np.testing.assert_allclose(
        cached, load_target(tmp_path / "target.json").log_unnormalized(points)
    )
    with pytest.raises(ValueError, match="already exists"):
        prepare_dataset(target, tmp_path, samples=16)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("plots_only", [False, True])
def test_native_morph_bridge_workflow(tmp_path, prepared, plots_only, monkeypatch, capsys):
    from benchmarks import highdim_suite

    calls = []
    native_fit = highdim_suite.fit

    def counted_fit(*positional, **keyword):
        calls.append(keyword)
        return native_fit(*positional, **keyword)

    monkeypatch.setattr(highdim_suite, "fit", counted_fit)
    jax.config.update("jax_enable_x64", True)
    args = Namespace(
        dataset=None,
        target_seed=2,
        target_orders=[1, 2, 3, 4, 5],
        shapes=["gaussian"],
        log_evidence=-7.0,
        fit_samples=64,
        posterior_samples=64,
        proposals=64,
        backend="cpu",
        morph_order=2,
        n_seeds=4,
        tc_samples=32,
        bandwidth_scale=1.0,
        regularization=0.0,
        query_chunk_size=32,
        candidate_batch_size=8,
        seed_batch_size=4,
        sgm_early_stop=True,
        proposal_chunk_size=32,
        plot_samples=32,
        rtol=1e-6,
        max_iterations=1000,
        no_plots=True,
        plots_only=plots_only,
        progress_reports=0 if prepared and plots_only else 20,
    )
    if prepared:
        args.dataset = tmp_path / "data"
        target = make_random_target(
            20, seed=2, orders=[1, 2, 3, 4, 5], shapes=["gaussian"], log_evidence=-7.0
        )
        prepare_dataset(target, args.dataset, samples=128, seed=9, chunk_size=64)
    fit_cache = {}
    result = run_case(args, 20, 9, tmp_path, fit_cache=fit_cache)
    assert result["complete"] and result["candidate_count"] == 190
    assert result["seed_count"] == 4 and result["tc_sample_count"] == 32
    assert len(result["blocks"]) == 10 and all(len(b) == 2 for b in result["blocks"])
    assert result["analytic_control"]["log_evidence_error"] == pytest.approx(0, abs=1e-10)
    assert np.isfinite(result["log_evidence"]) and np.isfinite(result["forward_kl"])
    assert abs(result["log_evidence_error"]) < 1
    assert result["target_evaluations"] == 64
    assert result["total_runtime_seconds"] >= result["workflow_seconds"]
    assert result["proposal_survived_count"] == result["proposal_count"]
    assert result["proposal_survival_fraction"] == 1
    summary = (tmp_path / "summary.txt").read_text()
    assert "KL(Post || Morph):" in summary
    assert "Survived proposals (finite target density): 64/64 (100.0%)" in summary
    assert "Parameters in known dependent posterior components:" in summary
    assert (tmp_path / "metrics.json").exists() is not plots_only
    assert (tmp_path / "target.json").exists() is not plots_only
    captured = capsys.readouterr()
    output = captured.out
    assert "total runtime=" in output and "log Z=" in output
    assert "fit completed" in output
    if args.progress_reports:
        for label in ("candidate scores", "SGM seeds", "Proposal batches", "Bridge"):
            assert label in captured.err
    else:
        assert not captured.err
    assert "proposals 32/64" not in output
    assert bool(calls[0]["candidate_progress_every"]) == bool(args.progress_reports)
    assert bool(calls[0]["seed_progress_every"]) == bool(args.progress_reports)
    if plots_only:
        assert "KL=" not in output
    args.proposals = 32
    second = run_case(args, 20, 9, tmp_path, fit_cache=fit_cache)
    assert len(calls) == 1 and second["fit_reused"]
    assert second["fit_seconds"] == 0 and second["morph_seconds"] == 0
    assert second["original_fit_seconds"] == result["fit_seconds"]
    assert second["training_sha256"] == result["training_sha256"]
    assert second["posterior_sha256"] == result["posterior_sha256"]
    assert second["blocks"] == result["blocks"]
    assert second["proposal_count"] == 32 and second["posterior_count"] == 64


def test_benchmark_sgm_five_seed_default_and_explicit_override():
    assert sgm_seed_count(200, 2) == 5
    assert sgm_seed_count(500, 2) == 5
    assert sgm_seed_count(500, 3) == 5
    assert sgm_seed_count(500, 2, requested=4) == 4
    assert sgm_seed_count(500, 2, requested=1) == 1
    assert sgm_seed_count(500, 2, requested=400) == 400
    assert sgm_seed_count(3, 1) == 3
    assert sgm_seed_count(8, 1) == 5
    assert sgm_seed_count(9, 1) == 5
    assert sgm_seed_count(16, 1) == 5
    assert sgm_seed_count(17, 1) == 5
    assert sgm_seed_count(1, 1, requested=10) == 1


def test_runner_allows_more_than_previous_candidate_cap(tmp_path, monkeypatch):
    from benchmarks import highdim_suite

    cases = []

    def capture_case(args, dimension, seed, directory, fit_cache=None):
        cases.append(dimension)
        return dict(complete=True)

    monkeypatch.setattr(highdim_suite, "run_case", capture_case)
    monkeypatch.setattr(
        "sys.argv", ["highdim_suite", "--dimensions", "500", "--output", str(tmp_path)]
    )
    highdim_suite.main()
    assert cases == [500]  # 124,750 pair candidates exceed the previous library cap.
