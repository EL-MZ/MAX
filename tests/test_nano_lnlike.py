"""Validate the non-JAX evidence path used by the real NANOGrav script."""

import json
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest

import jax_morph
import nano_lnlike as nano


def gaussian_target(point):
    # Deliberately NumPy-only: this cannot be passed to estimate_evidence's JIT.
    return float(2.3 - np.log(2 * np.pi) - 0.5 * np.sum(np.asarray(point) ** 2))


@pytest.mark.parametrize("workers", [1, 2])
def test_external_target_bridge(workers):
    samples = np.random.default_rng(42).normal(size=(1200, 2))
    options = dict(n_proposal=1000, max_tc_samples=64, progress_every=0, rtol=1e-6)
    if workers == 1:
        result, fitted, proposals, densities = nano.jax_morphz(samples, gaussian_target, **options)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            result, fitted, proposals, densities = nano.jax_morphz(
                samples, gaussian_target, pool=pool, **options
            )
    assert bool(result.converged)
    assert abs(float(result.log_evidence) - 2.3) < 0.15
    assert proposals.shape == (1000, 2)
    np.testing.assert_allclose(
        densities["log_target_posterior"], [gaussian_target(row) for row in samples[600:]]
    )
    report = nano.minimal_report(result, fitted, samples, densities, 1.0)
    assert -0.15 < report["kl"] < 0.3


def test_cold_chain_and_matching_cached_rows(tmp_path):
    (tmp_path / "pars.txt").write_text("x\ny\n")
    cold = np.arange(60, dtype=float).reshape(10, 6)
    np.savetxt(tmp_path / "chain_1.0.txt", cold)
    np.savetxt(tmp_path / "chain_2.txt", cold + 1000)
    samples, cached, params = nano.read_chain(tmp_path, burn_in=1, thin=2, max_samples=4)
    assert params == ["x", "y"]
    np.testing.assert_array_equal(samples, cold[1::2][:4, :2])
    np.testing.assert_array_equal(cached, cold[1::2][:4, -4])
    (tmp_path / "chain_1.0.txt").unlink()
    with pytest.raises(ValueError, match="cold chain"):
        nano.read_chain(tmp_path)


def test_bounded_target_retains_outside_proposals():
    samples = np.random.default_rng(7).uniform(-1, 1, size=(1200, 1))

    def target(point):
        return -np.log(2) if abs(point[0]) <= 1 else -np.inf

    result, _, proposals, densities = nano.jax_morphz(
        samples,
        target,
        order=1,
        n_proposal=1000,
        posterior_log_density=np.full(len(samples), -np.log(2)),
        posterior_iat=1,
        progress_every=0,
        rtol=1e-6,
    )
    assert bool(result.converged)
    assert abs(float(result.log_evidence)) < 0.05
    assert len(proposals) == len(densities["log_target_proposal"]) == 1000
    outside = np.abs(proposals[:, 0]) > 1
    assert np.any(outside)
    assert np.all(np.isneginf(densities["log_target_proposal"][outside]))


def test_prior_support_skips_likelihood(monkeypatch):
    class OutsidePrior:
        def get_lnprior(self, theta):
            return -np.inf

        def get_lnlikelihood(self, theta):
            pytest.fail("Likelihood must not be called outside prior support")

    monkeypatch.setattr(nano, "pta_global", OutsidePrior())
    assert nano.ln_post(np.zeros(2)) == -np.inf


def test_align_parameters():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        nano.align_parameters(samples, ["y", "x"], ["x", "y"]), samples[:, ::-1]
    )
    with pytest.raises(ValueError, match="parameters differ"):
        nano.align_parameters(samples, ["y", "z"], ["x", "y"])
    with pytest.raises(ValueError, match="unique"):
        nano.align_parameters(samples, ["x", "x"], ["x", "y"])
    np.testing.assert_array_equal(
        nano.align_parameters(
            samples, ["gw_crn_log10_A", "gw_crn_gamma"], ["gamma_gw", "log10_A_gw"]
        ),
        samples[:, ::-1],
    )
    np.testing.assert_array_equal(
        nano.align_parameters(
            samples, ["gw_HD_log10_A", "gw_HD_gamma"], ["gamma_gw", "log10_A_gw"]
        ),
        samples[:, ::-1],
    )
    with pytest.raises(ValueError, match="parameters differ"):
        nano.align_parameters(
            samples, ["gw_HD_log10_A", "gw_HD_gamma"], ["gamma_gw", "log10_A_gw"], "curn"
        )


@pytest.mark.parametrize(
    "model, folder", [("irn", "irn_100k"), ("curn", "curn_100k"), ("HD", "HD_100k")]
)
def test_model_paths(model, folder):
    args = nano.parse_args(["--model", model])
    assert args.model == model.lower()
    assert args.chain_dir == nano.SOURCE_DIR / folder
    assert args.output.name == f"nanograv-{model.lower()}-jax"
    custom = nano.parse_args(["--model", model, "--chain-dir", "custom", "--output", "out"])
    assert str(custom.chain_dir) == "custom"
    assert str(custom.output) == "out"


def test_worker_uses_selected_model(monkeypatch):
    calls = []
    sentinel = object()

    def build(datadir, model_type):
        calls.append((datadir, model_type))
        return sentinel

    monkeypatch.setattr(nano, "_build_pta", build)
    monkeypatch.setattr(nano, "pta_global", None)
    nano._worker_init("data", "hd")
    assert calls == [("data", "hd")]
    assert nano.pta_global is sentinel


def test_zero_overlap_does_not_save_densities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples = np.random.default_rng(8).normal(size=(20, 1))
    with pytest.raises(ValueError, match="no proposal draws overlap"):
        nano.jax_morphz(
            samples,
            lambda point: -np.inf,
            order=1,
            n_proposal=8,
            posterior_log_density=np.zeros(len(samples)),
            progress_every=0,
        )
    assert not list(tmp_path.iterdir())


def test_repeated_estimates_reuse_fit_and_skip_plots(tmp_path, monkeypatch):
    samples = np.random.default_rng(32).normal(size=(600, 2))
    names = ["x", "y"]
    output = tmp_path / "out"
    calls = []
    fits = []
    original_fit = jax_morph.fit

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        fits.append(result)
        return result

    def target(point):
        calls.append(point.copy())
        return gaussian_target(point)

    monkeypatch.setattr(nano, "read_chain", lambda *a, **kw: (samples, None, names))
    monkeypatch.setattr(nano, "_build_pta", lambda *a: SimpleNamespace(param_names=names))
    monkeypatch.setattr(nano, "pta_global", None)
    monkeypatch.setattr(nano, "ln_post", target)
    monkeypatch.setattr(jax_morph, "fit", fit)
    nano.main(
        [
            "--workers",
            "1",
            "--no-plots",
            "--n-estimates",
            "2",
            "--n-proposal",
            "600",
            "--output",
            str(output),
            "--progress-every",
            "0",
            "--rtol",
            "1e-6",
        ]
    )
    assert len(fits) == 1
    assert len(calls) == 300 + 2 * 600
    np.testing.assert_array_equal(calls[:300], samples[300:])
    assert not np.array_equal(calls[300:900], calls[900:])
    assert [path.name for path in output.iterdir()] == ["result.json"]
    result = json.loads((output / "result.json").read_text())
    assert result["n_estimates"] == result["n_converged"] == 2
    assert [report["estimate"] for report in result["estimates"]] == [1, 2]
    for report in result["estimates"]:
        assert report["converged"]
        assert report["log_evidence"] == pytest.approx(2.3, abs=0.15)
    first, second = [report["log_evidence"] for report in result["estimates"]]
    assert result["log_evidence_mean"] == pytest.approx((first + second) / 2)
    assert result["log_evidence_std"] == pytest.approx(abs(first - second) / np.sqrt(2))


def test_three_12d_corner_plots(tmp_path):
    rng = np.random.default_rng(11)
    posterior = rng.normal(size=(400, 36))
    proposals = rng.normal(size=(700, 36))
    paths = nano.plot_corner_chunks(
        posterior, proposals, [f"parameter_{i}" for i in range(36)], tmp_path
    )
    assert [path.name for path in paths] == [f"corner_{i:02d}_12d.png" for i in range(1, 4)]
    assert all(path.stat().st_size > 10000 for path in paths)
    assert not list(tmp_path.glob("*.npz"))
