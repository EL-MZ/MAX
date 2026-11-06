"""NANOGrav IRN, CURN, or HD evidence with JAX MorphZ and Enterprise workers.

Adapted from /home/apokrypha/Local NT/Nanograv/nano_lnlike.py. The PTA model
follows gwb_morph.ipynb. Enterprise evaluates log likelihood + normalized log prior on
the host; JAX fits/samples the proposal and solves the bridge equation.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402 -- thread limits must precede NumPy initialization

pta_global = None
SOURCE_DIR = Path("/home/apokrypha/Local NT/Nanograv")
MODEL_CHAINS = {"irn": "irn_100k", "curn": "curn_100k", "hd": "HD_100k"}


def _build_pta(datadir: str, model_type="curn"):
    """Construct the notebook's IRN/CURN/HD PTA in parent or spawned worker."""
    if model_type not in MODEL_CHAINS:
        raise ValueError(f"Unknown PTA model: {model_type}")
    from enterprise.signals import (
        gp_signals,
        parameter,
        selections,
        signal_base,
        utils,
        white_signals,
    )
    from h5pulsar.pulsar import FilePulsar

    files = sorted(glob.glob(str(Path(datadir) / "hdf5" / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No pulsar HDF5 files in {Path(datadir) / 'hdf5'}")
    psrs = [FilePulsar(hdf5_file) for hdf5_file in files]
    with open(Path(datadir) / "15yr_wn_dict.json") as fp:
        wn_params = json.load(fp)

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    tspan = np.max(tmax) - np.min(tmin)
    selection = selections.Selection(selections.by_backend)
    efac = parameter.Constant()
    t2equad = parameter.Constant()
    ecorr = parameter.Constant()
    log10_a = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    mn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    pl = utils.powerlaw(log10_A=log10_a, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=tspan)
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    model = tm + mn + ec + rn
    if model_type != "irn":
        log10_a_gw = parameter.Uniform(-18, -14)("log10_A_gw")
        gamma_gw = parameter.Uniform(0, 7)("gamma_gw")
        cpl = utils.powerlaw(log10_A=log10_a_gw, gamma=gamma_gw)
        if model_type == "hd":
            common = gp_signals.FourierBasisCommonGP(
                cpl, orf=utils.hd_orf(), components=14, Tspan=tspan, name="gw"
            )
        else:
            common = gp_signals.FourierBasisGP(spectrum=cpl, components=14, Tspan=tspan, name="gw")
        model += common
    pta = signal_base.PTA([model(p) for p in psrs])
    pta.set_default_params(wn_params)
    return pta


def _worker_init(datadir: str, model_type="curn"):
    global pta_global
    pta_global = _build_pta(datadir, model_type)


def ln_post(theta):
    """Picklable host callback; avoid expensive likelihood outside prior support."""
    if pta_global is None:
        raise RuntimeError("PTA model not initialized")
    prior = float(pta_global.get_lnprior(theta))
    if np.isneginf(prior):
        return -np.inf
    if not np.isfinite(prior):
        raise ValueError(f"Invalid PTA log prior: {prior}")
    return float(pta_global.get_lnlikelihood(theta)) + prior


def read_chain(folder_path, *, burn_in=0, thin=20, max_samples=None):
    """Load the cold PTMCMC chain, preserving parameter and temporal order.

    PTMCMC appends four diagnostic columns; -4 is log posterior. Recompute
    targets by default because existing chains may use a different convention.
    """
    folder = Path(folder_path)
    params = np.atleast_1d(np.loadtxt(folder / "pars.txt", dtype=str)).tolist()
    # Temperature 1 is the posterior; never silently use a hot chain.
    files = [path for path in folder.glob("chain_*.txt") if float(path.stem[6:]) == 1.0]
    if len(files) != 1:
        raise ValueError(f"Expected one cold chain (chain_1.txt or chain_1.0.txt) in {folder}")
    if burn_in < 0 or thin < 1 or (max_samples is not None and max_samples < 4):
        raise ValueError("burn must be >= 0, thin >= 1, and max_samples >= 4")
    chain = np.loadtxt(files[0], ndmin=2)
    if chain.shape[1] != len(params) + 4:
        raise ValueError("Chain must contain len(pars.txt) parameter columns plus 4 diagnostics")
    chain = chain[burn_in::thin]
    if max_samples is not None:
        chain = chain[:max_samples]
    samples, cached = chain[:, :-4], chain[:, -4]
    if len(samples) < 4 or not np.all(np.isfinite(samples)):
        raise ValueError("Need at least four finite post-burn-in posterior samples")
    print(f"Loaded {files[0]}: {len(samples)} samples, {len(params)} parameters", flush=True)
    return samples, cached, params


def align_parameters(samples, params, model_names, model_type=None):
    """Map chain columns to PTA order, refusing missing or ambiguous names."""
    # Supplied CURN/HD chains use older names for the same two GW parameters.
    aliases = {
        "gw_crn_gamma": "gamma_gw",
        "gw_crn_log10_A": "log10_A_gw",
        "gw_HD_gamma": "gamma_gw",
        "gw_HD_log10_A": "log10_A_gw",
    }
    if model_type is not None:
        prefix = {"curn": "gw_crn_", "hd": "gw_HD_", "irn": ""}[model_type]
        aliases = {
            name: target for name, target in aliases.items() if prefix and name.startswith(prefix)
        }
    params = [aliases.get(name, name) if name not in model_names else name for name in params]
    if len(set(params)) != len(params) or len(set(model_names)) != len(model_names):
        raise ValueError("Parameter names must be unique")
    if set(params) != set(model_names):
        raise ValueError(
            "Chain and PTA parameters differ: "
            f"chain only={sorted(set(params) - set(model_names))}, "
            f"PTA only={sorted(set(model_names) - set(params))}"
        )
    if params != model_names:
        print("Reordering chain parameter columns to match PTA.param_names", flush=True)
    columns = {name: index for index, name in enumerate(params)}
    return samples[:, [columns[name] for name in model_names]]


def iter_morphz_estimates(
    samples,
    log_density,
    *,
    pool=None,
    posterior_log_density=None,
    n_proposal=20000,
    n_estimates=1,
    seed=0,
    order=2,
    n_seeds=5,
    max_tc_samples=100,
    chunk_size=128,
    progress_every=1000,
    posterior_iat=None,
    rtol=0.1,
    max_iterations=None,
    plot_output=None,
    param_names=None,
):
    """Fit once and yield bridge estimates using fresh draws from that fit."""
    # Lazy imports keep spawned likelihood workers from initializing JAX/GPU.
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax_morph import bridge_sampling, fit, log_prob, sample

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 4 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be a finite (n, d) array with n >= 4")
    if n_proposal < 2 or chunk_size < 1:
        raise ValueError("n_proposal must be >= 2 and chunk_size >= 1")
    if n_estimates < 1:
        raise ValueError("n_estimates must be >= 1")
    max_tc_samples = 200 if max_tc_samples is None else max_tc_samples
    if not 1 <= max_tc_samples <= 200:
        raise ValueError("max_tc_samples must be between 1 and 200")
    split = len(samples) // 2
    fit_key, proposal_key = jax.random.split(jax.random.PRNGKey(seed))
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Fitting Morph on {split} draws; holding out {len(samples) - split}", flush=True)
    fitted = fit(
        samples[:split],
        order=order,
        n_seeds=n_seeds,
        key=fit_key,
        max_tc_samples=max_tc_samples,
        chunk_size=chunk_size,
        candidate_progress_every=progress_every,
        seed_progress_every=progress_every,
    )
    held_out = samples[split:]

    def evaluate(points, label):
        values = np.empty(len(points))
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            batch = points[start:stop]
            values[start:stop] = (
                pool.map(log_density, batch) if pool is not None else list(map(log_density, batch))
            )
            print(f"{label}: {stop}/{len(points)} evaluated", flush=True)
        return values

    if posterior_log_density is None:
        posterior_target = evaluate(held_out, "Posterior target")
    else:
        cached = np.asarray(posterior_log_density, dtype=float)
        if cached.shape != (len(samples),) or not np.all(np.isfinite(cached)):
            raise ValueError("posterior_log_density must have one finite value per input row")
        posterior_target = cached[split:]
    posterior_proposal = log_prob(fitted.state, held_out, chunk_size=chunk_size)
    for index in range(n_estimates):
        if n_estimates > 1:
            print(f"Evidence estimate {index + 1}/{n_estimates}", flush=True)
        key = proposal_key if index == 0 else jax.random.fold_in(proposal_key, index)
        proposals = np.asarray(sample(key, fitted.state, n_proposal))
        if index == 0 and plot_output is not None:
            plot_corner_chunks(held_out, proposals, param_names, plot_output)
        proposal_target = evaluate(proposals, "Proposal target")
        inside = np.count_nonzero(np.isfinite(proposal_target))
        print(f"Proposals inside target support: {inside}/{n_proposal}", flush=True)
        # Keep out-of-prior draws with target=-inf; rejecting them biases evidence.
        proposal_proposal = log_prob(fitted.state, proposals, chunk_size=chunk_size)
        densities = {
            "log_target_posterior": posterior_target,
            "log_proposal_posterior": np.asarray(posterior_proposal),
            "log_target_proposal": proposal_target,
            "log_proposal_proposal": np.asarray(proposal_proposal),
        }
        try:
            bridge = bridge_sampling(
                posterior_target,
                posterior_proposal,
                proposal_target,
                proposal_proposal,
                posterior_iat=posterior_iat,
                rtol=rtol,
                max_iterations=max_iterations,
                progress_every=progress_every,
            )
        except ValueError as exc:
            if n_estimates == 1 or not str(exc).startswith("no proposal draws overlap"):
                raise
            print(f"Estimate {index + 1} failed: {exc}", flush=True)
            bridge = None
        yield bridge, fitted, proposals, densities


def jax_morphz(samples, log_density, **options):
    """Run the single-estimate adapter for non-JAX targets."""
    return next(iter_morphz_estimates(samples, log_density, n_estimates=1, **options))


def plot_corner_chunks(posterior, proposals, param_names, output):
    """Overlay three consecutive 12D chunks (first 36 parameters) as PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    dimension = posterior.shape[1]
    names = param_names if param_names is not None else [f"theta_{i}" for i in range(dimension)]
    for number, start in enumerate(range(0, min(dimension, 36), 12), 1):
        stop = min(start + 12, dimension)
        width = stop - start
        points = posterior[:, start:stop]
        draws = proposals[:, start:stop]
        low = np.minimum(points.min(axis=0), draws.min(axis=0))
        high = np.maximum(points.max(axis=0), draws.max(axis=0))
        padding = np.maximum(high - low, 1e-6) * 0.03
        ranges = list(zip(low - padding, high + padding))
        labels = [name.replace("_red_noise_", "\nRN ") for name in names[start:stop]]
        fig = plt.figure(figsize=(max(5, width * 1.6), max(5, width * 1.6)))
        options = dict(
            bins=30,
            range=ranges,
            labels=labels,
            smooth=1.0,
            smooth1d=None,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            levels=(0.68, 0.95),
            label_kwargs={"fontsize": 7},
        )
        try:
            corner.corner(
                points,
                fig=fig,
                color="#2563eb",
                hist_kwargs={"density": True, "linewidth": 1.4, "color": "#2563eb"},
                **options,
            )
            corner.corner(
                draws,
                fig=fig,
                color="#ea580c",
                hist_kwargs={"density": True, "linewidth": 1.4, "color": "#ea580c"},
                **options,
            )
            axes = np.asarray(fig.axes).reshape(width, width)
            for index in range(width):
                peak = max(
                    np.histogram(values[:, index], bins=30, range=ranges[index], density=True)[
                        0
                    ].max()
                    for values in (points, draws)
                )
                axes[index, index].set_ylim(0, peak * 1.1)
            fig.suptitle(f"Posterior vs Morph — parameters {start + 1}–{stop} ({width}D)", y=0.99)
            fig.legend(
                handles=[
                    Line2D([], [], color="#2563eb", label="Posterior (held out)"),
                    Line2D([], [], color="#ea580c", label="Morph"),
                ],
                loc="upper right",
                frameon=False,
            )
            path = output / f"corner_{number:02d}_{width}d.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            paths.append(path)
            print(f"Saved corner plot: {path}", flush=True)
        finally:
            plt.close(fig)
    return paths


def minimal_report(bridge, fitted, samples, densities, elapsed_seconds):
    """Keep the original scalar summary and runtime plus forward KL in nats."""
    converged = bridge is not None and bool(bridge.converged)
    kl = (
        float(
            np.mean(densities["log_target_posterior"] - densities["log_proposal_posterior"])
            - float(bridge.log_evidence)
        )
        if converged
        else None
    )
    return {
        "log_evidence": float(bridge.log_evidence) if bridge is not None else None,
        "relative_error": float(bridge.relative_error) if converged else None,
        "converged": converged,
        "iterations": int(bridge.iterations) if bridge is not None else 0,
        "posterior_iat": float(bridge.posterior_iat) if bridge is not None else None,
        "posterior_overlap": float(bridge.posterior_overlap) if bridge is not None else None,
        "proposal_overlap": float(bridge.proposal_overlap) if bridge is not None else None,
        "importance_ess": float(bridge.importance_ess) if bridge is not None else None,
        "proposal_support_fraction": float(np.mean(np.isfinite(densities["log_target_proposal"]))),
        "fit_samples": len(samples) // 2,
        "tc_samples": fitted.diagnostics.tc_sample_count,
        "bridge_samples": len(samples) - len(samples) // 2,
        "dimension": samples.shape[1],
        "elapsed_seconds": elapsed_seconds,
        "kl": kl,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str.lower,
        choices=tuple(MODEL_CHAINS),
        default="curn",
        help="PTA model: IRN only, uncorrelated common red noise, or Hellings–Downs",
    )
    parser.add_argument(
        "--datadir", type=Path, default=SOURCE_DIR / "15yr_stochastic_analysis/tutorials/data"
    )
    parser.add_argument("--chain-dir", type=Path, help="Defaults to the selected model's chain")
    parser.add_argument(
        "--output", type=Path, help="Defaults to benchmark-results/nanograv-MODEL-jax"
    )
    parser.add_argument("--burn", type=int, default=0, help="Discard this many original chain rows")
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--max-samples", type=int, help="Limit retained rows for a smoke run")
    parser.add_argument("--n-proposal", type=int, default=5000)
    parser.add_argument(
        "--n-estimates",
        type=int,
        default=1,
        help="Number of evidence estimates from one Morph fit (default: 1)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip all corner plots")
    parser.add_argument("--workers", type=int, default=4, help="1 evaluates in the parent process")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=5, help="SGM seed count (default 5)")
    parser.add_argument(
        "--max-tc-samples",
        type=int,
        default=200,
        help="TC scoring sample limit (1–200; default %(default)s); KDE still uses all fit rows",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--posterior-iat", type=float)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument(
        "--use-cached-posterior",
        action="store_true",
        help="Use chain column -4 only if it is normalized log likelihood + log prior",
    )
    args = parser.parse_args(argv)
    if args.n_estimates < 1:
        parser.error("--n-estimates must be >= 1")
    if args.chain_dir is None:
        args.chain_dir = SOURCE_DIR / MODEL_CHAINS[args.model]
    if args.output is None:
        args.output = Path(f"benchmark-results/nanograv-{args.model}-jax")
    return args


def main(argv=None):
    global pta_global
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    # Refuse to overwrite an earlier real-data result.
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose another --output: {args.output}")
    samples, cached, params = read_chain(
        args.chain_dir, burn_in=args.burn, thin=args.thin, max_samples=args.max_samples
    )
    print(f"Building {args.model.upper()} PTA model in parent process...", flush=True)
    pta_global = _build_pta(str(args.datadir), args.model)
    samples = align_parameters(samples, params, list(pta_global.param_names), args.model)
    params = list(pta_global.param_names)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    options = dict(
        posterior_log_density=cached if args.use_cached_posterior else None,
        n_proposal=args.n_proposal,
        n_estimates=args.n_estimates,
        seed=args.seed,
        order=args.order,
        n_seeds=args.n_seeds,
        max_tc_samples=args.max_tc_samples,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
        posterior_iat=args.posterior_iat,
        rtol=args.rtol,
        max_iterations=args.max_iterations,
        plot_output=None if args.no_plots else args.output,
        param_names=params,
    )

    def run(pool=None):
        estimates = []
        for index, (bridge, fitted, _, densities) in enumerate(
            iter_morphz_estimates(samples, ln_post, pool=pool, **options), 1
        ):
            report = minimal_report(bridge, fitted, samples, densities, time.monotonic() - start)
            if args.n_estimates > 1:
                report = {"estimate": index, **report}
            estimates.append(report)
            print(json.dumps(report, indent=2), flush=True)
        if args.n_estimates == 1:
            result = estimates[0]
        else:
            values = [
                row["log_evidence"]
                for row in estimates
                if row["converged"] and np.isfinite(row["log_evidence"])
            ]
            summary = {
                "n_estimates": len(estimates),
                "n_converged": len(values),
                "log_evidence_mean": float(np.mean(values)) if values else None,
                "log_evidence_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
            result = {**summary, "estimates": estimates}
            print(json.dumps(summary, indent=2), flush=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return all(row["converged"] for row in estimates)

    if args.workers == 1:
        all_converged = run()
    else:
        with mp.get_context("spawn").Pool(
            args.workers, initializer=_worker_init, initargs=(str(args.datadir), args.model)
        ) as pool:
            all_converged = run(pool)
    print(f"Saved results to {args.output.resolve()}", flush=True)
    if not all_converged:
        raise SystemExit("At least one bridge estimate did not converge; see result.json")


if __name__ == "__main__":
    main()
