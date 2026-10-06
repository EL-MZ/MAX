"""Run the unchanged Morph pair search and JAX MorphZ on exact random-block targets.

No screening, pruning, new factorization, or estimator is inserted. Target
block orders 1--5 and Morph order (default 2) are separate options. Use
benchmarks.highdim_prepare to generate 3k--10kD data without fitting Morph.
"""

import argparse
import json
import os
import platform
import subprocess
from collections import Counter
from math import comb
from pathlib import Path
from time import perf_counter

import jax
import jaxlib
import numpy as np
from scipy.special import logsumexp

from jax_morph import bridge_sampling, fit, log_prob, sample

from .distributions import block_metrics
from .highdim_io import digest, save_json
from .highdim_target import SHAPES, load_target, make_random_target
from .memory import MemoryMonitor
from .regression_target import make_regression_target


def timed(call):
    start = perf_counter()
    value = call()
    jax.block_until_ready(value)
    jax.effects_barrier()
    return value, perf_counter() - start


def sgm_seed_count(dimension, order, requested=None):
    count = comb(dimension, order)
    return min(count, 5 if requested is None else requested)


def bridge_record(bridge, exact_log_z, average_iterations=5):
    log_z = float(bridge.log_evidence)
    relative_error = float(bridge.relative_error)
    overlap = min(float(bridge.posterior_overlap), float(bridge.proposal_overlap))
    converged = bool(bridge.converged)
    usable = converged and np.isfinite(relative_error) and relative_error <= 0.2 and overlap >= 0.01
    return {
        "log_evidence": log_z,
        "final_iteration_log_evidence": float(bridge.final_log_evidence),
        "log_evidence_averaging_iterations": min(int(bridge.iterations), average_iterations),
        "exact_log_evidence": exact_log_z,
        "log_evidence_error": log_z - exact_log_z,
        "relative_error": relative_error,
        "posterior_overlap": float(bridge.posterior_overlap),
        "proposal_overlap": float(bridge.proposal_overlap),
        "importance_ess": float(bridge.importance_ess),
        "iterations": int(bridge.iterations),
        "converged": converged,
        "status": "usable_overlap" if usable else "poor_overlap_or_nonconvergence",
        "within_3_conditional_se": bool(
            np.isfinite(relative_error) and abs(log_z - exact_log_z) <= 3 * relative_error
        ),
        "history": np.asarray(bridge.log_evidence_history)[: int(bridge.iterations) + 1],
    }


def run_case(args, dimension, seed, directory, fit_cache=None):
    streams = [
        np.random.default_rng(stream)
        for stream in np.random.SeedSequence([seed, dimension]).spawn(3)
    ]
    if args.dataset:
        target = load_target(args.dataset / "target.json")
        points = np.load(args.dataset / "posterior_samples.npy", mmap_mode="r")
        cached_logs = np.load(args.dataset / "posterior_log_density.npy", mmap_mode="r")
        required = args.fit_samples + args.posterior_samples
        if (
            points.shape[1] != dimension
            or len(points) < required
            or cached_logs.shape != (len(points),)
        ):
            raise ValueError("prepared dataset has incompatible dimensions or too few rows")
        training = np.asarray(points[: args.fit_samples])
        held_out = np.asarray(points[args.fit_samples : required])
        posterior_target = np.asarray(cached_logs[args.fit_samples : required])
        if not np.allclose(posterior_target, target.log_unnormalized(held_out), atol=1e-8, rtol=0):
            raise ValueError("cached log densities disagree with the exact target")
    else:
        if getattr(args, "family", "synthetic") == "regression":
            target = make_regression_target(
                dimension,
                args.target_seed,
                group_orders=args.group_orders,
                variance_coordinate=args.variance_coordinate,
                observations_range=args.observations_range,
                cross_group_coupling=args.cross_group_coupling,
            )
        else:
            target = make_random_target(
                dimension,
                args.target_seed,
                orders=args.target_orders,
                shapes=args.shapes,
                log_evidence=args.log_evidence,
            )
        training = target.sample(streams[0], args.fit_samples)
        held_out = target.sample(streams[1], args.posterior_samples)
        posterior_target = target.log_unnormalized(held_out)
    save_json(directory / "target.json", target.description())
    if not np.isfinite(posterior_target).all():
        raise RuntimeError("nonfinite exact posterior density")
    identity_error = float(
        np.max(np.abs(posterior_target - target.logpdf(held_out) - target.log_evidence))
    )
    if identity_error > 1e-8:
        raise RuntimeError("prior * likelihood / posterior disagrees with analytical evidence")
    control_points = target.sample(streams[2], min(128, args.proposals))
    control_logs = target.logpdf(control_points)
    control, control_seconds = timed(
        lambda: bridge_sampling(
            posterior_target,
            target.logpdf(held_out),
            target.log_unnormalized(control_points),
            control_logs,
            posterior_iat=1.0,
        )
    )
    if not bool(control.converged) or abs(float(control.log_evidence) - target.log_evidence) > 1e-8:
        raise RuntimeError("exact-proposal bridge arithmetic control failed")
    case = {
        "dimension": dimension,
        "seed": seed,
        "target_seed": target.seed,
        "family": target.description().get("family", "synthetic"),
        "posterior_identity_max_error": identity_error,
        "training_sha256": digest(training),
        "posterior_sha256": digest(held_out),
        "true_order_counts": dict(Counter(map(len, target.blocks))),
        "true_shape_counts": dict(Counter(block.shape for block in target.factors)),
        "analytic_control": bridge_record(control, target.log_evidence),
        "analytic_control_seconds": control_seconds,
    }
    save_json(directory / "metrics.json", case)
    fit_key, proposal_key = jax.random.split(jax.random.PRNGKey(seed))
    seed_count = sgm_seed_count(dimension, args.morph_order, args.n_seeds)
    reports = getattr(args, "progress_reports", 20)
    candidate_count = comb(dimension, args.morph_order)
    candidate_every = (candidate_count + reports - 1) // reports if reports else 0
    seed_every = (seed_count + reports - 1) // reports if reports else 0
    fit_options = dict(
        order=args.morph_order,
        n_seeds=seed_count,
        bandwidth_scale=args.bandwidth_scale,
        regularization=args.regularization,
        max_tc_samples=args.tc_samples,
        chunk_size=args.query_chunk_size,
        candidate_batch_size=args.candidate_batch_size,
        seed_batch_size=args.seed_batch_size,
        sgm_early_stop=args.sgm_early_stop,
        max_candidates=args.max_candidates,
        candidate_progress_every=candidate_every,
        seed_progress_every=seed_every,
    )
    cache_key = (seed, case["training_sha256"], tuple(sorted(fit_options.items())))
    cached = fit_cache.get(cache_key) if fit_cache is not None else None
    device = jax.devices()[0] if args.backend == "gpu" else None
    with MemoryMonitor(device) as memory:
        workflow_start = perf_counter()
        print(
            f"  {dimension}D seed {seed}: existing Morph order {args.morph_order}, "
            f"{seed_count} SGM seeds",
            flush=True,
        )
        if cached is None:
            fitted, fit_seconds = timed(lambda: fit(training, key=fit_key, **fit_options))
            original_fit_seconds = fit_seconds
            if fit_cache is not None:
                fit_cache[cache_key] = (fitted, fit_seconds)
        else:
            fitted, original_fit_seconds = cached
            fit_seconds = 0.0
            print("    reusing the previous native Morph fit", flush=True)
        print(f"    fit completed in {fit_seconds:.2f}s; evaluating held-out posterior", flush=True)
        posterior_proposal, posterior_seconds = timed(
            lambda: log_prob(fitted.state, held_out, chunk_size=args.query_chunk_size)
        )
        posterior_proposal = np.asarray(posterior_proposal)
        differences = posterior_target - target.log_evidence - posterior_proposal
        case.update(
            blocks=fitted.state.blocks,
            morph_order=args.morph_order,
            forward_kl=float(differences.mean()),
            forward_kl_mc_se=float(differences.std(ddof=1) / np.sqrt(len(differences))),
            kl_per_dimension=float(differences.mean() / dimension),
            fit_seconds=fit_seconds,
            original_fit_seconds=original_fit_seconds,
            fit_reused=cached is not None,
            posterior_log_prob_seconds=posterior_seconds,
            candidate_count=len(fitted.diagnostics.candidates),
            tc_sample_count=fitted.diagnostics.tc_sample_count,
            retained_total_correlation=float(fitted.diagnostics.retained_total_correlation),
            seed_count=len(fitted.diagnostics.seed_totals),
        )
        case.update(block_metrics(target.blocks, fitted.state.blocks))
        save_json(directory / "metrics.json", case)
        warm_points = held_out[: min(args.proposal_chunk_size, len(held_out))]
        _, warm_seconds = timed(
            lambda: log_prob(fitted.state, warm_points, chunk_size=args.query_chunk_size)
        )
        case["warm_log_prob_seconds_per_point"] = warm_seconds / len(warm_points)
        targets, logs, previews = [], [], []
        sample_seconds, density_seconds, target_seconds = 0.0, 0.0, 0.0
        chunks = (args.proposals + args.proposal_chunk_size - 1) // args.proposal_chunk_size
        keys = jax.random.split(proposal_key, chunks)
        for index, start in enumerate(range(0, args.proposals, args.proposal_chunk_size)):
            count = min(args.proposal_chunk_size, args.proposals - start)
            proposal, seconds = timed(lambda: sample(keys[index], fitted.state, count))
            sample_seconds += seconds
            proposal_logs, seconds = timed(
                lambda: log_prob(fitted.state, proposal, chunk_size=args.query_chunk_size)
            )
            density_seconds += seconds
            proposal = np.asarray(proposal)
            begin = perf_counter()
            target_logs = target.log_unnormalized(proposal)
            target_seconds += perf_counter() - begin
            targets.append(target_logs)
            logs.append(np.asarray(proposal_logs))
            have = sum(len(batch) for batch in previews)
            if have < args.plot_samples:
                previews.append(proposal[: args.plot_samples - have])
            completed = start + count
            proposal_every = (args.proposals + reports - 1) // reports if reports else 0
            if reports and (
                index == 0
                or completed == args.proposals
                or completed // proposal_every != start // proposal_every
            ):
                print(f"    proposals {completed:,}/{args.proposals:,}", flush=True)
        targets, logs = np.concatenate(targets), np.concatenate(logs)
        case["outside_support_fraction"] = float(np.mean(np.isneginf(targets)))
        if np.isfinite(targets).any():
            bridge, bridge_seconds = timed(
                lambda: bridge_sampling(
                    posterior_target,
                    posterior_proposal,
                    targets,
                    logs,
                    posterior_iat=1.0,
                    rtol=args.rtol,
                    max_iterations=args.max_iterations,
                    average_iterations=getattr(args, "bridge_average_iterations", 5),
                    progress_every=(
                        ((args.max_iterations or 10000) + reports - 1) // reports if reports else 0
                    ),
                )
            )
            case.update(
                bridge_record(
                    bridge, target.log_evidence, getattr(args, "bridge_average_iterations", 5)
                )
            )
        else:
            bridge_seconds = 0.0
            case.update(
                exact_log_evidence=target.log_evidence,
                log_evidence=None,
                final_iteration_log_evidence=None,
                log_evidence_averaging_iterations=0,
                log_evidence_error=None,
                relative_error=None,
                posterior_overlap=0.0,
                proposal_overlap=0.0,
                importance_ess=0.0,
                iterations=0,
                converged=False,
                status="no_proposals_in_target_support",
                within_3_conditional_se=False,
                history=[],
            )
        case.update(
            importance_log_evidence=float(logsumexp(targets - logs) - np.log(len(logs))),
            proposal_sample_seconds=sample_seconds,
            proposal_log_prob_seconds=density_seconds,
            proposal_target_seconds=target_seconds,
            bridge_seconds=bridge_seconds,
            morph_seconds=fit_seconds,
            bridge_sampling_total_seconds=(
                posterior_seconds
                + sample_seconds
                + density_seconds
                + target_seconds
                + bridge_seconds
            ),
            workflow_seconds=perf_counter() - workflow_start,
            target_evaluations=args.proposals,
            fit_count=len(training),
            posterior_count=len(held_out),
            proposal_count=args.proposals,
        )
    case["memory"] = memory.final_report
    save_json(directory / "metrics.json", case)
    if not args.no_plots:
        from .highdim_plots import plot_case

        plot_case(target, held_out[: args.plot_samples], np.concatenate(previews), case, directory)
    case["complete"] = True
    save_json(directory / "metrics.json", case)
    print(
        f"    KL={case['forward_kl']:.3f}, pair recall={case['pair_recall']}, "
        f"delta logZ={case['log_evidence_error']}, "
        f"overlap={case['proposal_overlap']:.3g}, {case['status']}",
        flush=True,
    )
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[200])
    parser.add_argument("--family", choices=("regression", "synthetic"), default="regression")
    parser.add_argument("--group-orders", nargs="+", type=int, default=[5])
    parser.add_argument(
        "--variance-coordinate", choices=("variance", "log_variance"), default="variance"
    )
    parser.add_argument("--observations-range", nargs=2, type=int, default=[8, 24])
    parser.add_argument("--cross-group-coupling", type=float, default=0.0)
    parser.add_argument("--target-orders", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES[:4]))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261006])
    parser.add_argument("--target-seed", type=int, default=20261006)
    parser.add_argument("--log-evidence", type=float, help="default: -0.1 * dimension")
    parser.add_argument("--dataset", type=Path, help="a prepared single-dimension data directory")
    parser.add_argument("--fit-samples", type=int, default=1000)
    parser.add_argument("--posterior-samples", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=1000)
    parser.add_argument(
        "--proposal-counts",
        nargs="+",
        type=int,
        help="run these proposal counts sharing one Morph fit; overrides --proposals",
    )
    parser.add_argument("--morph-order", type=int, default=2)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="top-ranked SGM seeds to try; default: 5; explicit values override",
    )
    parser.add_argument(
        "--tc-samples", type=int, help="default: native Morph uses all fitting rows"
    )
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--seed-batch-size", type=int, default=16)
    parser.add_argument("--sgm-early-stop", action="store_true")
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--proposal-chunk-size", type=int, default=256)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.1,
        help="absolute delta-logZ stopping tolerance; default: 0.1",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="optional bridge iteration cap; default: run until delta-logZ tolerance",
    )
    parser.add_argument(
        "--bridge-average-iterations",
        type=int,
        default=5,
        help="report the mean of this many final computed logZ values; default: 5",
    )
    parser.add_argument(
        "--progress-reports",
        type=int,
        default=20,
        help="approximate number of flushed progress reports per stage; 0 disables",
    )
    parser.add_argument("--plot-samples", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/highdim"))
    args = parser.parse_args()
    if args.progress_reports < 0:
        parser.error("progress-reports must be nonnegative")
    if args.proposal_counts and (
        min(args.proposal_counts) < 2 or len(set(args.proposal_counts)) != len(args.proposal_counts)
    ):
        parser.error("proposal-counts must be unique and at least 2")
    if args.dataset:
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        if not manifest.get("complete"):
            parser.error("prepared dataset is incomplete")
        args.dimensions = [manifest["dimension"]]
        args.family = (
            "regression" if manifest.get("family") == "grouped_regression" else "synthetic"
        )
        if len(args.seeds) != 1:
            parser.error(
                "a prepared dataset uses fixed posterior draws; use one proposal/search seed"
            )
    if args.family == "regression" and args.log_evidence is not None and not args.dataset:
        parser.error("regression evidence is determined by the model, not assigned")
    for name in ("dimensions", "target_orders", "shapes", "seeds"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"duplicate {name}")
    if (
        min(args.target_orders) < 1
        or max(args.target_orders) > 5
        or min(args.dimensions) < max(args.target_orders)
    ):
        parser.error("target orders must be in 1..5 and fit every dimension")
    if not 1 <= args.morph_order < 10 or args.morph_order > min(args.dimensions):
        parser.error("invalid Morph order")
    if (
        args.fit_samples <= max(args.morph_order, max(args.target_orders))
        or args.posterior_samples < 2
        or args.proposals < 2
    ):
        parser.error("too few samples")
    for name in (
        "n_seeds",
        "tc_samples",
        "max_candidates",
        "candidate_batch_size",
        "seed_batch_size",
        "query_chunk_size",
        "proposal_chunk_size",
        "max_iterations",
        "bridge_average_iterations",
        "plot_samples",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name} must be positive")
    if args.tc_samples is not None and args.tc_samples <= args.morph_order:
        parser.error("tc-samples must exceed Morph order")
    for name in ("bandwidth_scale", "rtol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.regularization) or args.regularization < 0:
        parser.error("regularization must be finite and nonnegative")
    if args.log_evidence is not None and not np.isfinite(args.log_evidence):
        parser.error("log-evidence must be finite")
    if min(args.seeds + [args.target_seed]) < 0:
        parser.error("seeds must be nonnegative")
    for dimension in args.dimensions:
        count = comb(dimension, args.morph_order)
        print(
            f"{dimension:,}D Morph order {args.morph_order}: {count:,} native candidates, "
            f"{sgm_seed_count(dimension, args.morph_order, args.n_seeds):,} SGM seeds",
            flush=True,
        )
        if not args.dry_run and count > args.max_candidates:
            parser.error(
                f"{count:,} candidates exceed --max-candidates {args.max_candidates:,}; "
                "explicitly choose a limit"
            )
    if args.dry_run:
        return
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if any(device.platform != args.backend for device in devices):
        raise RuntimeError(f"requested {args.backend}, found {devices}; use scripts/run-highdim.sh")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("resume", "dry_run")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "configuration.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != config:
            parser.error("output exists; use matching --resume settings or a new output directory")
    else:
        save_json(path, config)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    run = {
        "metadata": {
            "revision": revision,
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "devices": [str(device) for device in devices],
            "device_kind": [device.device_kind for device in devices],
            "logical_cpus": os.cpu_count(),
            "precision": "float64",
            "configuration": config,
            "note": (
                "Calls unchanged jax_morph.fit, log_prob, sample and bridge_sampling. "
                "Independent fitting, held-out posterior and proposal draws; posterior IAT=1. "
                "Conditional MC errors exclude training variability. Timings include "
                "synchronization, transfers and first-call compilation. Exact-proposal "
                "control checks bridge arithmetic, not pair-Morph accuracy."
            ),
        },
        "cases": [],
    }
    for dimension in args.dimensions:
        for seed in args.seeds:
            fit_cache = {}
            for proposal_count in args.proposal_counts or [args.proposals]:
                args.proposals = proposal_count
                suffix = f"_proposals{proposal_count}" if args.proposal_counts else ""
                directory = args.output / f"{dimension}d_seed{seed}{suffix}"
                directory.mkdir(exist_ok=True)
                path = directory / "metrics.json"
                previous = json.loads(path.read_text()) if args.resume and path.exists() else None
                if previous and previous.get("complete"):
                    print(f"Skipping completed {directory.name}", flush=True)
                    case = previous
                else:
                    case = run_case(args, dimension, seed, directory, fit_cache=fit_cache)
                run["cases"].append(case)
                save_json(args.output / "results.json", run)
            fit_cache.clear()
            jax.clear_caches()
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
