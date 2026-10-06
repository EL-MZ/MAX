"""Reproducible CPU/GPU TC timing with at most 300 joint sample rows.

Run with scripts/run-tc-benchmark.sh on Linux.
Direct Python calls use --backend cpu or gpu.
All candidates are evaluated unless --limit-candidates is explicitly supplied
for a pilot. No SGM, Morph fitting, or sample generation is in the TC timers.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
from itertools import combinations, islice
from math import comb
from pathlib import Path
from time import perf_counter

import jax
import jaxlib
import numpy as np

from jax_morph import estimate_total_correlations

from .memory import MemoryMonitor, combine_pass_memory, format_mib

SCENARIOS = ("independent", "gaussian_pairs", "gaussian_triplets", "three_way", "mixed")


def make_samples(scenario, dimension, samples=300, seed=20261005):
    """Shuffle independent blocks; noisy parity triples have independent pairs.

    Parity block: A,B independent uniform signs; (A+e1,B+e2,A*B+e3),
    with independent N(0,0.25**2) noise. Gaussian blocks have unit variance,
    rho=0.85 for pairs and rho=0.7 for triples. Mixed cycles pair/parity/singleton.
    """
    if scenario not in SCENARIOS or dimension < 3 or not 4 <= samples <= 300:
        raise ValueError("require a known scenario, dimension >= 3, and 4..300 samples")
    rng = np.random.default_rng(seed)
    values = np.empty((samples, dimension), dtype=np.float64)
    blocks, kinds = [], []
    index = 0
    while index < dimension:
        if scenario == "mixed":
            kind = ("pair", "parity", "singleton")[len(blocks) % 3]
        else:
            kind = {
                "independent": "singleton",
                "gaussian_pairs": "pair",
                "gaussian_triplets": "triplet",
                "three_way": "parity",
            }[scenario]
        width = min({"singleton": 1, "pair": 2, "triplet": 3, "parity": 3}[kind], dimension - index)
        if width == 1:
            kind = "singleton"
        if kind == "parity" and width < 3:
            # Residual columns are independent; never introduce a fake parity pair.
            kind = "singleton"
            width = 1
        block = tuple(range(index, index + width))
        if kind == "parity":
            signs = rng.choice([-1.0, 1.0], size=(samples, 2))
            values[:, block] = np.column_stack((signs, signs.prod(axis=1)))
            values[:, block] += rng.normal(scale=0.25, size=(samples, 3))
        else:
            rho = 0.85 if kind == "pair" else 0.7 if kind == "triplet" else 0.0
            covariance = (1 - rho) * np.eye(width) + rho * np.ones((width, width))
            values[:, block] = rng.multivariate_normal(np.zeros(width), covariance, size=samples)
        blocks.append(block)
        kinds.append(kind)
        index += width
    permutation = rng.permutation(dimension)
    inverse = np.argsort(permutation)
    truth = [sorted(int(inverse[i]) for i in block) for block in blocks]
    values = np.ascontiguousarray(values[:, permutation])
    metadata = {
        "scenario": scenario,
        "dimension": dimension,
        "samples_per_parameter": samples,
        "seed": seed,
        "blocks": truth,
        "block_kinds": kinds,
        "permutation_new_to_old": permutation.tolist(),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "parity_noise_sd": 0.25,
        "pair_rho": 0.85,
        "triplet_rho": 0.7,
    }
    return values, metadata


def score_with_progress(
    samples, dimension, order, total, args, batch, label, saved=None, device=None
):
    """Stream candidate combinations; retain only one outer chunk in memory."""
    candidates = combinations(range(dimension), order)
    completed, api_seconds = 0, 0.0
    score_sum, score_min, score_max = 0.0, float("inf"), float("-inf")
    with MemoryMonitor(device, interval=args.memory_interval) as memory:
        wall_start = perf_counter()
        while completed < total:
            blocks = np.asarray(
                list(islice(candidates, min(args.progress_every, total - completed))),
                dtype=np.int32,
            )
            start = perf_counter()
            result = estimate_total_correlations(
                samples,
                blocks,
                max_samples=args.samples,
                chunk_size=args.chunk_size,
                candidate_batch_size=batch,
            )
            jax.block_until_ready(result)
            api_seconds += perf_counter() - start
            raw = np.asarray(result.raw_scores)
            if result.sample_count > 300 or not np.isfinite(raw).all():
                raise RuntimeError("Invalid sample count or nonfinite TC scores")
            score_sum += float(raw.sum())
            score_min = min(score_min, float(raw.min()))
            score_max = max(score_max, float(raw.max()))
            if saved is not None:
                saved[completed : completed + len(raw)] = raw
            completed += len(raw)
            elapsed = perf_counter() - wall_start
            eta = elapsed * (total - completed) / completed
            print(
                f"  {label}: TC {completed:,} / {total:,} "
                f"({100 * completed / total:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s "
                f"| {memory.progress()}",
                flush=True,
            )
        wall_seconds = perf_counter() - wall_start
    summary = {"raw_tc_mean": score_sum / total, "raw_tc_min": score_min, "raw_tc_max": score_max}
    return api_seconds, wall_seconds, summary, memory.final_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[100])
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--memory-interval",
        type=float,
        default=0.1,
        help="seconds between RSS/JAX memory samples (default: 0.1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=4096,
        help="completed TCs per outer chunk/progress update",
    )
    parser.add_argument(
        "--limit-candidates", type=int, help="optional prefix cap for a limited timing pilot"
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="write all first-pass raw scores to disk using a memory map",
    )
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if not 4 <= args.samples <= 300 or args.repeats < 1 or min(args.dimensions) < 3:
        parser.error("samples must be 4..300, repeats >= 1, dimensions >= 3")
    if min(args.orders) < 2 or max(args.orders) >= 10:
        parser.error("TC block orders must be 2..9")
    if max(args.orders) > min(args.dimensions) or max(args.orders) >= args.samples:
        parser.error("every order must be <= every dimension and < the sample count")
    if min(args.candidate_batch_sizes) < 1 or args.chunk_size < 1 or args.progress_every < 1:
        parser.error("batch, query chunk, and progress chunk sizes must be positive")
    if args.limit_candidates is not None and args.limit_candidates < 1:
        parser.error("candidate limit must be positive")
    if not np.isfinite(args.memory_interval) or args.memory_interval <= 0:
        parser.error("memory interval must be finite and positive")
    for name in ("dimensions", "orders", "scenarios", "candidate_batch_sizes"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"duplicate {name} would overwrite results")
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if any(device.platform != args.backend for device in devices):
        raise RuntimeError(f"Requested {args.backend}; found {devices}")
    memory_device = devices[0] if args.backend == "gpu" else None
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "metadata": {
            "backend": args.backend,
            "devices": [str(device) for device in devices],
            "device_kind": [device.device_kind for device in devices],
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "precision": "float64",
            "memory_device": str(memory_device) if memory_device is not None else None,
            "memory_note": "MiB = 2**20 bytes. RSS and GPU live/pool are sampled per pass; "
            "case peaks span first pass and warm repeats. nvidia-smi samples whole-device VRAM "
            "at most once per second plus pass endpoints and includes other applications. "
            "Native CPU high-water and GPU allocator/pool high-water counters are process "
            "lifetime values, not per-case peaks. Earlier-case caches remain resident. "
            "Unsupported counters are null. Memory polling can add timing overhead; explicit "
            "start/stop snapshots are outside timers. Only the default JAX device is monitored.",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "timing_note": "API time sums synchronized public TC calls in streamed outer chunks; "
            "each chunk includes validation, transfers, and marginal KDE recomputation. "
            "API timers exclude candidate generation, printing, and saving. Separate wall "
            "times include those overheads. First pass can compile; warm repeats reuse "
            "compilation. Only the first case per shape/batch is cold. No SGM or Morph fit.",
        },
        "targets": [],
        "records": [],
        "complete": False,
    }
    print("TC only; float64; no SGM or partition selection.", flush=True)
    for dimension in args.dimensions:
        for order in args.orders:
            count = comb(dimension, order)
            evaluated = min(count, args.limit_candidates or count)
            print(
                f"PLAN {dimension}D order={order}: {count:,} total TCs, "
                f"{evaluated:,} per pass, {evaluated * (args.repeats + 1):,} "
                "including first pass and warm repeats per scenario/batch",
                flush=True,
            )
    for dimension in args.dimensions:
        for scenario in args.scenarios:
            samples, truth = make_samples(scenario, dimension, args.samples, args.seed)
            run["targets"].append(truth)
            np.save(args.output / f"samples_{scenario}_{dimension}d.npy", samples)
            for order in args.orders:
                count = comb(dimension, order)
                evaluated = min(count, args.limit_candidates or count)
                reference = None
                for batch in args.candidate_batch_sizes:
                    print(
                        f"START {args.backend} {scenario} {dimension}D order={order} "
                        f"TCs={evaluated:,}/{count:,} samples={len(samples)} batch={batch}",
                        flush=True,
                    )
                    filename = f"raw_scores_{scenario}_{dimension}d_o{order}_b{batch}.npy"
                    saved = (
                        np.lib.format.open_memmap(
                            args.output / filename, mode="w+", dtype=np.float64, shape=(evaluated,)
                        )
                        if args.save_scores
                        else None
                    )
                    first, first_wall, summary, first_memory = score_with_progress(
                        samples,
                        dimension,
                        order,
                        evaluated,
                        args,
                        batch,
                        "first pass",
                        saved,
                        device=memory_device,
                    )
                    if saved is not None:
                        saved.flush()
                        del saved
                    warm, warm_wall, warm_memory = [], [], []
                    for repeat in range(args.repeats):
                        seconds, wall_seconds, repeated, pass_memory = score_with_progress(
                            samples,
                            dimension,
                            order,
                            evaluated,
                            args,
                            batch,
                            f"warm {repeat + 1}/{args.repeats}",
                            device=memory_device,
                        )
                        np.testing.assert_allclose(
                            list(repeated.values()), list(summary.values()), rtol=1e-10, atol=1e-10
                        )
                        warm.append(seconds)
                        warm_wall.append(wall_seconds)
                        warm_memory.append(pass_memory)
                    if reference is not None:
                        np.testing.assert_allclose(
                            list(reference.values()), list(summary.values()), rtol=1e-10, atol=1e-10
                        )
                    reference = summary
                    record = {
                        "scenario": scenario,
                        "dimension": dimension,
                        "order": order,
                        "samples_per_parameter": len(samples),
                        "candidates": evaluated,
                        "total_candidates": count,
                        "all_candidates": evaluated == count,
                        "candidate_batch_size": batch,
                        "progress_every": args.progress_every,
                        "chunk_size": args.chunk_size,
                        "memory_interval": args.memory_interval,
                        "input_sha256": truth["sha256"],
                        "api_first_seconds": first,
                        "wall_first_seconds": first_wall,
                        "api_warm_seconds": warm,
                        "wall_warm_seconds": warm_wall,
                        "api_warm_median_seconds": float(np.median(warm)),
                        "wall_warm_median_seconds": float(np.median(warm_wall)),
                        "api_candidates_per_second": evaluated / float(np.median(warm)),
                        "raw_score_file": filename if args.save_scores else None,
                        "memory_first": first_memory,
                        "memory_warm": warm_memory,
                        **combine_pass_memory([first_memory, *warm_memory]),
                        **summary,
                    }
                    run["records"].append(record)
                    (args.output / "results.json").write_text(json.dumps(run, indent=2) + "\n")
                    scalar_keys = [
                        key
                        for key, value in run["records"][0].items()
                        if not isinstance(value, (list, dict))
                    ]
                    with (args.output / "timings.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=scalar_keys, extrasaction="ignore"
                        )
                        writer.writeheader()
                        writer.writerows(run["records"])
                    print(
                        f"DONE API warm median={record['api_warm_median_seconds']:.3f}s; "
                        f"wall={record['wall_warm_median_seconds']:.3f}s; "
                        f"RAM sampled peak={format_mib(record['cpu_rss_peak_mib'])} MiB; "
                        f"GPU live/pool sampled peaks="
                        f"{format_mib(record['gpu_live_peak_mib'])}/"
                        f"{format_mib(record['gpu_pool_peak_mib'])} MiB; "
                        f"GPU device sampled peak="
                        f"{format_mib(record['gpu_device_used_peak_mib'])} MiB",
                        flush=True,
                    )
    run["complete"] = True
    (args.output / "results.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
