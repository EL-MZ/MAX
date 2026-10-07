"""Prepare exact-posterior data at 3k--10kD without running Morph or importing JAX."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .correlation_levels import CORRELATION_LEVELS, DEFAULT_CORRELATION_LEVELS
from .highdim_io import save_json
from .highdim_target import SHAPES, make_random_target
from .regression_target import make_regression_target


def prepare_dataset(target, directory, *, samples=2000, seed=1, chunk_size=256):
    """Stream iid posterior draws and cached unnormalized log densities to .npy files.

    Seeds and chunk size determine the random stream. These files can be passed
    directly to Morph; only target.json contains the ground-truth partition.
    """
    if samples < 4 or chunk_size < 1 or seed < 0:
        raise ValueError("require samples >=4, chunk_size >=1 and seed >=0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(
        (directory / name).exists()
        for name in ("target.json", "posterior_samples.npy", "manifest.json")
    ):
        raise ValueError("dataset already exists; choose a new output directory")
    save_json(directory / "target.json", target.description())
    points = np.lib.format.open_memmap(
        directory / "posterior_samples.npy",
        mode="w+",
        dtype=np.float64,
        shape=(samples, target.dimension),
    )
    logs = np.lib.format.open_memmap(
        directory / "posterior_log_density.npy",
        mode="w+",
        dtype=np.float64,
        shape=(samples,),
    )
    # Chunk size affects RNG consumption and is recorded for reproducibility.
    rng = np.random.default_rng(seed)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        batch = target.sample(rng, stop - start)
        values = target.log_unnormalized(batch)
        if not np.isfinite(batch).all() or not np.isfinite(values).all():
            raise RuntimeError("nonfinite exact samples or densities")
        points[start:stop], logs[start:stop] = batch, values
        points.flush()
        logs.flush()
        print(f"  {target.dimension:,}D posterior {stop:,}/{samples:,}", flush=True)
    manifest = {
        "dimension": target.dimension,
        "posterior_samples": samples,
        "sample_seed": seed,
        "target_seed": target.seed,
        "chunk_size": chunk_size,
        "exact_log_evidence": target.log_evidence,
        "family": target.description().get("family", "synthetic"),
        "precision": "float64",
        "true_order_counts": dict(Counter(map(len, target.blocks))),
        "true_shape_counts": dict(Counter(block.shape for block in target.factors)),
        "posterior_samples_file": "posterior_samples.npy",
        "posterior_log_density_file": "posterior_log_density.npy",
        "target_file": "target.json",
        "complete": True,
        "note": (
            "Independent exact posterior draws, equal weights, IAT=1. Cached log density is "
            "log L + log prior. Generation never runs Morph. Grouped regression uses "
            "exact normal-inverse-gamma posterior sampling and analytical model evidence. "
            "The synthetic family is a separately constructed normalization stress test."
        ),
    }
    save_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[3000, 5000, 10000])
    parser.add_argument("--family", choices=("regression", "synthetic"), default="regression")
    parser.add_argument("--group-orders", nargs="+", type=int, default=[5])
    parser.add_argument(
        "--correlation-levels",
        nargs="+",
        choices=CORRELATION_LEVELS,
        default=DEFAULT_CORRELATION_LEVELS,
        help="posterior correlation levels; default: low medium strong",
    )
    parser.add_argument(
        "--variance-coordinate", choices=("variance", "log_variance"), default="variance"
    )
    parser.add_argument("--observations-range", nargs=2, type=int, default=[8, 24])
    parser.add_argument("--cross-group-coupling", type=float, default=0.0)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES[:4]))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--target-seed", type=int, default=20261006)
    parser.add_argument("--sample-seed", type=int, default=20261007)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--log-evidence", type=float, help="default: -0.1 * dimension")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/highdim-data"))
    args = parser.parse_args()
    if args.samples < 4 or args.chunk_size < 1 or min(args.target_seed, args.sample_seed) < 0:
        parser.error("invalid sample count, chunk size, or seed")
    if len(args.dimensions) != len(set(args.dimensions)):
        parser.error("duplicate dimensions")
    if args.family == "regression" and args.log_evidence is not None:
        parser.error("regression evidence is determined by the model, not assigned")
    for dimension in args.dimensions:
        if args.family == "regression":
            target = make_regression_target(
                dimension,
                args.target_seed,
                group_orders=args.group_orders,
                variance_coordinate=args.variance_coordinate,
                observations_range=args.observations_range,
                cross_group_coupling=args.cross_group_coupling,
                correlation_levels=args.correlation_levels,
            )
        else:
            target = make_random_target(
                dimension,
                args.target_seed,
                orders=args.orders,
                shapes=args.shapes,
                log_evidence=args.log_evidence,
                correlation_levels=args.correlation_levels,
            )
        prepare_dataset(
            target,
            args.output / f"{dimension}d",
            samples=args.samples,
            seed=args.sample_seed,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
