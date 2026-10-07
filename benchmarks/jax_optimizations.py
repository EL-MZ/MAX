"""Small synchronized comparisons against the pre-optimization Git checkpoint.

Run with python -m benchmarks.jax_optimizations. Select the backend through
JAX_PLATFORMS. First-call times include compilation but are not isolated cold
process measurements; earlier cases can warm shared primitives.
"""

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

import jax_morph

CHECKPOINT = "5af3a4e52b5a6c34aa4887d0bffd6372339404f9"


def measure(function, repeats):
    start = perf_counter()
    value = function()
    jax.block_until_ready(value)
    first = perf_counter() - start
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        jax.block_until_ready(value)
        timings.append(perf_counter() - start)
    return {"first_ms": first * 1000, "warm_median_ms": float(np.median(timings)) * 1000}, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(21)
    data, high = rng.normal(size=(96, 12)), rng.normal(size=(96, 64))
    points = jax.numpy.asarray(high[:32])

    def target(point):
        return -0.5 * jax.numpy.sum(point * point)

    results = {}
    with tempfile.TemporaryDirectory(prefix="max-jax-baseline-") as folder:
        package = Path(folder) / "_jax_morph_checkpoint"
        package.mkdir()
        paths = subprocess.check_output(
            ["git", "ls-tree", "--name-only", CHECKPOINT, "src/jax_morph/"], cwd=root, text=True
        ).splitlines()
        for path in paths:
            if path.endswith(".py"):
                (package / Path(path).name).write_bytes(
                    subprocess.check_output(["git", "show", f"{CHECKPOINT}:{path}"], cwd=root)
                )
        sys.path.insert(0, folder)
        try:
            baseline = importlib.import_module("_jax_morph_checkpoint")
            for label, module in (("before", baseline), ("after", jax_morph)):
                blocks = module.candidate_blocks(12, 2)
                timing, tc = measure(
                    lambda: module.estimate_total_correlations(data, blocks, chunk_size=32),
                    args.repeats,
                )
                results.setdefault("tc_96x12", {})[label] = timing
                timing, fitted = measure(lambda: module.fit(high, order=1), args.repeats)
                results.setdefault("fit_order1_96x64", {})[label] = timing
                timing, density = measure(
                    lambda: module.log_prob(fitted.state, points, chunk_size=32), args.repeats
                )
                results.setdefault("density_32x64", {})[label] = timing
                evaluator = importlib.import_module(module.__name__ + ".evidence")._evaluate
                timing, _ = measure(lambda: evaluator(target, points, 16), args.repeats)
                results.setdefault("target_32x64", {})[label] = timing
                if label == "before":
                    previous_tc, previous_density = np.asarray(tc.raw_scores), np.asarray(density)
                else:
                    np.testing.assert_allclose(tc.raw_scores, previous_tc, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(density, previous_density, rtol=1e-11, atol=1e-12)
        finally:
            sys.path.remove(folder)
    print(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "jax": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "repeats": args.repeats,
                "dtype": "float64",
                "comparisons": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
