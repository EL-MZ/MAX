"""Compare the optimized SGM against a JAX scan of every candidate per seed.

Reports compilation separately from synchronized warm execution. Both algorithms
must return identical selections and seed totals before any timing is reported.
"""

import argparse
import json
import platform
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jax_morph import candidate_blocks, sgm_kernel

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("dimension", "n_seeds"))
def full_scan_reference(blocks, scores, *, dimension, n_seeds):
    permutation = jnp.argsort(-scores, stable=True)
    blocks, scores = blocks[permutation], scores[permutation]
    block_count = dimension // blocks.shape[1]

    def run_seed(seed):
        selected = jnp.full((block_count,), -1, dtype=jnp.int32).at[0].set(seed)
        used = jnp.zeros((dimension,), dtype=bool).at[blocks[seed]].set(True)

        def visit(index, carry):
            count, used, selected, total = carry
            block = blocks[index]

            def accept(_):
                return (
                    count + 1,
                    used.at[block].set(True),
                    selected.at[count].set(index),
                    total + scores[index],
                )

            return jax.lax.cond(
                (count < block_count) & ~jnp.any(used[block]), accept, lambda _: carry, None
            )

        initial = (jnp.int32(1), used, selected, scores[seed])
        _, _, selected, total = jax.lax.fori_loop(0, len(blocks), visit, initial)
        return selected, total

    selected, totals = jax.lax.map(
        run_seed, jnp.arange(n_seeds, dtype=jnp.int32), batch_size=min(16, n_seeds)
    )
    return permutation[selected[jnp.argmax(totals)]], totals


def time_call(function, repeats):
    start = perf_counter()
    result = function()
    jax.block_until_ready(result)
    first_call = perf_counter() - start
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(function())
        timings.append(perf_counter() - start)
    return result, first_call, float(np.median(timings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    blocks = candidate_blocks(args.dimension, args.order)
    seeds = min(args.seeds, len(blocks))
    scores = jnp.asarray(np.random.default_rng(12).beta(5, 2, len(blocks)))
    reference, reference_first, reference_warm = time_call(
        lambda: full_scan_reference(blocks, scores, dimension=args.dimension, n_seeds=seeds),
        args.repeats,
    )
    optimized, optimized_first, optimized_warm = time_call(
        lambda: sgm_kernel(blocks, scores, dimension=args.dimension, n_seeds=seeds),
        args.repeats,
    )
    np.testing.assert_array_equal(reference[0], optimized.candidate_indices)
    np.testing.assert_allclose(reference[1], optimized.seed_totals, rtol=1e-12)
    report = {
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "dimension": args.dimension,
        "order": args.order,
        "candidates": len(blocks),
        "seeds": seeds,
        "repeats": args.repeats,
        "reference_first_call_seconds": reference_first,
        "optimized_first_call_seconds": optimized_first,
        "reference_warm_median_seconds": reference_warm,
        "optimized_warm_median_seconds": optimized_warm,
        "warm_speedup": reference_warm / optimized_warm,
        "identical_selection": True,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
