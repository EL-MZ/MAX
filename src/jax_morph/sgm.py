"""Seeded Greedy Maximization (Algorithm 1 of Zahraoui et al.).

All numerical selection work stays in JAX. Candidate enumeration and input
validation are host operations. This is a heuristic, including for order two.
"""

from functools import partial
from itertools import combinations
from math import comb
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._validation import positive_integer


class SGMResult(NamedTuple):
    blocks: jax.Array
    candidate_indices: jax.Array
    total_score: jax.Array
    seed_index: jax.Array
    seed_totals: jax.Array


def candidate_blocks(dimension, order=2, *, max_candidates=100_000):
    """Enumerate all blocks, in lexicographic order; never silently prune them."""
    dimension = positive_integer(dimension, "dimension")
    order = positive_integer(order, "order")
    positive_integer(max_candidates, "max_candidates")
    if order > dimension:
        raise ValueError("order cannot exceed dimension")
    count = comb(dimension, order)
    if count > max_candidates:
        raise ValueError(
            f"{count:,} candidate blocks exceed max_candidates={max_candidates:,}; "
            "choose a smaller order or explicitly raise the limit"
        )
    return jnp.asarray(list(combinations(range(dimension), order)), dtype=jnp.int32)


def _greedy_seed(seed, sorted_blocks, sorted_scores, dimension, early_stop):
    """Scan ranked candidates, optionally stopping when the partition is full."""
    candidate_count, order = sorted_blocks.shape
    block_count = dimension // order
    selected = jnp.full((block_count,), -1, dtype=jnp.int32).at[0].set(seed)
    used = jnp.zeros((dimension,), dtype=bool).at[sorted_blocks[seed]].set(True)
    initial = (jnp.int32(0), jnp.int32(1), used, selected, sorted_scores[seed])

    def unfinished(carry):
        cursor, count, *_ = carry
        return (cursor < candidate_count) & (count < block_count)

    def scan_candidate(carry):
        cursor, count, used, selected, total = carry
        block = sorted_blocks[cursor]

        def accept(_):
            return (
                count + 1,
                used.at[block].set(True),
                selected.at[count].set(cursor),
                total + sorted_scores[cursor],
            )

        can_accept = (count < block_count) & ~jnp.any(used[block])
        count, used, selected, total = jax.lax.cond(
            can_accept, accept, lambda _: (count, used, selected, total), None
        )
        return cursor + 1, count, used, selected, total

    if early_stop:
        final = jax.lax.while_loop(unfinished, scan_candidate, initial)
    else:
        final = jax.lax.fori_loop(
            0, candidate_count, lambda _, carry: scan_candidate(carry), initial
        )
    _, _, _, selected, total = final
    return selected, total


@partial(jax.jit, static_argnames=("dimension", "n_seeds", "seed_batch_size", "early_stop"))
def sgm_kernel(blocks, scores, *, dimension, n_seeds, seed_batch_size=16, early_stop=False):
    """JIT-compatible core for a complete candidate collection and valid scores.

    Stable sorting makes ties reproducible. Seeds are processed in bounded
    batches; no candidate-by-candidate conflict matrix is constructed.
    """
    permutation = jnp.argsort(-scores, stable=True)
    sorted_blocks, sorted_scores = blocks[permutation], scores[permutation]
    selected, totals = jax.lax.map(
        lambda seed: _greedy_seed(seed, sorted_blocks, sorted_scores, dimension, early_stop),
        jnp.arange(n_seeds, dtype=jnp.int32),
        batch_size=min(seed_batch_size, n_seeds),
    )
    best_seed = jnp.argmax(totals)
    indices = permutation[selected[best_seed]]
    return SGMResult(blocks[indices], indices, totals[best_seed], permutation[best_seed], totals)


def seeded_greedy_maximization(
    blocks,
    scores,
    *,
    dimension,
    n_seeds=None,
    seed_batch_size=16,
    early_stop=False,
):
    """Select floor(d/L) disjoint blocks using the top K seeds.

    ``n_seeds=None`` tries every seed. Supply the *complete* set of length-L
    combinations; incomplete graphs would require a separate set-packing API.
    Early stopping preserves selections but its speed depends on device/compiler.
    """
    dimension = positive_integer(dimension, "dimension")
    positive_integer(seed_batch_size, "seed_batch_size")
    blocks_np, scores_np = np.asarray(blocks), np.asarray(scores)
    if blocks_np.ndim != 2 or blocks_np.shape[0] == 0:
        raise ValueError("blocks must be a nonempty (candidates, order) matrix")
    order = blocks_np.shape[1]
    if order < 1 or order > dimension or not np.issubdtype(blocks_np.dtype, np.integer):
        raise ValueError("blocks must contain integer dimension indices with 1 <= order <= d")
    canonical = np.sort(blocks_np, axis=1)
    if (
        np.any(canonical < 0)
        or np.any(canonical >= dimension)
        or np.any(np.diff(canonical, axis=1) == 0)
        or len(np.unique(canonical, axis=0)) != comb(dimension, order)
        or len(canonical) != comb(dimension, order)
    ):
        raise ValueError("blocks must contain every length-order combination exactly once")
    if scores_np.shape != (len(canonical),):
        raise ValueError("scores must have shape (n_candidates,)")
    if not np.isfinite(scores_np).all() or np.any(scores_np < 0):
        raise ValueError("scores must be finite and nonnegative")
    n_seeds = len(canonical) if n_seeds is None else positive_integer(n_seeds, "n_seeds")
    n_seeds = min(n_seeds, len(canonical))
    return sgm_kernel(
        jnp.asarray(blocks_np),
        jnp.asarray(scores_np, dtype=jnp.result_type(scores_np, 1.0)),
        dimension=dimension,
        n_seeds=n_seeds,
        seed_batch_size=seed_batch_size,
        early_stop=early_stop,
    )
