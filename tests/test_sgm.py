from itertools import combinations

import jax
import numpy as np
import pytest

from jax_morph import candidate_blocks, seeded_greedy_maximization, sgm_kernel


def reference_sgm(blocks, scores, dimension, n_seeds):
    """Direct, readable transcription of paper Algorithm 1, independent of JAX."""
    ranked = np.argsort(-scores, kind="stable")
    totals, runs = [], []
    for seed in ranked[:n_seeds]:
        chosen = [seed]
        used = set(blocks[seed])
        for index in ranked:
            if len(chosen) == dimension // blocks.shape[1]:
                break
            if not used.intersection(blocks[index]):
                chosen.append(index)
                used.update(blocks[index])
        runs.append(chosen)
        totals.append(sum(scores[index] for index in chosen))
    return np.asarray(runs[np.argmax(totals)]), np.asarray(totals)


@pytest.mark.parametrize("dimension,order", [(7, 2), (9, 3), (8, 4), (4, 4), (5, 1)])
@pytest.mark.parametrize("n_seeds", [1, 5, None])
@pytest.mark.parametrize("early_stop", [False, True])
def test_matches_paper_reference(dimension, order, n_seeds, early_stop):
    blocks = np.asarray(candidate_blocks(dimension, order))
    scores = np.random.default_rng(61).uniform(size=len(blocks))
    count = len(blocks) if n_seeds is None else min(n_seeds, len(blocks))
    expected, totals = reference_sgm(blocks, scores, dimension, count)
    actual = seeded_greedy_maximization(
        blocks,
        scores,
        dimension=dimension,
        n_seeds=n_seeds,
        seed_batch_size=3,
        early_stop=early_stop,
    )
    np.testing.assert_array_equal(actual.candidate_indices, expected)
    np.testing.assert_allclose(actual.seed_totals, totals, rtol=1e-12)
    selected = np.asarray(actual.blocks).ravel()
    assert len(set(selected)) == (dimension // order) * order


def test_ties_are_stable_and_kernel_accepts_traced_scores():
    blocks = candidate_blocks(7, 2)
    scores = np.zeros(len(blocks))
    expected, totals = reference_sgm(np.asarray(blocks), scores, 7, 4)
    compiled = jax.jit(
        lambda values: sgm_kernel(blocks, values, dimension=7, n_seeds=4, seed_batch_size=2)
    )
    result = compiled(scores)
    np.testing.assert_array_equal(result.candidate_indices, expected)
    np.testing.assert_allclose(result.seed_totals, totals)


def test_more_seeds_cannot_reduce_retained_score_and_exact_small_optimum():
    blocks = np.asarray(candidate_blocks(6, 2))
    scores = np.random.default_rng(10).uniform(size=len(blocks))
    one = seeded_greedy_maximization(blocks, scores, dimension=6, n_seeds=1)
    all_seeds = seeded_greedy_maximization(blocks, scores, dimension=6)
    assert all_seeds.total_score >= one.total_score
    optimum = max(
        scores[list(indices)].sum()
        for indices in combinations(range(len(blocks)), 3)
        if len(set(blocks[list(indices)].ravel())) == 6
    )
    assert all_seeds.total_score <= optimum + 1e-12


def test_candidate_limit_and_incomplete_collection():
    with pytest.raises(ValueError, match="exceed"):
        candidate_blocks(30, 5, max_candidates=10_000)
    blocks = candidate_blocks(5, 2)
    with pytest.raises(ValueError, match="every"):
        seeded_greedy_maximization(blocks[:-1], np.ones(len(blocks) - 1), dimension=5)
    with pytest.raises(ValueError, match="nonnegative"):
        seeded_greedy_maximization(blocks, -np.ones(len(blocks)), dimension=5)
