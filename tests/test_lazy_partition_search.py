"""Correctness checks for the isolated on-demand-TC research prototype."""

from itertools import combinations

import numpy as np
import pytest

from benchmarks.lazy_partition_search import (
    TCOracle,
    canonical,
    pair_screen,
    restricted_optimum,
    screened_pool,
    stochastic_swaps,
    validate_partition,
)
from jax_morph.correlation import estimate_total_correlations


def test_cached_tc_matches_production_and_enforces_budget():
    x = np.random.default_rng(81).normal(size=(32, 7))
    oracle = TCOracle(x, 3)
    blocks = [(0, 1, 2), (2, 3, 4), (1, 5, 6)]
    expected = estimate_total_correlations(x, np.asarray(blocks)).scores
    np.testing.assert_allclose(oracle.evaluate(blocks), expected, atol=1e-10)
    work = oracle.kernel_blocks
    np.testing.assert_allclose(oracle.evaluate([blocks[0][::-1]]), expected[:1], atol=1e-10)
    assert oracle.kernel_blocks == work and len(oracle.cache) == 3
    with pytest.raises(ValueError, match="budget"):
        oracle.evaluate([(0, 3, 6)])


class TableOracle:
    def __init__(self, scores, budget):
        self.scores, self.budget, self.cache = scores, budget, {}

    def evaluate(self, blocks):
        result = []
        for b in blocks:
            b = tuple(sorted(b))
            if b not in self.cache:
                assert len(self.cache) < self.budget
                self.cache[b] = self.scores[b]
            result.append(self.cache[b])
        return np.asarray(result)


def test_restricted_solver_matches_enumeration():
    rng = np.random.default_rng(7)
    blocks = list(combinations(range(6), 2))
    scores = dict(zip(blocks, rng.random(len(blocks))))
    oracle = TableOracle(scores, len(blocks))
    chosen, status = restricted_optimum(oracle, blocks, 6, 2, [(0, 1), (2, 3), (4, 5)])
    expected = max(
        sum(scores[b] for b in partition)
        for partition in combinations(blocks, 3)
        if len(set(i for b in partition for i in b)) == 6
    )
    assert status["pool_optimal"]
    assert sum(scores[b] for b in chosen) == pytest.approx(expected)


def test_screen_pool_feasible_and_bounded_without_full_enumeration():
    rng = np.random.default_rng(77)
    matrix = pair_screen(rng.normal(size=(100, 25)))
    pool, scaffold = screened_pool(matrix, 3, 50, rng)
    assert len(pool) == len(set(pool)) == 50
    assert set(scaffold).issubset(pool)
    validate_partition(scaffold, 25, 3)


def test_stochastic_moves_allow_a_residual_singleton_to_enter_a_block():
    scores = {b: float(4 in b) for b in combinations(range(5), 2)}
    oracle = TableOracle(scores, 10)
    initial = canonical([(0, 1), (2, 3)])
    chosen, _ = stochastic_swaps(oracle, initial, 5, 2, np.random.default_rng(42))
    validate_partition(chosen, 5, 2)
    assert sum(scores[b] for b in chosen) == 1
    assert len(oracle.cache) <= 10
