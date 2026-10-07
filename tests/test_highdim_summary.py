"""Reports distinguish nonlinear dependence from empirical linear correlation."""

import numpy as np

from benchmarks.highdim_summary import posterior_dependence
from benchmarks.highdim_target import make_random_target
from benchmarks.regression_target import make_regression_target


def test_parity_is_counted_as_dependent_despite_zero_pairwise_correlation():
    target = make_random_target(5, seed=7, orders=[5], shapes=["parity"])
    record = posterior_dependence(target, target.sample(np.random.default_rng(3), 4000))
    assert record["dependent_parameters"] == 5
    assert record["component_order_counts"] == {5: 1}
    assert record["linear_correlated_parameters"] == 0
    assert record["type_counts"] == {"higher-order parity dependence (pairwise independent)": 1}


def test_coupled_components_and_base_groups_have_distinct_order_counts():
    target = make_regression_target(21, seed=2, cross_group_coupling=0.8)
    record = posterior_dependence(target)
    assert record["component_order_counts"] == {10: 2, 1: 1}
    assert record["base_group_order_counts"] == {5: 4, 1: 1}
    assert record["dependent_parameters"] == 20
    assert record["independent_parameters"] == 1
    assert record["linear_correlated_parameters"] is None
    assert record["type_counts"]["cross-group linear shear (coupled group pairs)"] == 2
    assert (
        record["type_counts"]["nonlinear coefficient-variance scale dependence (base groups)"] == 4
    )


def test_linear_statistics_count_unique_parameters_and_positive_negative_pairs():
    target = make_random_target(6, seed=4, orders=[2], shapes=["gaussian"])
    rng = np.random.default_rng(9)
    truth = np.zeros((1000, 6))
    for block, sign in zip(target.blocks, (1, -1, 0)):
        latent = rng.normal(size=len(truth))
        truth[:, block[0]] = latent
        truth[:, block[1]] = sign * latent
    record = posterior_dependence(target, truth)
    assert record["linear_correlated_parameters"] == 4
    assert record["positive_linear_pairs"] == 1
    assert record["negative_linear_pairs"] == 1
    assert record["component_order_counts"] == {2: 3}
