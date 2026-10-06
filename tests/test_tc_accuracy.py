import numpy as np
import pytest

from benchmarks.tc_accuracy import difference_summary, nested_subsets


def test_differences_match_candidates_before_pooling():
    reference = np.array([1.0, 3.0])
    estimates = np.array([[2.0, 1.0], [3.0, 2.0]])
    result = difference_summary(estimates, reference)
    errors = np.array([1.0, -2.0, 2.0, -1.0])
    assert result["mean_difference_nats"] == 0.0
    assert result["std_difference_nats"] == pytest.approx(np.std(errors, ddof=1))
    assert result["mean_absolute_difference_nats"] == 1.5
    assert result["std_absolute_difference_nats"] == pytest.approx(np.std(abs(errors), ddof=1))
    assert result["std_trial_mean_difference_nats"] == pytest.approx(np.sqrt(0.5))


def test_nested_subsets_are_reproducible_without_duplicate_rows():
    subsets = nested_subsets(1000, [10, 50, 500], 5, 123)
    np.testing.assert_array_equal(subsets, nested_subsets(1000, [10, 50, 500], 5, 123))
    assert subsets.shape == (5, 500)
    assert not np.array_equal(subsets[0], subsets[1])
    for subset in subsets:
        assert len(set(subset)) == 500
        assert subset.min() >= 0 and subset.max() < 1000


def test_reference_scores_produce_zero_differences():
    result = difference_summary(np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([1.0, 2.0]))
    assert result["mean_difference_nats"] == result["std_difference_nats"] == 0.0
    assert result["mean_absolute_difference_nats"] == 0.0
