"""Shared, reproducible correlation controls for benchmark targets."""

import numpy as np

CORRELATION_RANGES = {"low": (0.10, 0.30), "medium": (0.40, 0.65), "strong": (0.80, 0.95)}
CORRELATION_LEVELS = (*CORRELATION_RANGES, "original")
DEFAULT_CORRELATION_LEVELS = ["low", "medium", "strong"]


def validate_levels(levels):
    levels = tuple(dict.fromkeys(levels))
    if not levels or any(level not in CORRELATION_LEVELS for level in levels):
        raise ValueError("correlation levels must be low, medium, strong or original")
    return levels


def signed_correlation(width, strength, signs):
    """Positive definite equicorrelation with independently signed coordinates."""
    return (1 - strength) * np.eye(width) + strength * np.outer(signs, signs)


def parameter_levels(target):
    return {
        index: getattr(group, "correlation_level", "original")
        for group in getattr(target, "factors", ())
        for index in group.indices
    }
