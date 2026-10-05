"""Host-side validation, kept outside compiled numerical kernels."""

import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def samples_and_weights(samples, weights=None):
    """Validate (draws, dimensions) data and normalize nonnegative weights."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or min(samples.shape) < 1 or samples.shape[0] < 2:
        raise ValueError("samples must have shape (n >= 2, d >= 1)")
    if not np.isrealobj(samples) or not np.isfinite(samples).all():
        raise ValueError("samples must be finite and real")
    if not np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(float)
    weights = np.ones(len(samples)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(samples),):
        raise ValueError("weights must have shape (n,)")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    # Scaling before summing also accepts very large, unnormalized input weights.
    weights = weights / weights.max()
    weights = weights / weights.sum()
    if np.count_nonzero(weights) < 2:
        raise ValueError("at least two samples must have positive weight")
    return samples, weights
