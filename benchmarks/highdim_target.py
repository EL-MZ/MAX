"""Seeded, exactly normalized, shuffled blocks of random orders and shapes."""

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

from .correlation_levels import CORRELATION_RANGES, signed_correlation, validate_levels

SHAPES = ("gaussian", "banana", "mixture", "sinh_arcsinh", "parity")
LOG_2PI = np.log(2 * np.pi)


@dataclass
class RandomBlock:
    indices: tuple[int, ...]
    shape: str
    correlation: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    curvature: np.ndarray
    skew: np.ndarray
    tail: np.ndarray
    means: np.ndarray
    probabilities: np.ndarray
    correlation_level: str = "original"
    correlation_strength: float | None = None
    physical_transform: np.ndarray | None = None

    def sample(self, rng, n):
        width = len(self.indices)
        latent = rng.normal(size=(n, width)) @ np.linalg.cholesky(self.correlation).T
        if self.shape in ("mixture", "parity"):
            components = rng.choice(len(self.means), size=n, p=self.probabilities)
            latent = 0.45 * latent + self.means[components]
        if self.shape == "banana" and width > 1:
            latent[:, 1:] += self.curvature[1:] * (latent[:, :1] ** 2 - 1)
        if self.shape == "sinh_arcsinh":
            latent = np.sinh((np.arcsinh(latent) + self.skew) / self.tail)
        values = self.scale * latent
        if self.physical_transform is not None:
            values = values @ self.physical_transform.T
        return self.location + values

    def logpdf(self, points):
        width = len(self.indices)
        values = points - self.location
        transform_logdet = 0.0
        if self.physical_transform is not None:
            values = np.linalg.solve(self.physical_transform, values.T).T
            transform_logdet = np.linalg.slogdet(self.physical_transform)[1]
        latent = values / self.scale
        log_jacobian = np.full(len(points), -np.log(self.scale).sum())
        log_jacobian -= transform_logdet
        if self.shape == "banana" and width > 1:
            latent = latent.copy()
            latent[:, 1:] -= self.curvature[1:] * (latent[:, :1] ** 2 - 1)
        if self.shape == "sinh_arcsinh":
            argument = self.tail * np.arcsinh(latent) - self.skew
            # Stable log(cosh(argument)); inverse-Jacobian of the SAS transform.
            log_jacobian += (
                np.log(self.tail)
                + np.logaddexp(argument, -argument)
                - np.log(2)
                - 0.5 * np.log1p(latent**2)
            ).sum(axis=1)
            latent = np.sinh(argument)
        cholesky = np.linalg.cholesky(self.correlation)
        normalizer = 0.5 * width * LOG_2PI + np.log(np.diag(cholesky)).sum()
        if self.shape in ("mixture", "parity"):
            residual = (latent[:, None, :] - self.means) / 0.45
            white = residual @ np.linalg.inv(cholesky).T
            component_logs = -0.5 * (white**2).sum(axis=2) - normalizer
            component_logs -= width * np.log(0.45)
            return log_jacobian + logsumexp(component_logs + np.log(self.probabilities), axis=1)
        white = latent @ np.linalg.inv(cholesky).T
        return log_jacobian - 0.5 * (white**2).sum(axis=1) - normalizer

    def description(self):
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in vars(self).items()
        }


@dataclass
class RandomBlockTarget:
    dimension: int
    factors: tuple[RandomBlock, ...]
    seed: int
    log_evidence: float
    correlation_levels: tuple[str, ...] = ("original",)

    @property
    def blocks(self):
        return tuple(block.indices for block in self.factors)

    def sample(self, rng, n):
        if n < 1:
            raise ValueError("n must be positive")
        values = np.empty((n, self.dimension))
        for block in self.factors:
            values[:, block.indices] = block.sample(rng, n)
        return values

    def block_logpdf(self, points):
        points = np.atleast_2d(points)
        if points.shape[1] != self.dimension:
            raise ValueError("points have the wrong dimension")
        return np.column_stack([block.logpdf(points[:, block.indices]) for block in self.factors])

    def logpdf(self, points):
        # Do not retain an n x n_blocks matrix at 10kD.
        points = np.atleast_2d(points)
        if points.shape[1] != self.dimension:
            raise ValueError("points have the wrong dimension")
        result = np.zeros(len(points))
        for block in self.factors:
            result += block.logpdf(points[:, block.indices])
        return result

    def log_unnormalized(self, points):
        return self.logpdf(points) + self.log_evidence

    def log_prior(self, points):
        points = np.atleast_2d(points)
        return -0.5 * (points**2).sum(axis=1) - 0.5 * self.dimension * LOG_2PI

    def log_likelihood(self, points):
        """Constructed likelihood; proper N(0,I) prior gives the known evidence."""
        return self.log_unnormalized(points) - self.log_prior(points)

    def description(self):
        return {
            "dimension": self.dimension,
            "seed": self.seed,
            "exact_log_evidence": self.log_evidence,
            "correlation_levels": list(self.correlation_levels),
            "correlation_ranges": CORRELATION_RANGES,
            "correlation_note": (
                "Physical affine transforms calibrate Pearson correlation using analytical "
                "moments (Gaussian, banana, mixture) or Gaussian quadrature (sinh_arcsinh). "
                "Pure parity blocks retain pairwise independence."
            ),
            "support": "R^d",
            "construction": "log L = log p + exact_log_evidence - log N(0,I)",
            "factors": [block.description() for block in self.factors],
        }


def _physical_covariance(block):
    """Moments before the optional physical affine transform."""
    covariance = block.correlation.copy()
    if block.shape == "banana":
        curvature = block.curvature.copy()
        curvature[0] = 0
        covariance += 2 * np.outer(curvature, curvature)
    elif block.shape == "mixture":
        mean = block.probabilities @ block.means
        centered = block.means - mean
        covariance = 0.45**2 * covariance + centered.T @ (block.probabilities[:, None] * centered)
    elif block.shape == "sinh_arcsinh":
        # Deterministic moment quadrature; no posterior samples used for calibration.
        nodes, weights = roots_hermitenorm(64)
        weights /= np.sqrt(2 * np.pi)
        transformed = np.sinh((np.arcsinh(nodes[:, None]) + block.skew) / block.tail)
        means = weights @ transformed
        covariance = np.diag(weights @ (transformed**2) - means**2)
        for i in range(len(block.indices)):
            for j in range(i):
                r = block.correlation[i, j]
                second = r * nodes[:, None] + np.sqrt(1 - r**2) * nodes[None, :]
                second = np.sinh((np.arcsinh(second) + block.skew[j]) / block.tail[j])
                moment = np.sum(
                    weights[:, None] * weights[None, :] * transformed[:, i, None] * second
                )
                covariance[i, j] = covariance[j, i] = moment - means[i] * means[j]
    return block.scale[:, None] * covariance * block.scale[None, :]


def make_random_target(
    dimension,
    seed=20261006,
    *,
    orders=(1, 2, 3, 4, 5),
    shapes=SHAPES[:4],
    log_evidence=None,
    correlation_levels=("original",),
):
    """Random block widths, signs, strengths, shapes, scales, and column order.

    Cover each requested order once when d permits, then sample orders uniformly.
    Residual coordinates become singleton blocks. Parity is an optional pure
    higher-order stress test; every proper subset of its coordinates is independent.
    """
    orders = tuple(sorted(set(orders)))
    levels = validate_levels(correlation_levels)
    if not orders or min(orders) < 1 or max(orders) > 5 or dimension < max(orders):
        raise ValueError("require dimension >= max(orders), and orders in 1..5")
    if not shapes or any(shape not in SHAPES for shape in shapes):
        raise ValueError("unknown or empty shapes")
    if log_evidence is not None and not np.isfinite(log_evidence):
        raise ValueError("log_evidence must be finite")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dimension)
    widths, remaining = [], dimension
    if sum(orders) <= dimension:
        widths.extend(rng.permutation(orders).tolist())
        remaining -= sum(orders)
    while remaining:
        eligible = [order for order in orders if order <= remaining]
        width = int(rng.choice(eligible)) if eligible else 1
        widths.append(width)
        remaining -= width
    factors, offset = [], 0
    for width in widths:
        shape = str(rng.choice(shapes))
        if shape == "parity" and width < 3:
            shape = "gaussian"
        loadings = rng.uniform(0.55, 0.92, width) * rng.choice([-1, 1], width)
        correlation = np.diag(1 - loadings**2) + np.outer(loadings, loadings)
        means = np.zeros((1, width))
        probabilities = np.ones(1)
        if shape == "mixture":
            direction = rng.choice([-1, 1], width) * rng.uniform(1.4, 2.4, width)
            means = np.array([-direction, direction])
            probability = rng.uniform(0.3, 0.7)
            probabilities = np.array([probability, 1 - probability])
        if shape == "parity":
            signs = np.array(list(product([-1.0, 1.0], repeat=width - 1)))
            means = 1.5 * np.column_stack((signs, signs.prod(axis=1)))
            probabilities = np.full(len(means), 1 / len(means))
            # Independent component noise preserves proper-subset independence.
            correlation = np.eye(width)
        factors.append(
            RandomBlock(
                indices=tuple(map(int, permutation[offset : offset + width])),
                shape=shape,
                correlation=correlation,
                location=rng.uniform(-1, 1, width),
                scale=np.exp(rng.uniform(np.log(0.6), np.log(1.6), width)),
                curvature=rng.uniform(0.15, 0.45, width) * rng.choice([-1, 1], width),
                skew=rng.uniform(-0.5, 0.5, width),
                tail=rng.uniform(0.85, 1.2, width),
                means=means,
                probabilities=probabilities,
            )
        )
        offset += width
    correlation_rng = np.random.default_rng(np.random.SeedSequence([seed, 73129]))
    eligible_index = 0
    for block in factors:
        if len(block.indices) < 2 or block.shape == "parity":
            continue
        level = levels[eligible_index % len(levels)]
        eligible_index += 1
        if level == "original":
            continue
        rho = float(correlation_rng.uniform(*CORRELATION_RANGES[level]))
        signs = correlation_rng.choice([-1, 1], len(block.indices))
        covariance = _physical_covariance(block)
        scales = np.sqrt(np.diag(covariance))
        desired = scales[:, None] * signed_correlation(len(scales), rho, signs) * scales[None, :]
        block.physical_transform = np.linalg.solve(
            np.linalg.cholesky(covariance).T, np.linalg.cholesky(desired).T
        ).T
        block.correlation_level, block.correlation_strength = level, rho
    return RandomBlockTarget(
        dimension,
        tuple(factors),
        seed,
        -0.1 * dimension if log_evidence is None else float(log_evidence),
        levels,
    )


def load_target(path):
    """Reconstruct the exact target from a saved target.json, without relying on RNG versions."""
    description = json.loads(Path(path).read_text())
    if description.get("family") == "grouped_regression":
        from .regression_target import load_regression_target

        return load_regression_target(path)
    array_names = (
        "correlation",
        "location",
        "scale",
        "curvature",
        "skew",
        "tail",
        "means",
        "probabilities",
    )
    factors = []
    for record in description["factors"]:
        record = dict(record)
        record["indices"] = tuple(record["indices"])
        for name in array_names:
            record[name] = np.asarray(record[name], dtype=np.float64)
        if record.get("physical_transform") is not None:
            record["physical_transform"] = np.asarray(
                record["physical_transform"], dtype=np.float64
            )
        factors.append(RandomBlock(**record))
    target = RandomBlockTarget(
        description["dimension"],
        tuple(factors),
        description["seed"],
        description["exact_log_evidence"],
        tuple(description.get("correlation_levels", ["original"])),
    )
    if sorted(i for block in target.blocks for i in block) != list(range(target.dimension)):
        raise ValueError("saved blocks do not partition the dimensions")
    return target
