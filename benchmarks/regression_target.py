"""Proper grouped normal--inverse-gamma regression with analytical evidence.

v ~ IG(a0,b0); beta|v ~ N(m0,v V0); y|beta,v ~ N(X beta,v I).
IG density uses b0**a0/Gamma(a0) * v**(-a0-1) * exp(-b0/v).
Evidence is computed from fixed observations and hyperparameters, never assigned.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

LOG_2PI = np.log(2 * np.pi)


def gaussian_log_evidence(design, observations, prior_mean, prior_covariance, noise_covariance):
    """Exact Gaussian regression log Z; solve in observation space using Cholesky."""
    design, observations = np.asarray(design), np.asarray(observations)
    covariance = noise_covariance + design @ prior_covariance @ design.T
    cholesky = cho_factor(covariance, lower=True)
    residual = observations - design @ prior_mean
    log_determinant = 2 * np.log(np.diag(cholesky[0])).sum()
    return float(
        -0.5
        * (len(observations) * LOG_2PI + log_determinant + residual @ cho_solve(cholesky, residual))
    )


def _conditional_normal_logpdf(beta, mean, covariance, variances):
    width = len(mean)
    if width == 0:
        return np.zeros(len(variances))
    cholesky = np.linalg.cholesky(covariance)
    white = (beta - mean) @ np.linalg.inv(cholesky).T
    return (
        -0.5 * width * (LOG_2PI + np.log(variances))
        - np.log(np.diag(cholesky)).sum()
        - 0.5 * (white**2).sum(axis=1) / variances
    )


def _inverse_gamma_logpdf(variances, shape, scale):
    return (
        shape * np.log(scale) - gammaln(shape) - (shape + 1) * np.log(variances) - scale / variances
    )


@dataclass
class RegressionGroup:
    indices: tuple[int, ...]
    design: np.ndarray
    observations: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    a0: float
    b0: float
    variance_coordinate: str = "variance"

    @property
    def shape(self):
        return "normal_inverse_gamma"

    def __post_init__(self):
        width = len(self.indices) - 1
        if not 0 <= width <= 4 or self.design.ndim != 2 or self.design.shape[1] != width:
            raise ValueError("group must have 0..4 coefficients plus one variance")
        if self.observations.shape != (len(self.design),) or len(self.design) < 1:
            raise ValueError("invalid observations")
        if self.prior_mean.shape != (width,) or self.prior_covariance.shape != (width, width):
            raise ValueError("invalid prior dimensions")
        if not np.isfinite(self.a0) or not np.isfinite(self.b0) or self.a0 <= 0 or self.b0 <= 0:
            raise ValueError("inverse-gamma prior must be proper: a0,b0 > 0")
        if self.variance_coordinate not in ("variance", "log_variance"):
            raise ValueError("unknown variance coordinate")
        if not all(
            np.isfinite(a).all()
            for a in (self.design, self.observations, self.prior_mean, self.prior_covariance)
        ):
            raise ValueError("all model arrays must be finite")
        if width:
            prior_cholesky = cho_factor(self.prior_covariance, lower=True)
            precision0 = cho_solve(prior_cholesky, np.eye(width))
            posterior_precision = precision0 + self.design.T @ self.design
            posterior_cholesky = cho_factor(posterior_precision, lower=True)
            self.posterior_covariance = cho_solve(posterior_cholesky, np.eye(width))
            self.posterior_mean = cho_solve(
                posterior_cholesky, precision0 @ self.prior_mean + self.design.T @ self.observations
            )
            displacement = self.posterior_mean - self.prior_mean
            prior_quadratic = displacement @ precision0 @ displacement
            log_det_ratio = (
                -2 * np.log(np.diag(posterior_cholesky[0])).sum()
                - 2 * np.log(np.diag(prior_cholesky[0])).sum()
            )
        else:
            self.posterior_covariance = np.empty((0, 0))
            self.posterior_mean = np.empty(0)
            prior_quadratic, log_det_ratio = 0.0, 0.0
        residual = self.observations - self.design @ self.posterior_mean
        self.an = self.a0 + len(self.observations) / 2
        # Equivalent to y'y + m0'V0^-1 m0 - mn'Vn^-1 mn, avoiding subtraction.
        self.bn = self.b0 + 0.5 * (residual @ residual + prior_quadratic)
        self.log_evidence = float(
            -0.5 * len(self.observations) * LOG_2PI
            + 0.5 * log_det_ratio
            + self.a0 * np.log(self.b0)
            - self.an * np.log(self.bn)
            + gammaln(self.an)
            - gammaln(self.a0)
        )

    def sample(self, rng, n):
        variances = self.bn / rng.gamma(self.an, size=n)
        beta = self.posterior_mean + np.sqrt(variances[:, None]) * (
            rng.normal(size=(n, len(self.posterior_mean)))
            @ np.linalg.cholesky(self.posterior_covariance).T
        )
        coordinate = np.log(variances) if self.variance_coordinate == "log_variance" else variances
        return np.column_stack((beta, coordinate))

    def _valid_values(self, points):
        points = np.atleast_2d(points)
        coordinate = points[:, -1]
        if self.variance_coordinate == "log_variance":
            with np.errstate(over="ignore", under="ignore"):
                variances = np.exp(coordinate)
            jacobian = coordinate
        else:
            variances = coordinate
            jacobian = np.zeros(len(points))
        valid = np.isfinite(points).all(axis=1) & np.isfinite(variances) & (variances > 0)
        return points[valid, :-1], variances[valid], jacobian[valid], valid

    def logpdf(self, points):
        beta, variances, jacobian, valid = self._valid_values(points)
        result = np.full(len(valid), -np.inf)
        result[valid] = (
            _inverse_gamma_logpdf(variances, self.an, self.bn)
            + _conditional_normal_logpdf(
                beta, self.posterior_mean, self.posterior_covariance, variances
            )
            + jacobian
        )
        return result

    def log_prior(self, points):
        beta, variances, jacobian, valid = self._valid_values(points)
        result = np.full(len(valid), -np.inf)
        result[valid] = (
            _inverse_gamma_logpdf(variances, self.a0, self.b0)
            + _conditional_normal_logpdf(beta, self.prior_mean, self.prior_covariance, variances)
            + jacobian
        )
        return result

    def log_likelihood(self, points):
        beta, variances, _, valid = self._valid_values(points)
        result = np.full(len(valid), -np.inf)
        residual = self.observations - beta @ self.design.T
        result[valid] = (
            -0.5 * len(self.observations) * (LOG_2PI + np.log(variances))
            - 0.5 * (residual**2).sum(axis=1) / variances
        )
        return result

    def description(self):
        names = (
            "indices",
            "design",
            "observations",
            "prior_mean",
            "prior_covariance",
            "a0",
            "b0",
            "variance_coordinate",
        )
        result = {
            key: getattr(self, key).tolist()
            if isinstance(getattr(self, key), np.ndarray)
            else getattr(self, key)
            for key in names
        }
        result["posterior"] = {
            "mean": self.posterior_mean.tolist(),
            "covariance_scale": self.posterior_covariance.tolist(),
            "shape": self.an,
            "scale": self.bn,
        }
        result["exact_log_evidence"] = self.log_evidence
        return result


@dataclass
class RegressionTarget:
    dimension: int
    factors: tuple[RegressionGroup, ...]
    seed: int
    cross_group_coupling: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.cross_group_coupling):
            raise ValueError("cross-group coupling must be finite")

    @property
    def coupling_pairs(self):
        eligible = [i for i, group in enumerate(self.factors) if len(group.indices) > 1]
        return tuple(zip(eligible[::2], eligible[1::2]))

    @property
    def blocks(self):
        if self.cross_group_coupling:
            paired = {i for pair in self.coupling_pairs for i in pair}
            return tuple(
                self.factors[a].indices + self.factors[b].indices for a, b in self.coupling_pairs
            ) + tuple(group.indices for i, group in enumerate(self.factors) if i not in paired)
        return tuple(group.indices for group in self.factors)

    @property
    def log_evidence(self):
        return sum(group.log_evidence for group in self.factors)

    def sample(self, rng, n):
        points = np.empty((n, self.dimension))
        for group in self.factors:
            points[:, group.indices] = group.sample(rng, n)
        for a, b in self.coupling_pairs if self.cross_group_coupling else ():
            # Disjoint coefficient shears have determinant one. Variances are unchanged.
            points[:, self.factors[a].indices[0]] += (
                self.cross_group_coupling * points[:, self.factors[b].indices[0]]
            )
        return points

    def _sum(self, points, method):
        points = np.atleast_2d(points)
        if points.shape[1] != self.dimension:
            raise ValueError("wrong target dimension")
        if self.cross_group_coupling:
            points = points.copy()
            for a, b in self.coupling_pairs:
                points[:, self.factors[a].indices[0]] -= (
                    self.cross_group_coupling * points[:, self.factors[b].indices[0]]
                )
        result = np.zeros(len(points))
        for group in self.factors:
            result += getattr(group, method)(points[:, group.indices])
        return result

    def logpdf(self, points):
        return self._sum(points, "logpdf")

    def log_prior(self, points):
        return self._sum(points, "log_prior")

    def log_likelihood(self, points):
        return self._sum(points, "log_likelihood")

    def log_unnormalized(self, points):
        # Evaluate the actual likelihood and prior, independently of log Z.
        return self.log_prior(points) + self.log_likelihood(points)

    def description(self):
        return {
            "family": "grouped_regression",
            "dimension": self.dimension,
            "seed": self.seed,
            "exact_log_evidence": self.log_evidence,
            "cross_group_coupling": self.cross_group_coupling,
            "coupling_note": (
                "Physical coefficients are a determinant-one cross-group linear shear of "
                "the independent conjugate coefficients; evaluate prior and likelihood "
                "through the inverse shear. Evidence is unchanged."
            ),
            "construction": (
                "proper normal-inverse-gamma regression; log f = log prior + log likelihood"
            ),
            "support": "positive variance coordinates"
            if self.factors[0].variance_coordinate == "variance"
            else "R^d (log variances with Jacobian)",
            "factors": [group.description() for group in self.factors],
        }


def make_regression_target(
    dimension,
    seed=20261006,
    *,
    group_orders=(5,),
    variance_coordinate="variance",
    observations_range=(8, 24),
    cross_group_coupling=0.0,
):
    """Default: four coefficients and one variance per group, all columns shuffled.

    Mixed orders 1--5 are optional: each group has order-1 coefficients and one
    variance. At 10kD default this gives exactly 2,000 independent 5D groups.
    """
    orders = tuple(sorted(set(group_orders)))
    if not orders or min(orders) < 1 or max(orders) > 5 or dimension < max(orders):
        raise ValueError("group orders must be 1..5 and fit the dimension")
    if not 1 <= observations_range[0] <= observations_range[1]:
        raise ValueError("invalid observation range")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dimension)
    widths, remaining = [], dimension
    if sum(orders) <= dimension:
        widths.extend(rng.permutation(orders).tolist())
        remaining -= sum(orders)
    while remaining:
        eligible = [order for order in orders if order <= remaining]
        width = int(rng.choice(eligible)) if eligible else remaining
        widths.append(width)
        remaining -= width
    groups, offset = [], 0
    for width in widths:
        coefficients = width - 1
        n = int(rng.integers(observations_range[0], observations_range[1] + 1))
        if coefficients:
            loadings = rng.uniform(0.65, 0.95, coefficients) * rng.choice([-1, 1], coefficients)
            correlation = np.diag(1 - loadings**2) + np.outer(loadings, loadings)
            design = rng.normal(size=(n, coefficients)) @ np.linalg.cholesky(correlation).T
            scales = np.exp(rng.uniform(-0.4, 0.4, coefficients))
            prior_covariance = np.diag(scales**2)
        else:
            design, prior_covariance = np.empty((n, 0)), np.empty((0, 0))
        prior_mean = rng.normal(scale=0.4, size=coefficients)
        a0 = float(rng.uniform(2.1, 4.0))
        b0 = float((a0 - 1) * np.exp(rng.uniform(-0.7, 0.7)))
        true_variance = b0 / rng.gamma(a0)
        true_beta = prior_mean + np.sqrt(true_variance) * (
            np.linalg.cholesky(prior_covariance) @ rng.normal(size=coefficients)
        )
        observations = design @ true_beta + rng.normal(scale=np.sqrt(true_variance), size=n)
        groups.append(
            RegressionGroup(
                tuple(map(int, permutation[offset : offset + width])),
                design,
                observations,
                prior_mean,
                prior_covariance,
                a0,
                b0,
                variance_coordinate,
            )
        )
        offset += width
    return RegressionTarget(dimension, tuple(groups), seed, cross_group_coupling)


def load_regression_target(path):
    description = json.loads(Path(path).read_text())
    groups = []
    for record in description["factors"]:
        width = len(record["indices"]) - 1
        groups.append(
            RegressionGroup(
                tuple(record["indices"]),
                np.asarray(record["design"]).reshape(len(record["observations"]), width),
                np.asarray(record["observations"]),
                np.asarray(record["prior_mean"]),
                np.asarray(record["prior_covariance"]).reshape(width, width),
                record["a0"],
                record["b0"],
                record["variance_coordinate"],
            )
        )
    target = RegressionTarget(
        description["dimension"],
        tuple(groups),
        description["seed"],
        description.get("cross_group_coupling", 0.0),
    )
    if sorted(i for block in target.blocks for i in block) != list(range(target.dimension)):
        raise ValueError("saved groups do not partition the dimensions")
    if not np.isclose(target.log_evidence, description["exact_log_evidence"], rtol=0, atol=1e-8):
        raise ValueError("saved evidence disagrees with model data and hyperparameters")
    return target
