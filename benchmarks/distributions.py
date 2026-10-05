"""Analytic block targets inspired by EL-MZ/Morph-approx-lab.

The lab's Gaussian-copula construction is implemented without Streamlit or a
global RNG. Unlike its plotting likelihood, this logpdf preserves -inf outside
the support. Nonlinear and mixture blocks are additional stress tests.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.special import logsumexp

SCENARIOS = (
    "independent_mixed",
    "strong_pairs",
    "weak_pairs",
    "triplets",
    "banana_pairs",
    "multimodal_pairs",
)


@dataclass
class BlockTarget:
    name: str
    dimension: int
    canonical_blocks: tuple
    permutation: np.ndarray
    marginals: tuple
    correlations: tuple
    labels: tuple

    @property
    def blocks(self):
        inverse = np.argsort(self.permutation)
        return tuple(tuple(int(inverse[i]) for i in block) for block in self.canonical_blocks)

    @property
    def restricted_support(self):
        return any(np.isfinite(dist.support()).any() for dist in self.marginals)

    def sample(self, rng, n):
        values = np.empty((n, self.dimension))
        for block, correlation in zip(self.canonical_blocks, self.correlations):
            size = len(block)
            if self.name == "banana_pairs" and size == 2:
                x = rng.normal(size=n)
                values[:, block] = np.column_stack((x, 0.7 * (x**2 - 1) + 0.2 * rng.normal(size=n)))
            elif self.name == "multimodal_pairs" and size == 2:
                signs = rng.choice([-1.0, 1.0], size=n)
                noise = rng.multivariate_normal(np.zeros(2), correlation * 0.25, size=n)
                values[:, block] = signs[:, None] * 2 + noise
            else:
                z = rng.multivariate_normal(np.zeros(size), correlation, size=n)
                u = np.clip(stats.norm.cdf(z), np.finfo(float).eps, 1 - np.finfo(float).eps)
                for j, index in enumerate(block):
                    values[:, index] = self.marginals[index].ppf(u[:, j])
        return values[:, self.permutation]

    def block_logpdf(self, points):
        """One log-density column per independent true block, in physical order."""
        points = np.atleast_2d(points)
        if points.shape[1] != self.dimension:
            raise ValueError("points have the wrong dimension")
        canonical = points[:, np.argsort(self.permutation)]
        columns = []
        for block, correlation in zip(self.canonical_blocks, self.correlations):
            values = canonical[:, block]
            if self.name == "banana_pairs" and len(block) == 2:
                x, y = values.T
                logs = stats.norm.logpdf(x) + stats.norm.logpdf(y, 0.7 * (x**2 - 1), 0.2)
            elif self.name == "multimodal_pairs" and len(block) == 2:
                logs = logsumexp(
                    [
                        stats.multivariate_normal.logpdf(
                            values, np.full(2, mean), correlation * 0.25
                        )
                        for mean in [-2.0, 2.0]
                    ],
                    axis=0,
                ) - np.log(2)
            else:
                marginal_logs = np.column_stack(
                    [self.marginals[index].logpdf(values[:, j]) for j, index in enumerate(block)]
                )
                inside = np.isfinite(marginal_logs).all(axis=1)
                logs = np.full(len(values), -np.inf)
                if inside.any():
                    u = np.column_stack(
                        [
                            self.marginals[index].cdf(values[inside, j])
                            for j, index in enumerate(block)
                        ]
                    )
                    z = stats.norm.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
                    copula = stats.multivariate_normal.logpdf(z, np.zeros(len(block)), correlation)
                    copula -= stats.norm.logpdf(z).sum(axis=1)
                    logs[inside] = copula + marginal_logs[inside].sum(axis=1)
            columns.append(logs)
        return np.column_stack(columns)

    def logpdf(self, points):
        return self.block_logpdf(points).sum(axis=1)

    def description(self):
        return {
            "scenario": self.name,
            "dimension": self.dimension,
            "true_blocks": self.blocks,
            "permutation_new_to_old": self.permutation.tolist(),
            "labels": self.labels,
            "latent_correlations": [value.tolist() for value in self.correlations],
            "marginal_note": (
                "Listed marginals apply to copula blocks and singleton tails. "
                "Banana and mixture pairs use the explicit joint parameters below."
            ),
            "marginals": [
                {"distribution": dist.dist.name, "args": dist.args, "kwargs": dist.kwds}
                for dist in self.marginals
            ],
            "banana_parameters": {"curvature": 0.7, "noise_sd": 0.2},
            "mixture_parameters": {"means": [-2, 2], "covariance_scale": 0.25},
        }


def make_target(name, dimension, seed=0):
    if name not in SCENARIOS or dimension < 1:
        raise ValueError("unknown scenario or invalid dimension")
    rng = np.random.default_rng(seed)
    width = 1 if name == "independent_mixed" else 3 if name == "triplets" else 2
    blocks = tuple(tuple(range(i, min(i + width, dimension))) for i in range(0, dimension, width))
    rho = 0.35 if name == "weak_pairs" else 0.7 if name == "triplets" else 0.85
    correlations = tuple(
        (1 - rho) * np.eye(len(b)) + rho * np.ones((len(b), len(b))) for b in blocks
    )
    families = (
        stats.norm(loc=-1, scale=1.2),
        stats.gamma(a=2.5, scale=0.8),
        stats.beta(a=2.2, b=4.0),
        stats.lognorm(s=0.6, scale=1.1),
        stats.expon(scale=0.9),
        stats.uniform(loc=-2, scale=3),
        stats.loguniform(a=0.2, b=4),
    )
    marginals = tuple(families[i % len(families)] for i in range(dimension))
    if name in ("banana_pairs", "multimodal_pairs"):
        marginals = tuple(stats.norm() for _ in range(dimension))
    permutation = rng.permutation(dimension)
    families_by_column = [dist.dist.name for dist in marginals]
    if name == "banana_pairs":
        families_by_column = ["norm" if i % 2 == 0 else "banana" for i in range(dimension)]
    elif name == "multimodal_pairs":
        families_by_column = [
            "mixture" if i < dimension - dimension % 2 else "norm" for i in range(dimension)
        ]
    labels = tuple(f"x{old + 1}:{families_by_column[old]}" for old in permutation)
    return BlockTarget(name, dimension, blocks, permutation, marginals, correlations, labels)


def block_metrics(true_blocks, estimated_blocks):
    """Pair recall measures retained dependencies; precision penalizes mergers.

    True singleton targets have no dependent pairs, so their recall is undefined.
    Forced grouping at order > 1 is still reported as false-positive edges.
    """

    def edges(blocks):
        return {tuple(sorted(pair)) for block in blocks for pair in combinations(block, 2)}

    truth, predicted = edges(true_blocks), edges(estimated_blocks)
    true_sets = {frozenset(block) for block in true_blocks}
    predicted_sets = {frozenset(block) for block in estimated_blocks}
    hits = len(truth & predicted)
    return {
        "pair_precision": hits / len(predicted) if predicted else None,
        "pair_recall": hits / len(truth) if truth else None,
        "true_pair_count": len(truth),
        "selected_pair_count": len(predicted),
        "retained_true_pairs": hits,
        "false_positive_pairs": len(predicted - truth),
        "exact_block_recall": len(true_sets & predicted_sets) / len(true_sets),
        "exact_partition": true_sets == predicted_sets,
    }
