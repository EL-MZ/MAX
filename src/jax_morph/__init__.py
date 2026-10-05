"""JAX Morph density approximation and post-processing evidence estimation."""

from .bridge import BridgeResult, bridge_kernel, bridge_sampling
from .correlation import estimate_total_correlations
from .evidence import EvidenceResult, estimate_evidence
from .kde import KDEState, fit_kde, kde_log_prob, sample_kde
from .morph import MorphFit, MorphState, fit, log_prob, sample
from .sgm import SGMResult, candidate_blocks, seeded_greedy_maximization, sgm_kernel

__all__ = [
    "BridgeResult",
    "EvidenceResult",
    "KDEState",
    "MorphFit",
    "MorphState",
    "SGMResult",
    "bridge_kernel",
    "bridge_sampling",
    "candidate_blocks",
    "estimate_evidence",
    "estimate_total_correlations",
    "fit",
    "fit_kde",
    "kde_log_prob",
    "log_prob",
    "sample",
    "sample_kde",
    "seeded_greedy_maximization",
    "sgm_kernel",
]
