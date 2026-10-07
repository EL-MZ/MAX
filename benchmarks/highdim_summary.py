"""Compact text reports for posterior dependence, Morph accuracy and runtime."""

from collections import Counter

import numpy as np

from .correlation_levels import CORRELATION_RANGES
from .highdim_io import format_runtime
from .regression_target import RegressionTarget


def posterior_dependence(target, truth=None, threshold=0.3):
    """Known components plus empirical linear correlations within those components.

    Component membership includes nonlinear and higher-order dependence, even
    when Pearson correlation is zero. Never allocate a full dimension matrix.
    """
    components = target.blocks
    dependent = {i for block in components if len(block) > 1 for i in block}
    types = Counter()
    if isinstance(target, RegressionTarget):
        for group in target.factors:
            if len(group.indices) > 2:
                types["linear coefficient correlation (base groups)"] += 1
            if len(group.indices) > 1:
                types["nonlinear coefficient-variance scale dependence (base groups)"] += 1
                if group.variance_shear is not None:
                    types["linear coefficient-variance correlation (calibrated base groups)"] += 1
        if target.cross_group_coupling:
            types["cross-group linear shear (coupled group pairs)"] = len(target.coupling_pairs)
    else:
        labels = {
            "gaussian": "linear Gaussian correlation",
            "banana": "nonlinear quadratic banana + latent Gaussian correlation",
            "mixture": "multimodal mixture + within-component linear correlation",
            "sinh_arcsinh": "nonlinear skew/tail transform + latent Gaussian correlation",
            "parity": "higher-order parity dependence (pairwise independent)",
        }
        for block in target.factors:
            if len(block.indices) > 1:
                types[labels.get(block.shape, block.shape)] += 1
                if getattr(block, "physical_transform", None) is not None:
                    types["calibrated physical affine correlation"] += 1
    correlated, positive, negative = set(), 0, 0
    has_samples = truth is not None and len(truth) >= 2
    if has_samples:
        for block in components:
            if len(block) < 2:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.nan_to_num(np.corrcoef(truth[:, block], rowvar=False))
            rows, columns = np.triu_indices(len(block), 1)
            values = correlation[rows, columns]
            selected = np.abs(values) >= threshold
            positive += int(np.sum(values >= threshold))
            negative += int(np.sum(values <= -threshold))
            correlated.update(block[i] for i in rows[selected])
            correlated.update(block[i] for i in columns[selected])
    return dict(
        dependent_parameters=len(dependent),
        independent_parameters=target.dimension - len(dependent),
        component_order_counts=dict(Counter(map(len, components))),
        base_group_order_counts=dict(Counter(len(block.indices) for block in target.factors)),
        type_counts=dict(types),
        correlation_level_group_counts=dict(
            Counter(
                getattr(group, "correlation_level", "original")
                for group in target.factors
                if len(group.indices) > 1
            )
        ),
        correlation_level_parameter_counts=dict(
            Counter(
                getattr(group, "correlation_level", "original")
                for group in target.factors
                if len(group.indices) > 1
                for _ in group.indices
            )
        ),
        linear_correlation_threshold=threshold,
        linear_correlated_parameters=len(correlated) if has_samples else None,
        positive_linear_pairs=positive if has_samples else None,
        negative_linear_pairs=negative if has_samples else None,
        correlation_sample_count=len(truth) if truth is not None else None,
    )


def _order_counts(counts):
    counts = {int(order): count for order, count in counts.items()}
    labels = {1: "singletons", 2: "pairs", 3: "triplets", 4: "quartets", 5: "quintuplets"}
    return ", ".join(
        f"{labels.get(order, f'{order}-parameter groups')}:{counts.get(order, 0)}"
        for order in sorted(set(labels) | set(counts))
    )


def _number(value):
    return f"{value:.9g}" if value is not None and np.isfinite(value) else "unavailable"


def write_case_summary(case, directory):
    """Always write summary.txt, including in plots-only mode."""
    dimension = case["dimension"]
    dependence = case["posterior_dependence"]
    dependent = dependence["dependent_parameters"]
    runtime = case.get("total_runtime_seconds", case.get("workflow_seconds"))
    error = case["relative_error"]
    relative_error = f"{error:.2%}" if error is not None and np.isfinite(error) else "unavailable"
    survived = case.get("proposal_survived_count")
    if survived is None and case.get("outside_support_fraction") is not None:
        survived = int(round((1 - case["outside_support_fraction"]) * case["proposal_count"]))
    survival = (
        f"{survived:,}/{case['proposal_count']:,} ({survived / case['proposal_count']:.1%})"
        if survived is not None
        else "unavailable"
    )
    lines = [
        f"Dimensions: {dimension:,} | Family: {case['family']} | Seed: {case['seed']}",
        f"Samples: fit={case['fit_count']:,}, posterior={case['posterior_count']:,}, "
        f"Morph proposals={case['proposal_count']:,}",
        f"Survived proposals (finite target density): {survival} | "
        f"Importance ESS: {_number(case['importance_ess'])}",
        f"Evidence log Z: {_number(case['log_evidence'])} | "
        f"Exact: {_number(case['exact_log_evidence'])} | "
        f"Error: {_number(case['log_evidence_error'])}",
        f"Relative SE: {relative_error} | "
        f"Converged: {case['converged']} | Status: {case['status']}",
        f"KL(Post || Morph): {_number(case['forward_kl'])} nats | "
        f"MC SE: {_number(case['forward_kl_mc_se'])} | "
        f"Per dimension: {_number(case['kl_per_dimension'])}",
        "",
        "Runtime (h:mm:ss; total computation excludes plot rendering):",
        f"  Total: {format_runtime(runtime)}",
        f"  Target setup: {format_runtime(case.get('target_setup_seconds'))} | "
        f"Exact-proposal bridge control: {format_runtime(case.get('analytic_control_seconds'))}",
        f"  Morph fit: {format_runtime(case['morph_seconds'])} | "
        f"Evidence workflow total: {format_runtime(case['bridge_sampling_total_seconds'])}",
        f"  Posterior Morph density: {format_runtime(case['posterior_log_prob_seconds'])} | "
        f"Proposal sampling: {format_runtime(case['proposal_sample_seconds'])}",
        f"  Proposal Morph density: {format_runtime(case['proposal_log_prob_seconds'])} | "
        f"Proposal target density: {format_runtime(case['proposal_target_seconds'])} | "
        f"Bridge iterations: {format_runtime(case['bridge_seconds'])}",
    ]
    if case.get("fit_reused"):
        lines.append(f"  Reused fit; original fit: {format_runtime(case['original_fit_seconds'])}")
    lines.extend(
        [
            "",
            f"Parameters in known dependent posterior components: {dependent:,}/{dimension:,} "
            f"({dependent / dimension:.1%}); independent: {dependence['independent_parameters']:,}",
            "Posterior component orders: " + _order_counts(dependence["component_order_counts"]),
            "Morph selected orders: " + _order_counts(Counter(map(len, case["blocks"]))),
            "Orders count disjoint groups, not every subset or pure interaction orders.",
        ]
    )
    if dependence["component_order_counts"] != dependence["base_group_order_counts"]:
        lines.append(
            "Posterior base-group orders: " + _order_counts(dependence["base_group_order_counts"])
        )
    level_counts = dependence.get("correlation_level_group_counts", {})
    if level_counts:
        lines.append("Configured correlation levels (before optional cross-group coupling):")
        for level, count in level_counts.items():
            interval = CORRELATION_RANGES.get(level)
            strength = f" |r|={interval[0]:.2f}..{interval[1]:.2f}" if interval else ""
            parameters = dependence["correlation_level_parameter_counts"][level]
            lines.append(f"  {level}:{count} base groups, {parameters} parameters{strength}")
    linear = dependence["linear_correlated_parameters"]
    if linear is None:
        lines.append("Empirical linear correlation: unavailable (insufficient or unsaved draws).")
    else:
        lines.append(
            f"Empirical |Pearson r| >= {dependence['linear_correlation_threshold']:.2f}: "
            f"{linear:,}/{dimension:,} parameters ({linear / dimension:.1%}); "
            f"positive pairs:{dependence['positive_linear_pairs']}, "
            f"negative pairs:{dependence['negative_linear_pairs']} "
            f"(n={dependence['correlation_sample_count']:,}; within known components)."
        )
    lines.append("Posterior dependence types (known target construction):")
    lines.extend(f"  {name}: {count}" for name, count in dependence["type_counts"].items())
    if not dependence["type_counts"]:
        lines.append("  Independent singleton parameters only.")
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
