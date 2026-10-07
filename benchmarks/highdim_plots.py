"""Correlation-ranked corner projections and bridge history, bounded in dimension."""

import numpy as np

from .correlation_levels import parameter_levels
from .highdim_io import format_runtime


def absolute_correlation(points):
    """Absolute Pearson correlations, treating constant columns as uncorrelated."""
    centered = points - points.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    correlation = np.clip(np.abs(normalized.T @ normalized), 0, 1)
    np.fill_diagonal(correlation, 0)
    return correlation


def select_corner_parameters(target, truth, dimensions=10, count=3):
    """Rank true dependency components by mean |r|; keep components together.

    Use held-out exact posterior draws only. Correlation matrices are computed
    within the target's small components, never across the full target. Prefer
    whole components that fit, splitting only when no remaining component fits.
    With configured levels, each panel prefers its corresponding level (cycling
    when there are more panels than levels), then ranks components within it.
    Panels do not reuse coordinates; small targets produce fewer/smaller panels.
    """
    if dimensions < 1 or count < 1:
        raise ValueError("corner dimensions and count must be positive")
    ranked = []
    levels_by_index = parameter_levels(target)
    requested_levels = getattr(target, "correlation_levels", ("original",))
    for block in target.blocks:
        indices = np.asarray(block, dtype=int)
        correlation = absolute_correlation(truth[:, indices])
        pairs = correlation[np.triu_indices(len(indices), 1)]
        score = float(pairs.mean()) if len(pairs) else 0.0
        order = np.argsort(-correlation.sum(axis=0), kind="stable")
        component_levels = {levels_by_index.get(int(i), "original") for i in indices}
        level = next(iter(component_levels)) if len(component_levels) == 1 else "mixed"
        ranked.append((score, list(map(int, indices[order])), level))
    ranked.sort(key=lambda item: -item[0])
    selections = []
    for panel in range(count):
        selected = []
        preferred = requested_levels[panel % len(requested_levels)]
        while ranked and len(selected) < dimensions:
            space = dimensions - len(selected)
            matching = [i for i, (_, _, level) in enumerate(ranked) if level == preferred]
            available = matching or list(range(len(ranked)))
            position = next((i for i in available if len(ranked[i][1]) <= space), available[0])
            score, indices, level = ranked.pop(position)
            selected.extend(indices[:space])
            if len(indices) > space:
                ranked.insert(position, (score, indices[space:], level))
        if not selected:
            break
        selections.append(tuple(selected))
    return selections


def plot_case(target, truth, proposals, metrics, directory, corner_dimensions=10, corner_count=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    plots = []
    selections = select_corner_parameters(target, truth, corner_dimensions, corner_count)
    levels_by_index = parameter_levels(target)
    for number, indices in enumerate(selections, 1):
        width = len(indices)
        figure, axes = plt.subplots(width, width, figsize=(1.8 * width, 1.8 * width), squeeze=False)
        arrays = {"truth": truth[:, indices], "Morph": proposals[:, indices]}
        bounds = [np.quantile(arrays["truth"][:, i], [0.005, 0.995]) for i in range(width)]
        for i, (low, high) in enumerate(bounds):
            if low == high:
                padding = max(1.0, abs(low)) * 0.05
                bounds[i] = np.array([low - padding, high + padding])
        correlation = absolute_correlation(arrays["truth"])
        pairs = correlation[np.triu_indices(width, 1)]
        mean_correlation = float(pairs.mean()) if len(pairs) else 0.0
        max_correlation = float(pairs.max()) if len(pairs) else 0.0
        levels = sorted({levels_by_index.get(i, "original") for i in indices})
        level_label = "/".join(levels)
        within_pairs = []
        positions = {index: i for i, index in enumerate(indices)}
        for block in target.blocks:
            included = [positions[i] for i in block if i in positions]
            submatrix = correlation[np.ix_(included, included)]
            within_pairs.extend(submatrix[np.triu_indices(len(included), 1)])
        within_mean = float(np.mean(within_pairs)) if within_pairs else 0.0
        for row in range(width):
            for column in range(width):
                axis = axes[row, column]
                if column > row:
                    axis.set_visible(False)
                    continue
                for (name, points), color in zip(arrays.items(), ("black", "tab:blue")):
                    if row == column:
                        axis.hist(
                            points[:, row],
                            bins=30,
                            range=bounds[row],
                            density=True,
                            histtype="step",
                            color=color,
                            label=name,
                        )
                    else:
                        histogram, xedges, yedges = np.histogram2d(
                            points[:, column],
                            points[:, row],
                            bins=25,
                            range=[bounds[column], bounds[row]],
                        )
                        histogram = gaussian_filter(histogram, 1)
                        if histogram.max() > 0:
                            axis.contour(
                                (xedges[1:] + xedges[:-1]) / 2,
                                (yedges[1:] + yedges[:-1]) / 2,
                                histogram.T,
                                levels=histogram.max() * np.array([0.2, 0.6]),
                                colors=[color],
                                linewidths=0.9,
                            )
                axis.set_xlim(bounds[column])
                if row != column:
                    axis.set_ylim(bounds[row])
                if row == width - 1:
                    axis.set_xlabel(f"x{indices[column]}")
                else:
                    axis.tick_params(labelbottom=False)
                if column == 0 and row:
                    axis.set_ylabel(f"x{indices[row]}")
                elif column:
                    axis.tick_params(labelleft=False)
                axis.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(
            f"{target.dimension}D target: correlation-ranked {width}D corner {number}, "
            f"{level_label} correlation, Morph order {metrics['morph_order']}\n"
            f"Exact posterior: max |r| = {max_correlation:.2f}, "
            f"within-component mean |r| = {within_mean:.2f} "
            f"(all-pair mean = {mean_correlation:.2f})"
        )
        figure.tight_layout()
        filename = f"corner_{number:02d}_{width}d.png"
        figure.savefig(directory / filename, dpi=130)
        plt.close(figure)
        plots.append(
            dict(
                file=filename,
                indices=list(indices),
                mean_absolute_correlation=mean_correlation,
                max_absolute_correlation=max_correlation,
                within_component_mean_absolute_correlation=within_mean,
                correlation_levels=levels,
            )
        )
    plot_evidence_history(target, metrics, directory)
    return plots


def plot_evidence_history(target, metrics, directory):
    """Thin displayed updates and highlight geometric restarts over the history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    history = np.asarray(metrics["history"])
    # Sample odd/even updates independently so thinning cannot hide one side
    # of a two-cycle. Do not connect them: dense zigzags obscure the evolution.
    displayed = []
    for parity in (0, 1):
        indices = np.arange(parity, len(history), 2)
        if len(indices):
            positions = np.linspace(0, len(indices) - 1, min(100, len(indices)), dtype=int)
            displayed.extend(indices[positions])
    displayed = np.sort(np.asarray(displayed, dtype=int))
    axis.plot(
        displayed, history[displayed], linestyle="none", marker=".", markersize=4,
        label="MorphZ iteration (thinned)" if len(displayed) < len(history) else "MorphZ iteration",
        color="tab:blue", zorder=2,
    )
    relative_error = metrics.get("relative_error")
    if (
        len(history)
        and np.isfinite(history[-1])
        and relative_error is not None
        and np.isfinite(relative_error)
        and relative_error >= 0
    ):
        # Delta method: SE(log Z) ~= SE(Z) / Z. The benchmark's reported
        # relative error is evaluated at its averaged evidence estimate.
        axis.errorbar(
            len(history) - 1,
            history[-1],
            yerr=relative_error,
            fmt="o",
            color="tab:blue",
            capsize=5,
            elinewidth=1.5,
            label=f"Final ± relative SE ({relative_error:.3g}; log-Z scale)",
        )
    axis.axhline(target.log_evidence, color="black", linestyle="--", label="Exact log Z")
    # History stores the first bridge update from the restart, not its seed.
    # Older benchmark records used the default restart schedule.
    if metrics.get("geometric_restart", True):
        interval = metrics.get("bridge_restart_interval", 1000)
        jumps = np.arange(interval + 1, len(history), interval, dtype=int)
        if len(jumps):
            axis.scatter(
                jumps, history[jumps], color="red", s=28, zorder=5,
                label="Jump iteration (after restart)",
            )
    title = "Bridge converged" if metrics["converged"] else "Bridge did not converge"
    runtime = metrics.get("total_runtime_seconds", metrics.get("workflow_seconds"))
    runtime_label = format_runtime(runtime)
    title += f"\n{target.dimension:,} dimensions | Total runtime: {runtime_label}"
    axis.set(xlabel="Bridge iteration", ylabel="log Z", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "evidence_history.png", dpi=150)
    plt.close(figure)
