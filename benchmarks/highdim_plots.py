"""Small true-block corner projections and bridge history; never allocate d*d plots."""

import numpy as np


def plot_case(target, truth, proposals, metrics, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    for width in range(1, 6):
        block = next((b for b in target.factors if len(b.indices) == width), None)
        if block is None:
            continue
        figure, axes = plt.subplots(width, width, figsize=(2.1 * width, 2.1 * width), squeeze=False)
        arrays = {"truth": truth[:, block.indices], "Morph": proposals[:, block.indices]}
        bounds = [np.quantile(arrays["truth"][:, i], [0.005, 0.995]) for i in range(width)]
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
                    axis.set_xlabel(f"x{block.indices[column]}")
                if column == 0 and row:
                    axis.set_ylabel(f"x{block.indices[row]}")
        axes[0, 0].legend(fontsize=7)
        label = "group projection" if getattr(target, "cross_group_coupling", 0) else "true order"
        figure.suptitle(
            f"{target.dimension}D, {label} {width} ({block.shape}), "
            f"Morph order {metrics['morph_order']}"
        )
        figure.tight_layout()
        figure.savefig(directory / f"corner_order_{width}_{block.shape}.png", dpi=130)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(metrics["history"], marker=".", label="MorphZ iteration")
    axis.axhline(target.log_evidence, color="black", linestyle="--", label="Exact log Z")
    title = "Bridge converged" if metrics["converged"] else "Bridge did not converge"
    axis.set(xlabel="Bridge iteration", ylabel="log Z", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "evidence_history.png", dpi=150)
    plt.close(figure)
