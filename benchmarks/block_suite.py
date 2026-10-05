"""Fit shuffled analytic block targets and save KL, speed, and corner plots.

Run from the repository root: python -m benchmarks.block_suite --output DIR
All sample batches contain at most 1,000 values for each parameter. Training,
forward-KL evaluation, and proposal draws use separate seeded random streams.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import corner
import jax
import jaxlib
import matplotlib
import numpy as np
import scipy
from scipy import stats
from scipy.special import logsumexp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from benchmarks.distributions import SCENARIOS, block_metrics, make_target  # noqa: E402
from jax_morph import fit, fit_kde, log_prob, sample, sgm_kernel  # noqa: E402
from jax_morph.morph import MorphState  # noqa: E402

jax.config.update("jax_enable_x64", True)
COLORS = {1: "#8b949e", 2: "#0072b2", 3: "#d55e00"}


def timed(function, repeats=1):
    start = perf_counter()
    result = function()
    jax.block_until_ready(result)
    first = perf_counter() - start
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(function())
        durations.append(perf_counter() - start)
    return result, first, float(np.median(durations))


def kl_estimate(log_p, log_q):
    ratios = np.asarray(log_p) - np.asarray(log_q)
    if not np.isfinite(ratios).all():
        raise ValueError("forward KL needs finite log-density ratios")
    mean = float(ratios.mean())
    se = float(ratios.std(ddof=1) / np.sqrt(len(ratios)))
    return {"forward_kl": mean, "forward_kl_se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def proposal_metrics(target, state, draws):
    log_p = target.logpdf(draws)
    log_q = np.asarray(log_prob(state, draws))
    inside = np.isfinite(log_p)
    leakage = float(1 - inside.mean())
    log_weights = log_p - log_q
    if inside.any():
        normalized = np.exp(log_weights - logsumexp(log_weights))
        ess = float(1 / np.sum(normalized**2))
    else:
        ess = 0.0
    infinite_reverse = target.restricted_support or not inside.all()
    reverse = kl_estimate(log_q, log_p) if not infinite_reverse else None
    return {
        "support_leakage_fraction": leakage,
        "importance_ess": ess,
        "importance_ess_fraction": ess / len(draws),
        "reverse_kl": reverse["forward_kl"] if reverse else None,
        "reverse_kl_se": reverse["forward_kl_se"] if reverse else None,
        "reverse_kl_infinite": bool(infinite_reverse),
        "reverse_kl_note": (
            "Gaussian KDE has positive mass outside true support: reverse KL is infinite."
            if infinite_reverse
            else "Monte Carlo reverse KL on independent proposal draws."
        ),
    }


def evaluate_fit(target, training, evaluation, truth_logs, order, tc_samples, seed, fit_repeats):
    fitted, fit_first, fit_warm = timed(
        lambda: fit(
            training,
            order=order,
            n_seeds=64,
            max_tc_samples=tc_samples,
            key=jax.random.key(seed),
            candidate_batch_size=8,
            chunk_size=128,
        ),
        repeats=fit_repeats,
    )
    logs, density_first, density_warm = timed(
        lambda: log_prob(fitted.state, evaluation), repeats=3
    )
    draws, sample_first, sample_warm = timed(
        lambda: sample(jax.random.key(seed + 1), fitted.state, len(training)), repeats=3
    )
    draws = np.asarray(draws)
    blocks = [list(block) for block in fitted.state.blocks]
    flattened = [i for block in blocks for i in block]
    assert sorted(flattened) == list(range(target.dimension)), "invalid partition"
    assert len(flattened) == len(set(flattened)), "overlapping partition"
    assert np.isfinite(np.asarray(logs)).all(), "nonfinite Morph density"
    sgm_first = sgm_warm = 0.0
    if order > 1:
        diagnostics = fitted.diagnostics
        _, sgm_first, sgm_warm = timed(
            lambda: sgm_kernel(
                diagnostics.candidates,
                diagnostics.total_correlations,
                dimension=target.dimension,
                n_seeds=min(64, len(diagnostics.candidates)),
            ),
            repeats=5,
        )
    record = {
        "scenario": target.name,
        "dimension": target.dimension,
        "order": order,
        "training_samples": len(training),
        "evaluation_samples": len(evaluation),
        "proposal_samples": len(draws),
        "tc_samples": fitted.diagnostics.tc_sample_count,
        "candidates": len(fitted.diagnostics.candidates),
        "sgm_seeds": len(fitted.diagnostics.seed_totals),
        "seed": seed,
        "true_blocks": target.blocks,
        "selected_blocks": blocks,
        "fit_first_call_seconds": fit_first,
        "fit_warm_median_seconds": fit_warm,
        "logprob_first_call_seconds": density_first,
        "logprob_warm_median_seconds": density_warm,
        "sample_first_call_seconds": sample_first,
        "sample_warm_median_seconds": sample_warm,
        "sgm_first_call_seconds": sgm_first,
        "sgm_warm_median_seconds": sgm_warm,
        "fit_warm_repeats": fit_repeats,
        "retained_total_correlation": float(fitted.diagnostics.retained_total_correlation),
        **kl_estimate(truth_logs, logs),
        **block_metrics(target.blocks, blocks),
        **proposal_metrics(target, fitted.state, draws),
    }
    return record, fitted, draws


def partition_matrix(blocks, dimension):
    matrix = np.eye(dimension)
    for block in blocks:
        matrix[np.ix_(block, block)] = 1
    return matrix


def save_case_plots(target, evaluation, fits, draws, output):
    # Display whole true blocks together; the model was fitted on shuffled columns.
    display = [index for block in target.blocks for index in block]
    for page, start in enumerate(range(0, target.dimension, 6), 1):
        indices = display[start : start + 6]
        combined = np.concatenate([evaluation[:, indices], *[x[:, indices] for x in draws.values()]])
        limits = [tuple(np.quantile(combined[:, j], [0.005, 0.995])) for j in range(len(indices))]
        options = dict(
            bins=22,
            range=limits,
            smooth=0.8,
            smooth1d=0.8,
            levels=(0.68, 0.95),
            plot_datapoints=False,
            fill_contours=False,
            hist_kwargs={"density": True},
            quiet=True,
        )
        fig = corner.corner(
            evaluation[:, indices],
            labels=[target.labels[i] for i in indices],
            color="#202124",
            **options,
        )
        for order in sorted(draws):
            corner.corner(draws[order][:, indices], fig=fig, color=COLORS[order], **options)
        handles = [Line2D([0], [0], color="#202124", label="Independent truth draw")]
        handles.extend(Line2D([0], [0], color=COLORS[o], label=f"Morph order {o}") for o in draws)
        fig.legend(handles=handles, loc="upper right", fontsize=10)
        fig.suptitle(f"{target.name} | {target.dimension}D | page {page} | 1,000 draws", y=1.01)
        fig.savefig(output / f"corner_{target.name}_{target.dimension:02d}d_p{page}.png", dpi=105, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, len(fits) + 1, figsize=(15, 4), constrained_layout=True)
    partitions = [("True blocks", target.blocks)] + [
        (f"Morph order {order}", fit_result.state.blocks) for order, fit_result in fits.items()
    ]
    for ax, (title, blocks) in zip(axes, partitions):
        matrix = partition_matrix(blocks, target.dimension)[np.ix_(display, display)]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("True-block display order")
    fig.suptitle(f"{target.name} | {target.dimension}D | same-factor adjacency")
    fig.savefig(output / f"blocks_{target.name}_{target.dimension:02d}d.png", dpi=150)
    plt.close(fig)


def save_summaries(records, baselines, output):
    main = [row for row in records if row["configuration"] == "main"]
    metrics = (
        ("kl", "Forward KL (nats; raw MC estimate)", "forward_kl"),
        ("speed", "Warm fit time (seconds; CPU)", "fit_warm_median_seconds"),
        ("block_recall", "Retained true dependent pairs", "pair_recall"),
        ("block_precision", "Selected pairs within true blocks", "pair_precision"),
    )
    for filename, ylabel, metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for name, ax in zip(SCENARIOS, axes.flat):
            for order in [1, 2, 3]:
                rows = sorted(
                    (row for row in main if row["scenario"] == name and row["order"] == order),
                    key=lambda row: row["dimension"],
                )
                rows = [row for row in rows if row[metric] is not None]
                if not rows:
                    continue
                ax.errorbar(
                    [row["dimension"] for row in rows],
                    [row[metric] for row in rows],
                    yerr=[1.96 * row["forward_kl_se"] for row in rows] if metric == "forward_kl" else None,
                    color=COLORS[order],
                    marker="o",
                    capsize=3,
                    label=f"Order {order}",
                )
            if metric == "forward_kl":
                for baseline, color in [("oracle", "#009e73"), ("normal", "#cc79a7")]:
                    rows = sorted((r for r in baselines if r["scenario"] == name), key=lambda r: r["dimension"])
                    ax.plot(
                        [r["dimension"] for r in rows],
                        [r[f"{baseline}_forward_kl"] for r in rows],
                        ":",
                        color=color,
                        label="Known-block KDE" if baseline == "oracle" else "Product normals",
                    )
                ax.axhline(0, color="#444444", linewidth=0.6)
            if metric == "fit_warm_median_seconds":
                ax.set_yscale("log")
            if metric.startswith("pair_"):
                ax.set_ylim(-0.05, 1.05)
            ax.set_title(name)
            ax.set_xlabel("Dimensions")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No true dependent pairs", transform=ax.transAxes, ha="center")
        fig.savefig(output / f"{filename}_summary.png", dpi=140)
        plt.close(fig)


def metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        cpu = subprocess.check_output(["lscpu"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, cpu = os.environ.get("GITHUB_SHA", "unknown"), platform.processor()
    core_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("src/jax_morph").glob("*.py"))
    }
    return {
        "benchmark_commit": commit,
        "merged_implementation_commit": "8ec30bb19cc89ccbe7fdea05b3cceffacdf494a6",
        "reference_repository": "https://github.com/EL-MZ/Morph-approx-lab",
        "reference_commit": "7dded6163028ce6dbc1719fa783a3e3a67ea306b",
        "implementation_sha256": core_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "corner": corner.__version__,
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "sample_policy": "1,000 rows per training, independent evaluation, and proposal batch",
        "main_tc_samples": 256,
        "sgm_seed_budget": 64,
        "candidate_batch_size": 8,
        "timing_note": (
            "Synchronized wall times. First calls may include compilation or reuse previous "
            "shape compilations. Warm fit repeats include host validation, TC, SGM, and KDE "
            "fit; density and sample medians use three repeats, SGM five. CPU results only."
        ),
        "kl_note": "Raw forward KL; MC SE conditional on fitted model; independent evaluation",
        "corner_note": "Six-dimensional pages cover every coordinate; 0.5%-99.5% plot limits",
    }


def save_report(records, baselines, meta, output):
    payload = {"metadata": meta, "records": records, "baselines": baselines}
    (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    flat = [{k: v for k, v in row.items() if not isinstance(v, (list, tuple, dict))} for row in records]
    if flat:
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def write_readme(records, output):
    rows = [row for row in records if row["dimension"] == 30 and row["configuration"] == "main"]
    lines = [
        "# JAX Morph block benchmark results",
        "",
        "Six analytic target families at 6, 12, 20, and 30 dimensions; shuffled input columns.",
        "Each fit uses 1,000 sample vectors (1,000 values per parameter), followed by separate",
        "1,000-vector evaluation and proposal batches. Main TC scoring uses a reproducible",
        "256-point subset; final KDE factors use all 1,000 fitting vectors. The 30D ablations",
        "compare TC scoring on all 1,000 fitting vectors. SGM tries up to 64 seeds.",
        "",
        "Forward KL is estimated on independent truth samples, in nats, without clipping.",
        "The reported SE describes evaluation Monte Carlo error, not variation between fits.",
        "These are single-training-seed benchmarks. Pair recall and precision measure",
        "retained dependencies and spurious mergers. Orders 2 and 3 force same-size blocks:",
        "they cannot reproduce every mixed partition exactly. Independent targets have",
        "undefined pair recall. The known-block KDE baseline removes selection error while",
        "retaining KDE smoothing error. Proposals outside bounded support are retained:",
        "reverse KL is infinite when leakage is observed; no support floor is substituted.",
        "",
        "![Forward KL](kl_summary.png)",
        "![Warm CPU fit time](speed_summary.png)",
        "![Pair recall](block_recall_summary.png)",
        "![Pair precision](block_precision_summary.png)",
        "",
        "## 30-dimensional main results",
        "",
        "| Target | Order | KL ± MC SE (nats) | Pair recall | Pair precision | Warm fit (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        recall = "—" if row["pair_recall"] is None else f"{row['pair_recall']:.3f}"
        precision = "—" if row["pair_precision"] is None else f"{row['pair_precision']:.3f}"
        lines.append(
            f"| {row['scenario']} | {row['order']} | {row['forward_kl']:.3f} ± "
            f"{row['forward_kl_se']:.3f} | {recall} | {precision} | "
            f"{row['fit_warm_median_seconds']:.3f} |"
        )
    lines += [
        "",
        "## Inspect all parameters",
        "",
        "Corner plots overlay the independent truth draw and all three Morph orders.",
        "Pages contain at most six parameters and together cover every parameter.",
        "Coordinates are displayed in true-block order; fitting used shuffled columns.",
        "Quantile limits affect plotting only. Full-support samples enter KL and diagnostics.",
        "",
    ]
    for name in SCENARIOS:
        paths = sorted((output / "plots").glob(f"corner_{name}_30d_*.png"))
        links = " · ".join(f"[page {i}](plots/{path.name})" for i, path in enumerate(paths, 1))
        lines.append(f"- **{name}:** {links} · [blocks](plots/blocks_{name}_30d.png)")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m pip install -e '.[test,benchmark]'",
        "python -m benchmarks.block_suite --output benchmark-results/blocks-2026-10-05",
        "```",
        "",
        "`results.json` contains configurations, all partitions, hardware, dependency versions,",
        "implementation hashes, first and warm timings, support leakage, and importance ESS.",
        "`metrics.csv` contains scalar results; `targets.json` records exact target parameters.",
        "The workflow pins dependencies in `.github/workflows/block-benchmarks.yml`.",
        "First-call timings may reuse compiled shapes from earlier cases; they are not isolated",
        "compiler timings. Warm fits include scoring and selection; SGM-only times are separate.",
        "No GPU speed or speedup relative to the original MorphZ is inferred from this CPU run.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/blocks-2026-10-05"))
    parser.add_argument("--dimensions", nargs="+", type=int, default=[6, 12, 20, 30])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--fit-repeats", type=int, default=1)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    if any(d < 3 or d > 30 for d in args.dimensions) or args.fit_repeats < 1:
        parser.error("dimensions must be 3..30 and fit-repeats >= 1")
    output = args.output
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    records, baselines, descriptions = [], [], []
    for dimension in args.dimensions:
        for name in args.scenarios:
            seed = 20261005 + dimension * 100 + SCENARIOS.index(name) * 10
            target = make_target(name, dimension, seed)
            descriptions.append(target.description())
            training = target.sample(np.random.default_rng(seed + 1), 1000)
            evaluation = target.sample(np.random.default_rng(seed + 2), 1000)
            truth_logs = target.logpdf(evaluation)
            oracle = MorphState(
                target.blocks,
                tuple(fit_kde(training[:, block]) for block in target.blocks),
                dimension,
            )
            oracle_kl = kl_estimate(truth_logs, log_prob(oracle, evaluation))
            normal_logs = stats.norm.logpdf(
                evaluation, training.mean(axis=0), training.std(axis=0, ddof=1)
            ).sum(axis=1)
            normal_kl = kl_estimate(truth_logs, normal_logs)
            baselines.append(
                {
                    "scenario": name,
                    "dimension": dimension,
                    "oracle_forward_kl": oracle_kl["forward_kl"],
                    "oracle_forward_kl_se": oracle_kl["forward_kl_se"],
                    "normal_forward_kl": normal_kl["forward_kl"],
                    "normal_forward_kl_se": normal_kl["forward_kl_se"],
                }
            )
            fits, draws = {}, {}
            for order in [1, 2, 3]:
                row, fitted, proposal = evaluate_fit(
                    target, training, evaluation, truth_logs, order, 256, seed + 3, args.fit_repeats
                )
                row["configuration"] = "main"
                records.append(row)
                fits[order], draws[order] = fitted, proposal
                print(json.dumps(row, allow_nan=False), flush=True)
                save_report(records, baselines, meta, output)
            save_case_plots(target, evaluation, fits, draws, plots)
            if dimension == 30 and name != "independent_mixed" and not args.skip_ablations:
                orders = [2, 3] if name in ("strong_pairs", "triplets") else [2]
                for order in orders:
                    row, _, _ = evaluate_fit(
                        target, training, evaluation, truth_logs, order, 1000, seed + 3, args.fit_repeats
                    )
                    row["configuration"] = "full_tc"
                    records.append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                    save_report(records, baselines, meta, output)
    (output / "targets.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    save_summaries(records, baselines, output)
    write_readme(records, output)
    print(f"Completed {len(records)} fitted-model records; results in {output}", flush=True)


if __name__ == "__main__":
    main()
