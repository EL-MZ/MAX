"""Compare subsampled raw TC estimates with a fixed full-sample reference.

Run: python -m benchmarks.tc_accuracy --output benchmark-results/tc-accuracy
The bands show descriptive standard deviations, not confidence intervals.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.distributions import SCENARIOS, make_target  # noqa: E402
from jax_morph import candidate_blocks, estimate_total_correlations  # noqa: E402


def difference_summary(estimates, reference):
    """Pool candidate-by-trial differences; also report variation of trial means.

    A candidate is a fixed block of columns, matched by index to the reference.
    Candidates and nested sample sizes are correlated; no independence assumption
    is made and these descriptive SDs must not be interpreted as standard errors.
    """
    estimates, reference = np.asarray(estimates), np.asarray(reference)
    if estimates.ndim != 2 or reference.shape != (estimates.shape[1],):
        raise ValueError("expected (trials, candidates) estimates and (candidates,) reference")
    if min(estimates.shape) < 1 or estimates.size < 2 or estimates.shape[0] < 2:
        raise ValueError("at least two trials are required")
    if not np.isfinite(estimates).all() or not np.isfinite(reference).all():
        raise ValueError("TC scores must be finite")
    errors = estimates - reference[None, :]
    absolute = np.abs(errors)
    return {
        "mean_difference_nats": float(errors.mean()),
        "std_difference_nats": float(errors.std(ddof=1)),
        "mean_absolute_difference_nats": float(absolute.mean()),
        "std_absolute_difference_nats": float(absolute.std(ddof=1)),
        "rmse_nats": float(np.sqrt(np.mean(errors**2))),
        "std_trial_mean_difference_nats": float(errors.mean(axis=1).std(ddof=1)),
        "std_trial_mean_absolute_difference_nats": float(absolute.mean(axis=1).std(ddof=1)),
        "reference_mean_tc_nats": float(reference.mean()),
        "reference_std_tc_nats": float(reference.std(ddof=1)) if reference.size > 1 else 0.0,
        "trials": estimates.shape[0],
        "candidates": estimates.shape[1],
        "pooled_differences": estimates.size,
    }


def nested_subsets(pool_size, sizes, repeats, seed):
    """Independent permutations per trial, nested prefixes across sample sizes."""
    if repeats < 2 or not sizes or min(sizes) < 2 or max(sizes) >= pool_size:
        raise ValueError("require >=2 trials and subset sizes in [2, pool_size)")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(pool_size)[: max(sizes)] for _ in range(repeats)])


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_differences(records, cases, sizes, output, absolute=False):
    names = list(dict.fromkeys(case["scenario"] for case in cases))
    columns = min(3, len(names))
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.1 * rows), squeeze=False)
    field = "absolute_difference" if absolute else "difference"
    colors = {2: "#0072b2", 3: "#d55e00", 4: "#009e73"}
    for ax, name in zip(axes.flat, names):
        for order in sorted({case["order"] for case in cases if case["scenario"] == name}):
            subset = sorted(
                [row for row in records if row["scenario"] == name and row["order"] == order],
                key=lambda row: row["samples"],
            )
            x = np.array([row["samples"] for row in subset])
            mean = np.array([row[f"mean_{field}_nats"] for row in subset])
            sd = np.array([row[f"std_{field}_nats"] for row in subset])
            color = colors.get(order, None)
            label = {2: "Pairs (order 2)", 3: "Triplets (order 3)"}.get(order, f"Order {order}")
            line = ax.plot(x, mean, "o-", color=color, label=label)[0]
            ax.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=0.16)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(sizes, labels=[str(size) for size in sizes])
        ax.minorticks_off()
        ax.set_title(name.replace("_", " ").capitalize())
        ax.set_xlabel("Samples used to fit and evaluate TC")
        ax.set_ylabel(
            "Mean |TC(n) - TC(1000)| (nats)" if absolute else "Mean TC(n) - TC(1000) (nats)"
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    dimension = cases[0]["dimension"]
    repeats = records[0]["trials"]
    title = (
        "Absolute TC differences" if absolute else "TC differences from the 1,000-sample reference"
    )
    heading = fig.suptitle(
        f"{title}\n{dimension} dimensions; {repeats} random subsets per sample count", y=0.99
    )
    heading.set_in_layout(False)
    fig.text(
        0.5,
        0.012,
        "Line: mean over all candidate blocks and trials. Band: mean +/- 1 sample SD "
        "over those differences (not a confidence interval).\n"
        "Each subset is drawn without replacement from the fixed 1,000-row reference dataset.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.91))
    stem = "tc_absolute_difference" if absolute else "tc_difference"
    fig.savefig(output / f"{stem}.png", dpi=180)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def write_report(args, cases, records):
    lines = [
        "# TC sample-count experiment",
        "",
        "The requested plot is `tc_difference.png`: mean signed TC(n) - TC(1000), "
        "with +/- one sample standard deviation. `tc_absolute_difference.png` shows "
        "the same comparison using absolute differences to avoid cancellation.",
        "",
        f"- Sample counts: {', '.join(map(str, args.samples))}.",
        "- Reference: all 1,000 rows of one fixed dataset per scenario.",
        f"- Trials: {args.repeats} independently randomized subsets per sample count.",
        f"- Dimensions: {cases[0]['dimension']}; block orders: {args.orders}.",
        "- All combinations of columns are evaluated, without candidate pruning.",
        "- Scores: raw, unclipped KDE TC estimates in nats, using the public TC API, "
        "Silverman bandwidth, no ridge, and float64.",
        "- Both fitting and evaluation use the subset, matching production TC thinning.",
        "- Each trial uses a new permutation without replacement. Sample counts are nested "
        "within a trial; orders share the same subsets.",
        "",
        "For each candidate block, subtract that same block's 1,000-row reference TC. "
        "The plotted mean and SD pool all candidate-by-trial differences. The SD is "
        "descriptive spread, not a standard error or confidence interval. Candidate "
        "blocks overlap, so they are not independent statistical replicates. The CSV "
        "also reports the SD across trial-averaged errors separately.",
        "",
        "The 1,000-row estimate is the requested reference, not exact population TC. "
        "Subsets overlap the reference, so this measures the error from thinning this "
        "dataset, not error on an independently generated dataset. No significance "
        "test, selection run, or minimum-sample threshold is imposed.",
        "",
        "## Mean absolute difference +/- SD, in nats",
        "",
        "| Distribution | Order | " + " | ".join(map(str, args.samples)) + " |",
        "|---|---:|" + "---:|" * len(args.samples),
    ]
    for case in cases:
        subset = sorted(
            [
                row
                for row in records
                if row["scenario"] == case["scenario"] and row["order"] == case["order"]
            ],
            key=lambda row: row["samples"],
        )
        cells = [
            f"{row['mean_absolute_difference_nats']:.4f} +/- "
            f"{row['std_absolute_difference_nats']:.4f}"
            for row in subset
        ]
        lines.append(f"| {case['scenario']} | {case['order']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m benchmarks.tc_accuracy --output benchmark-results/tc-accuracy "
            f"--dimensions {cases[0]['dimension']} --orders "
            + " ".join(map(str, args.orders))
            + f" --repeats {args.repeats} --seed {args.seed} --samples "
            + " ".join(map(str, args.samples)),
            "```",
            "",
            "`results.json` records configuration, device, datasets, and timing; "
            "`summary.csv` contains every plotted statistic; case NPZ files contain "
            "the reference and all subset scores, candidate identities, and subset indices. "
            "The input dataset for each scenario is saved as NPY.",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/tc-accuracy"))
    parser.add_argument("--samples", nargs="+", type=int, default=[10, 20, 30, 50, 100, 200, 500])
    parser.add_argument("--dimensions", type=int, default=12)
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    args = parser.parse_args()
    args.samples = sorted(set(args.samples))
    args.orders = sorted(set(args.orders))
    if args.repeats < 2 or not 2 <= min(args.samples) <= max(args.samples) < 1000:
        parser.error("require repeats >= 2 and sample counts in [2, 1000)")
    if min(args.orders) < 2 or max(args.orders) > args.dimensions:
        parser.error("require 2 <= orders <= dimensions")
    if max(args.orders) >= min(args.samples) or args.candidate_batch_size < 1:
        parser.error("require every sample count > every order, and positive batch size")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty to preserve earlier experiments")
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    records, cases = [], []
    config = {
        **vars(args),
        "output": str(args.output),
        "reference_samples": 1000,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "score_type": "raw_unclipped",
        "std_ddof": 1,
        "std_pooling": "candidate_by_trial_differences",
    }
    for scenario_index, scenario in enumerate(args.scenarios):
        dataset_seed = args.seed + scenario_index * 1000
        target = make_target(scenario, args.dimensions, seed=dataset_seed)
        samples = target.sample(np.random.default_rng(dataset_seed + 1), 1000)
        np.save(args.output / f"samples_{scenario}.npy", samples)
        indices = nested_subsets(1000, args.samples, args.repeats, dataset_seed + 2)
        for order in args.orders:
            blocks = np.asarray(candidate_blocks(args.dimensions, order))
            started = perf_counter()
            print(
                f"REFERENCE {scenario} order={order}: {len(blocks)} candidates, 1000 rows",
                flush=True,
            )
            reference = np.asarray(
                estimate_total_correlations(
                    samples,
                    blocks,
                    candidate_batch_size=args.candidate_batch_size,
                ).raw_scores
            )
            reference_seconds = perf_counter() - started
            scores = np.empty((len(args.samples), args.repeats, len(blocks)))
            seconds = []
            for size_index, size in enumerate(args.samples):
                started = perf_counter()
                for trial, subset in enumerate(indices):
                    scores[size_index, trial] = np.asarray(
                        estimate_total_correlations(
                            samples[subset[:size]],
                            blocks,
                            candidate_batch_size=args.candidate_batch_size,
                        ).raw_scores
                    )
                    if (trial + 1) % 25 == 0:
                        print(
                            f"  {scenario} order={order} n={size}: "
                            f"{trial + 1}/{args.repeats} trials",
                            flush=True,
                        )
                elapsed = perf_counter() - started
                seconds.append(elapsed)
                summary = difference_summary(scores[size_index], reference)
                records.append(
                    {
                        "scenario": scenario,
                        "dimension": args.dimensions,
                        "order": order,
                        "samples": size,
                        **summary,
                    }
                )
                print(
                    f"  DONE n={size}: delta={summary['mean_difference_nats']:.4f} "
                    f"+/- {summary['std_difference_nats']:.4f} nats, {elapsed:.1f}s",
                    flush=True,
                )
            stem = f"{scenario}_order{order}"
            np.savez_compressed(
                args.output / f"scores_{stem}.npz",
                sample_counts=np.array(args.samples),
                candidates=blocks,
                reference=reference,
                subset_scores=scores,
                subset_indices=indices,
            )
            cases.append(
                {
                    "scenario": scenario,
                    "dimension": args.dimensions,
                    "order": order,
                    "dataset_seed": dataset_seed,
                    "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                    "target": target.description(),
                    "candidates": len(blocks),
                    "reference_seconds": reference_seconds,
                    "subset_batch_seconds": seconds,
                    "scores_file": f"scores_{stem}.npz",
                }
            )
            write_csv(args.output / "summary.csv", records)
            (args.output / "results.json").write_text(
                json.dumps({"config": config, "cases": cases}, indent=2),
                encoding="utf-8",
            )
    plot_differences(records, cases, args.samples, args.output)
    plot_differences(records, cases, args.samples, args.output, absolute=True)
    write_report(args, cases, records)
    print(f"Saved plots and results in {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
