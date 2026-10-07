"""Generate the report and static research figure from saved experiment JSON."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark-results/lazy-partition-search"
NAMES = {
    "sgm": "SGM (all seeds)",
    "pearson_screen": "Pearson screen",
    "mi_screen": "Nonlinear MI screen",
    "stochastic": "Random-start swaps",
    "hybrid": "MI screen + swaps",
}
SCENARIOS = {
    "gaussian_pairs": "Gaussian pairs",
    "banana_pairs": "Banana pairs",
    "gaussian_triplets": "Gaussian triplets",
    "xor_triplets": "Pure triple interaction",
    "dense_gaussian": "Dense Gaussian",
    "independent": "Independent Gaussian",
}


def summarize(rows):
    baseline = {(r["scenario"], r["seed"]): r for r in rows if r["method"] == "sgm"}
    result = {}
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        for method in NAMES:
            group = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            scores = np.array([r["score_fraction_of_sgm"] * 100 for r in group])
            result[scenario, method] = {
                "score": scores.mean(),
                "minimum": scores.min(),
                "maximum": scores.max(),
                "speed": np.median(
                    [
                        baseline[scenario, r["seed"]]["selection_seconds"] / r["selection_seconds"]
                        for r in group
                    ]
                ),
                "seconds": np.median([r["selection_seconds"] for r in group]),
                "kl": np.mean([r["heldout_kl"] for r in group]),
                "delta_kl": np.mean(
                    [r["heldout_kl"] - baseline[scenario, r["seed"]]["heldout_kl"] for r in group]
                ),
                "rmse": np.sqrt(np.mean([r["log_z_error"] ** 2 for r in group])),
                "kernel": np.median([r["kernel_blocks"] for r in group]),
            }
    return result


def main():
    raw = json.loads((DATA / "all-seeds.json").read_text())
    rows = raw["rows"]
    assert len(rows) == 150
    summary = summarize(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    colors = ["#b68525", "#087f8c", "#b95065", "#5a53a3"]
    methods = list(NAMES)[1:]
    positions = np.arange(len(SCENARIOS))
    for j, (method, color) in enumerate(zip(methods, colors)):
        x = positions + (j - 1.5) * 0.19
        values = [summary[s, method] for s in SCENARIOS]
        scores = np.array([v["score"] for v in values])
        axes[0].bar(x, scores, width=0.18, color=color, label=NAMES[method])
        axes[0].errorbar(
            x,
            scores,
            yerr=[scores - [v["minimum"] for v in values], [v["maximum"] for v in values] - scores],
            fmt="none",
            color="#303030",
            lw=0.7,
            capsize=2,
        )
        axes[1].bar(x, [v["speed"] for v in values], width=0.18, color=color)
        axes[2].bar(x, [v["delta_kl"] for v in values], width=0.18, color=color)
    titles = ["Retained estimated TC", "Block-selection speedup", "Change in held-out KL"]
    labels = [
        "Percent of SGM score (higher is better)",
        "SGM time / method time",
        "Nats relative to SGM (lower is better)",
    ]
    for ax, title, label in zip(axes, titles, labels):
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.set_ylabel(label)
        ax.set_xticks(
            positions,
            [s.replace(" ", "\n", 1) for s in SCENARIOS.values()],
            rotation=35,
            ha="right",
            fontsize=9,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes[0].axhline(100, color="#333333", ls="--", lw=0.8)
    axes[0].set_ylim(0, 112)
    axes[1].axhline(1, color="#333333", ls="--", lw=0.8)
    axes[2].axhline(0, color="#333333", lw=0.8)
    fig.suptitle(
        "Morph block search with 80% fewer unique KDE-TC evaluations\n"
        "24 dimensions; 5 seeds; full-candidate, all-seed SGM baseline",
        fontsize=14,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False
    )
    fig.savefig(ROOT / "docs/lazy-partition-search.png", dpi=180)
    plt.close(fig)

    text = """# Reducing TC evaluations when fitting Morph

Experiment date: 6 October 2026 (Pacific/Auckland).

**Recommendation:** use cheap nonlinear dependence screening to build a small candidate
pool, evaluate KDE-based TC only for this pool, and solve the restricted partition
problem. Add stochastic repartition moves when validation justifies a larger search.
These experiments favor screening over random-start stochastic search. This is an
isolated research prototype; the production `fit` API was not modified.

## Why the search can be lazy

The supplied paper, *Enhancing evidence estimation through informed probability
density approximation*, Section III, Eqs. 5-8, defines the objective

$$F(B)=\\sum_{b\\in B}C(b),\\qquad |B|=\\lfloor d/L\\rfloor,$$

with disjoint blocks of width L and remaining coordinates fitted as singletons.
Algorithm 1 computes all binomial(d,L) block scores before starting SGM. Reducing
the number of SGM seeds only reduces the later selection stage.

For a repartition of two or three existing factors, only changed factors contribute:

$$\\Delta F=\\sum_{b\\in B_{new}}\\widehat C(b)-\\sum_{b\\in B_{old}}\\widehat C(b).$$

Cache scores by sorted coordinate tuple. Each new two-block proposal needs at most
two new TC evaluations; a three-block proposal needs at most three. Unchanged factors
cancel. Every state remains a valid normalized Morph product. Include residual
singletons in moves when d is not divisible by L.

## Alternatives implemented and tested

1. **Nonlinear screening + restricted optimization.** Compute pairwise MI from
   six quantile bins, using counts instead of KDE. For each coordinate form blocks
   with its five strongest neighbors, rank these by summed pairwise MI, and retain
   a budgeted shortlist. Include a random feasible partition and reserve about 20%
   of the pool for random blocks. Compute actual KDE TC only on this pool. A small
   binary set-packing MILP selects exactly floor(d/L) disjoint blocks. Its optimum
   is only an optimum within the pool; it gives no guarantee over omitted blocks.
2. **Stochastic partition search.** Start from a random partition. In each round,
   propose eight random repartitions of two factors (75%) or three factors (25%).
   Score unseen blocks in batches, select the best proposed move, occasionally
   accept a worsening move under a cooling temperature, and retain the best state
   seen. Reset to that best state every 30 rounds. This is an optimization heuristic,
   not an MCMC sampler or a convergence-certified annealing method.
3. **Hybrid.** Spend 60% of the TC budget on nonlinear screening and restricted
   optimization, then use the remainder for stochastic refinement. It cannot lower
   the score of its own starting partition, but its smaller initial pool means it
   need not beat the screening-only method under the same total budget.
4. **Pearson screening control.** Use squared Pearson correlation in place of
   binned MI. This tests whether simple linear screening is sufficient.

For L=2, replace the restricted MILP with maximum-weight, maximum-cardinality
general-graph matching, e.g. [Blossom via NetworkX](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html).
Retaining a feasible scaffold guarantees that the sparse graph admits the required
matching cardinality. Matching alone does not reduce the cost of obtaining edge
weights. For L>=3, the prototype uses [SciPy MILP / HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
with a five-second limit and checked status. All 90 restricted solves in the main
experiment reached an optimum for their candidate pools.

**Cost:** direct plug-in KDE scoring costs roughly O(n_TC^2) per block at fixed L.
Exhaustive evaluation costs O(binomial(d,L) n_TC^2), plus marginal scoring and
selection. A fixed B-block search replaces the combinatorial factor by B.
Screening still costs O(n_TC d^2) for cheap binned pair statistics, and generates
at most O(d binomial(k,L-1)) neighbor candidates with fixed k. At L=2 this still
examines every pair cheaply: the saving is avoiding *KDE* TC on all pairs.
For genuine subquadratic dependence discovery at very large d, random neighborhoods
or model-informed neighborhoods would be needed; they were not tested here.

## Test design

- 24 dimensions; six scenarios; five independent dataset/search seeds: 150 fitted
  densities across five methods. Baseline evaluates all blocks and tries **every
  SGM seed**, using the repository's JAX implementation with early stopping.
- 256 scoring samples, 512 final KDE fitting samples, 1,024 independent posterior
  evaluation samples, and 1,024 proposal samples for each bridge estimate. Scoring
  samples are a subset of fit samples; evaluation samples are independent.
- Same clipped in-sample KDE plug-in TC, Gaussian KDE convention, bandwidth and
  final fitting samples across methods. Every method has its own TC cache and cannot
  access baseline scores. All selected partitions are checked for validity.
- Budgets: 55 of 276 possible pairs; 404 of 2,024 possible triplets. Counts refer
  to unique block scores; padded 16-block kernel batches perform extra work,
  recorded separately in the raw results. Marginal entropies are computed once per
  method and included in elapsed time. Baseline-only candidate enumeration is
  outside the timer, which conservatively favors SGM.
- Timings include screening, score evaluation, Python orchestration and optimization;
  they exclude JIT warmup, final KDE fitting, quality checks and bridge sampling.
  They are **block-selection timings, not whole-MorphZ speedups**. Shared-machine
  timings are indicative. Medians below use paired seed-wise timing ratios.
- Exact normalized synthetic targets have log Z = 0. Reported evidence RMSE is
  sqrt(mean(log Z estimate squared)) across five runs. Held-out KL is Monte Carlo
  mean(log p - log fitted Morph), in nats; lower is better. Neither metric is used
  to tune these runs. All 150 bridge iterations converged; this alone does not
  establish evidence accuracy or good overlap.
- Gaussian pairs/triplets have within-block correlation 0.8. Banana pairs use
  y=0.8(x^2-1)+0.25 epsilon. Pure triple interactions use noisy sign triples
  (A,B,A*B), with independent fair signs A,B and independent Gaussian noise sd=0.3.
  Each pair is independent but the joint triple is dependent. Coordinates are
  permuted. Dense Gaussian uses four shared latent loadings plus diagonal noise.

![Results](lazy-partition-search.png)

TC bars show means and minimum-to-maximum ranges over five seeds. Values above
100% can occur because even all-seed SGM is a heuristic, not a global optimum.
On the independent target, population TC is zero: the positive estimated scores
reflect finite-sample KDE effects, so a score ratio there is not dependence recovery.

## Results

| Target | Method | Retained TC %: mean [min, max] | Selection speedup | Held-out KL | log Z RMSE |
|---|---|---:|---:|---:|---:|
"""
    for scenario in SCENARIOS:
        for method in NAMES:
            s = summary[scenario, method]
            text += (
                f"| {SCENARIOS[scenario]} | {NAMES[method]} | {s['score']:.2f} "
                f"[{s['minimum']:.2f}, {s['maximum']:.2f}] | {s['speed']:.2f}x | "
                f"{s['kl']:.3f} | {s['rmse']:.3f} |\n"
            )
    text += """
**Interpretation.** Nonlinear screening and the hybrid recovered the same partitions
as SGM in all 15 Gaussian-pair, banana-pair and Gaussian-triplet runs. Random-start
search spent too much of its small budget finding an initial structure. Pearson
screening missed the non-monotone banana dependencies. The hybrid gave no systematic
advantage over nonlinear screening here and usually had more overhead.

The dense Gaussian example retained about 99.85% of the baseline score with nonlinear
screening, but evidence RMSE was 0.195 versus 0.108 for SGM. Thus near-equality of
estimated TC is not sufficient evidence of equivalent downstream accuracy. Five
runs do not establish whether this difference persists. The banana baseline itself
had KL about 7.35 and evidence RMSE 0.239, illustrating the remaining KDE/overlap
problem even when block selection is correct.

The triple-interaction stress test is an explicit failure: nonlinear screening
retained only 20.39% of SGM's estimated TC; the hybrid retained 25.16%. Pairwise
independence leaves no population pairwise signal to screen on. Random exploration
helped only occasionally. None of these restricted searches guarantees detection
of pure higher-order dependence. In a black-box block-score problem, an unqueried
block can have arbitrarily large score, so no universal exact-optimum guarantee is
possible without structural assumptions or valid bounds.

## Fixed-budget dimension check

The additional 48-dimensional runs retain a budget of **404 triplets**, while the
candidate universe grows to **17,296**: 97.66% fewer unique KDE-TC evaluations.
Only two seeds per target were used, so this is a scaling spot check.

| Target | Method | TC %: mean [min, max] | Seconds | Speedup | KL | log Z RMSE |
|---|---|---:|---:|---:|---:|---:|
"""
    scaling = json.loads((DATA / "fixed-budget-48d.json").read_text())
    assert len(scaling["rows"]) == 20
    for (scenario, method), s in summarize(scaling["rows"]).items():
        text += (
            f"| {SCENARIOS[scenario]} | {NAMES[method]} | {s['score']:.2f} "
            f"[{s['minimum']:.2f}, {s['maximum']:.2f}] | {s['seconds']:.3f} | "
            f"{s['speed']:.2f}x | {s['kl']:.3f} | {s['rmse']:.3f} |\n"
        )
    text += (
        """
The large timing ratios above compare against *all-seed* SGM, including its
selection overhead. They should not be transferred directly to an implementation
using only a small number of seeds. The unique-TC counts do not depend on that
choice. In the 48d dense case, both SGM and nonlinear screening have held-out KL
around 19 nats and log Z RMSE around 1: neither provides a good density approximation
at this sample size and block order. Fast selection is not a remedy for this.

## Practical integration and further alternatives

Add a separate candidate-pool/partition-search API rather than passing an incomplete
pool to the existing SGM function, which correctly requires every combination.
Make `tc_budget`, exploration fraction, neighbor count, random key and stopping
criteria explicit. Return queried blocks/scores, selected blocks, budget use and
solver status. Then fit the ordinary Morph factors on all fitting samples.
Cache marginal entropies once across all search rounds. The current public TC
function recomputes them on each call, so calling it once per proposed move would
discard a useful saving; the prototype uses a separate cached oracle.

For the next practical version, start with nonlinear screening, preserve random
exploration and a feasible scaffold, then spend extra budget only if an independent
validation split shows an improvement in mean log Morph density or bridge overlap.
For a credible pure higher-order interaction, increase random higher-order probing,
use known model couplings to propose blocks, or revert to exhaustive scoring on a
smaller relevant coordinate group. Pairwise screening cannot certify absence.

Two further approaches are promising but **not tested** in this experiment:

- **Multiple scoring fidelities:** score proposed blocks on a small pilot sample,
  promote uncertain/promising blocks to a larger TC sample, and re-score finalists
  at one common fidelity. Combine with a fixed candidate budget. Repeated adaptive
  comparisons require care about selection bias; naive confidence bounds around
  biased in-sample KDE TC are not rigorous pruning certificates.
- **Hierarchical merging / beam growth:** grow groups along a sparse dependence
  graph using conditional MI, exploiting C(A union {j})=C(A)+I(X_j;X_A).
  Keep several partial candidates instead of committing to one merge, and repair
  block sizes at the end. Conditional MI captures effects after a useful group is
  found; it does not eliminate the initial pure-synergy discovery problem. A fixed
  beam and fixed neighbor count bound candidate growth, at the cost of pruning risk.

For KDE-based Morph, the paper's exact KL identity applies to exact target marginals.
With estimated KDE factors, also check held-out log density. Maximizing its blockwise
sum directly is a possible fitting objective; it is not identical to maximizing
clipped in-sample TC. Mixed block sizes or fitting residual variables jointly would
change the paper's fixed-order approximation and were not used here.

## Reproduction and limitations

From the repository root, with its scientific Python dependencies available:

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:.
python -m benchmarks.lazy_partition_search \\
  --output benchmark-results/lazy-partition-search/all-seeds.json
python -m benchmarks.lazy_partition_search --dimension 48 --tc-budget 404 --seeds 2 \\
  --scenarios gaussian_triplets dense_gaussian \\
  --output benchmark-results/lazy-partition-search/fixed-budget-48d.json
python -m benchmarks.lazy_partition_report
pytest -q tests/test_lazy_partition_search.py
```

The actual run used existing WSL Conda Python on CPU and an isolated temporary
bytecode cache (`PYTHONPYCACHEPREFIX=/tmp/morph-lazy-pycache`) to avoid stale OneDrive
cache files. Focused checks passed: TC equality to the production estimator, cache
and budget enforcement, restricted optimum versus exhaustive enumeration, feasible
candidate scaffold, and residual-singleton movement (four tests). These are synthetic
IID, unweighted tests, not posterior-chain or GPU benchmarks. The prototype uses
the current local source, including pre-existing changes, and existing environment
versions rather than installing the project's pinned environment. No guarantee for
real posterior data, unseen structures, bandwidths, larger L or evidence precision
is established by this pilot.

Raw observations and configuration:
[main JSON](../benchmark-results/lazy-partition-search/all-seeds.json)
and [48d JSON](../benchmark-results/lazy-partition-search/fixed-budget-48d.json).
Code: [experiment](../benchmarks/lazy_partition_search.py),
[report generator](../benchmarks/lazy_partition_report.py),
[checks](../tests/test_lazy_partition_search.py).

Environment: """
        + json.dumps(raw["environment"], sort_keys=True)
        + ".\n"
    )
    (ROOT / "docs/lazy-partition-search.md").write_text(text, encoding="utf-8")
    print("Wrote docs/lazy-partition-search.md and docs/lazy-partition-search.png")


if __name__ == "__main__":
    main()
