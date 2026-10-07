"""Small, reproducible experiments with on-demand TC for Morph partitions.

Research prototype; does not change the public fitting API. Run with the repo
on PYTHONPATH, JAX_PLATFORMS=cpu and JAX_ENABLE_X64=true. Requires SciPy.
"""

import argparse
import json
import math
import platform
from itertools import combinations
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp

from jax_morph.bridge import bridge_sampling
from jax_morph.correlation import estimate_total_correlations
from jax_morph.kde import _fit_kde, kde_log_prob
from jax_morph.morph import _fit_factors, log_prob, sample
from jax_morph.sgm import _select_validated


def canonical(blocks):
    return tuple(sorted(tuple(sorted(map(int, b))) for b in blocks))


def validate_partition(blocks, d, order):
    flat = [i for b in blocks for i in b]
    assert len(blocks) == d // order
    assert all(len(b) == order for b in blocks)
    assert len(set(flat)) == len(flat) and all(0 <= i < d for i in flat)


@jax.jit
def marginal_entropies(x):
    w = jnp.full((len(x),), 1 / len(x))

    def entropy(i):
        y = x[:, i, None]
        kde = _fit_kde(y, w, 1.0, 0.0)
        return -jnp.mean(kde_log_prob(kde, y, chunk_size=128))

    return jax.lax.map(entropy, jnp.arange(x.shape[1]))


@jax.jit
def tc_batch(x, entropy, blocks):
    w = jnp.full((len(x),), 1 / len(x))

    def score(b):
        y = x[:, b]
        kde = _fit_kde(y, w, 1.0, 0.0)
        return jnp.maximum(jnp.mean(kde_log_prob(kde, y, chunk_size=128)) + entropy[b].sum(), 0)

    return jax.lax.map(score, blocks, batch_size=16)


class TCOracle:
    """Independent cache per method; never consults baseline scores."""

    def __init__(self, x, budget):
        self.x = jnp.asarray(x)
        self.entropy = marginal_entropies(self.x)
        self.entropy.block_until_ready()
        self.cache = {}
        self.budget = budget
        self.kernel_blocks = 0

    def evaluate(self, blocks):
        blocks = [tuple(sorted(map(int, b))) for b in blocks]
        new = list(dict.fromkeys(b for b in blocks if b not in self.cache))
        if len(self.cache) + len(new) > self.budget:
            raise ValueError("TC budget exceeded")
        for start in range(0, len(new), 16):
            batch = new[start : start + 16]
            padded = batch + [batch[-1]] * (16 - len(batch))
            values = np.asarray(tc_batch(self.x, self.entropy, jnp.asarray(padded)))
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite TC")
            self.kernel_blocks += 16
            self.cache.update(zip(batch, map(float, values[: len(batch)])))
        return np.asarray([self.cache[b] for b in blocks])


def pair_screen(x, method="histogram", bins=6):
    """Cheap O(n d^2) screening, not pairwise KDE TC."""
    if method == "pearson":
        out = np.corrcoef(x.T) ** 2
    else:
        # Quantile bins allow non-monotone dependence (e.g. x versus x^2).
        ranks = np.argsort(np.argsort(x, axis=0), axis=0)
        codes = np.minimum(ranks * bins // len(x), bins - 1)
        out = np.zeros((x.shape[1], x.shape[1]))
        for i, j in combinations(range(x.shape[1]), 2):
            p = np.bincount(codes[:, i] * bins + codes[:, j], minlength=bins**2)
            p = p.reshape(bins, bins) / len(x)
            product = p.sum(0)[None, :] * p.sum(1)[:, None]
            nz = p > 0
            out[i, j] = out[j, i] = np.sum(p[nz] * np.log(p[nz] / product[nz]))
    np.fill_diagonal(out, -np.inf)
    return out


def random_partition(rng, d, order):
    return canonical(rng.permutation(d)[: d // order * order].reshape(-1, order))


def screened_pool(matrix, order, count, rng, exploration=0.2):
    """Scaffold ensures feasibility; local neighbors avoid full enumeration."""
    d = len(matrix)
    scaffold = random_partition(rng, d, order)
    pool = set(scaffold)
    proposed = set()
    k = min(d - 1, max(order, 5))
    for i in range(d):
        neighbors = np.argsort(-matrix[i], kind="stable")[:k]
        for others in combinations(neighbors, order - 1):
            proposed.add(tuple(sorted((i, *map(int, others)))))
    ranked = sorted(proposed, key=lambda b: (-sum(matrix[i, j] for i, j in combinations(b, 2)), b))
    exploit_count = max(len(pool), int(count * (1 - exploration)))
    for block in ranked:
        if len(pool) >= exploit_count:
            break
        pool.add(block)
    while len(pool) < count:
        pool.add(tuple(sorted(map(int, rng.choice(d, order, replace=False)))))
    return sorted(pool), scaffold


def restricted_optimum(oracle, blocks, d, order, fallback):
    """Small candidate-pool MILP; explicitly checks solver feasibility/status."""
    blocks = list(blocks)
    scores = oracle.evaluate(blocks)
    a = np.zeros((d + 1, len(blocks)))
    for j, block in enumerate(blocks):
        a[list(block), j] = 1
    a[-1] = 1
    lower = np.r_[np.zeros(d), d // order]
    upper = np.r_[np.ones(d), d // order]
    result = milp(
        -scores,
        integrality=np.ones(len(blocks)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, lower, upper),
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )
    chosen = canonical(fallback)
    if result.x is not None:
        candidate = canonical([b for b, value in zip(blocks, result.x) if value > 0.5])
        validate_partition(candidate, d, order)
        if oracle.evaluate(candidate).sum() >= oracle.evaluate(chosen).sum():
            chosen = candidate
    return chosen, {"milp_status": int(result.status), "pool_optimal": bool(result.success)}


def stochastic_swaps(oracle, initial, d, order, rng, max_rounds=300):
    """Batched random 2/3-block repartitions, occasional downhill moves.

    Keep the best state; optimize, do not claim to sample a stationary law.
    Residual singletons join proposals so odd d does not freeze unused variables.
    """
    current = canonical(initial)
    best = current
    best_score = float(oracle.evaluate(best).sum())
    accepted = 0
    for round_index in range(max_rounds):
        used = {i for b in current for i in b}
        tails = tuple((i,) for i in range(d) if i not in used)
        factors = list(current + tails)
        moves, unseen = [], set()
        for _ in range(8):
            q = min(len(factors), 3 if rng.random() < 0.25 else 2)
            ids = rng.choice(len(factors), q, replace=False)
            selected = [factors[i] for i in ids]
            full_count = sum(len(b) == order for b in selected)
            values = rng.permutation([v for b in selected for v in b])
            new = canonical(values[: full_count * order].reshape(-1, order))
            retained = [b for i, b in enumerate(factors) if i not in ids and len(b) == order]
            candidate = canonical(retained + list(new))
            missing = {b for b in new if b not in oracle.cache}
            if len(oracle.cache) + len(unseen | missing) > oracle.budget:
                continue
            unseen |= missing
            moves.append(candidate)
        if not moves:
            break
        oracle.evaluate(sorted(unseen))
        scores = np.asarray([oracle.evaluate(b).sum() for b in moves])
        win = int(np.argmax(scores))
        current_score = float(oracle.evaluate(current).sum())
        temperature = 0.03 * max(best_score / max(1, len(best)), 0.01)
        temperature *= max(0.05, 1 - len(oracle.cache) / oracle.budget)
        if scores[win] >= current_score or rng.random() < np.exp(
            min(0, (scores[win] - current_score) / temperature)
        ):
            current = moves[win]
            accepted += 1
        if scores[win] > best_score:
            best, best_score = moves[win], float(scores[win])
        if (round_index + 1) % 30 == 0:
            current = best
        if len(oracle.cache) == oracle.budget:
            break
    validate_partition(best, d, order)
    return best, {"accepted_moves": accepted, "rounds": round_index + 1}


class SyntheticTarget:
    def __init__(self, scenario, d, seed):
        self.scenario, self.d = scenario, d
        self.order = 2 if scenario.endswith("pairs") else 3
        self.permutation = np.random.default_rng(seed + 901).permutation(d)
        inverse = np.argsort(self.permutation)
        self.truth = canonical(inverse[i : i + self.order] for i in range(0, d, self.order))
        self.cov = None
        if scenario == "dense_gaussian":
            rng = np.random.default_rng(seed + 999)
            loadings = rng.normal(size=(d, 4))
            self.cov = loadings @ loadings.T + np.eye(d) * 0.8
            scale = np.sqrt(np.diag(self.cov))
            self.cov /= scale[:, None] * scale[None, :]
        elif scenario.startswith("gaussian"):
            q = self.order
            self.cov = np.kron(np.eye(d // q), 0.2 * np.eye(q) + 0.8 * np.ones((q, q)))
        elif scenario == "independent":
            self.cov = np.eye(d)
        if self.cov is not None:
            self.chol = np.linalg.cholesky(self.cov)
            self.constant = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(self.cov)[1])

    def draw(self, rng, n):
        if self.cov is not None:
            x = rng.normal(size=(n, self.d)) @ self.chol.T
        elif self.scenario == "banana_pairs":
            x = rng.normal(size=(n, self.d))
            x[:, 1::2] = 0.8 * (x[:, ::2] ** 2 - 1) + 0.25 * x[:, 1::2]
        else:
            signs = rng.choice([-1.0, 1.0], size=(n, self.d // 3, 2))
            means = np.concatenate((signs, signs.prod(axis=2, keepdims=True)), axis=2)
            x = means.reshape(n, self.d) + 0.3 * rng.normal(size=(n, self.d))
        return x[:, self.permutation]

    def logpdf(self, x):
        x = np.asarray(x)[:, np.argsort(self.permutation)]
        if self.cov is not None:
            z = np.linalg.solve(self.chol, x.T).T
            return self.constant - 0.5 * np.sum(z**2, axis=1)
        if self.scenario == "banana_pairs":
            z = x.copy()
            z[:, 1::2] = (x[:, 1::2] - 0.8 * (x[:, ::2] ** 2 - 1)) / 0.25
            return -0.5 * (self.d * np.log(2 * np.pi) + np.sum(z**2, axis=1)) - (
                self.d // 2 * np.log(0.25)
            )
        means = np.asarray([[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]])
        z = (x.reshape(-1, self.d // 3, 1, 3) - means) / 0.3
        logs = -0.5 * np.sum(z**2, axis=-1) - 3 * np.log(0.3 * np.sqrt(2 * np.pi))
        return np.sum(logsumexp(logs, axis=-1) - np.log(4), axis=1)


def quality(blocks, target, fit_x, test_x, seed):
    used = {i for b in blocks for i in b}
    blocks = tuple(blocks) + tuple((i,) for i in range(target.d) if i not in used)
    state = _fit_factors(fit_x, np.full(len(fit_x), 1 / len(fit_x)), blocks, 1.0, 0.0)
    log_m = np.asarray(log_prob(state, test_x))
    proposals = np.asarray(sample(jax.random.PRNGKey(seed + 701), state, len(test_x)))
    result = bridge_sampling(
        target.logpdf(test_x),
        log_m,
        target.logpdf(proposals),
        log_prob(state, proposals),
        posterior_iat=1.0,
        max_iterations=1000,
        rtol=1e-9,
        average_iterations=1,
    )
    return {
        "heldout_kl": float(np.mean(target.logpdf(test_x) - log_m)),
        "heldout_log_density": float(np.mean(log_m)),
        "log_z_error": float(result.log_evidence),  # normalized target: log Z = 0
        "bridge_re": float(result.relative_error),
        "bridge_converged": bool(result.converged),
        "block_recall": (
            len(set(canonical(blocks)) & set(target.truth)) / len(target.truth)
            if target.scenario not in ("independent", "dense_gaussian")
            else None
        ),
    }


METHODS = ("sgm", "pearson_screen", "mi_screen", "stochastic", "hybrid")
SCENARIOS = (
    "gaussian_pairs",
    "banana_pairs",
    "gaussian_triplets",
    "xor_triplets",
    "dense_gaussian",
    "independent",
)


def run(args):
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenarios:
        for seed in range(args.seeds):
            target = SyntheticTarget(scenario, args.dimension, seed)
            d, order = target.d, target.order
            rng = np.random.default_rng(1000 + seed)
            fit_x = target.draw(rng, args.fit_samples)
            tc_x = fit_x[: args.tc_samples]
            test_x = target.draw(rng, args.test_samples)
            total = math.comb(d, order)
            requested = (
                args.tc_budget if args.tc_budget is not None else int(total * args.budget_fraction)
            )
            budget = min(total, max(d // order, requested))
            # Compile and warm both common TC and production SGM kernels outside timings.
            warm = TCOracle(tc_x, d // order)
            initial = random_partition(rng, d, order)
            warm.evaluate(initial)
            all_blocks = np.asarray(list(combinations(range(d), order)), dtype=np.int32)
            _select_validated(
                jnp.asarray(all_blocks),
                jnp.ones(total),
                dimension=d,
                n_seeds=args.sgm_seeds,
                early_stop=True,
            ).total_score.block_until_ready()
            baseline_score = None
            for method_index, method in enumerate(METHODS):
                method_rng = np.random.default_rng(seed + 4001)
                start = perf_counter()
                oracle = TCOracle(tc_x, total if method == "sgm" else budget)
                diagnostics = {}
                if method == "sgm":
                    values = oracle.evaluate(all_blocks)
                    selected = _select_validated(
                        jnp.asarray(all_blocks),
                        jnp.asarray(values),
                        dimension=d,
                        n_seeds=args.sgm_seeds,
                        early_stop=True,
                    )
                    chosen = canonical(np.asarray(selected.blocks))
                elif method == "stochastic":
                    chosen, diagnostics = stochastic_swaps(
                        oracle, random_partition(method_rng, d, order), d, order, method_rng
                    )
                else:
                    matrix = pair_screen(
                        tc_x, "pearson" if method == "pearson_screen" else "histogram"
                    )
                    pool_size = max(d // order, int(budget * 0.6)) if method == "hybrid" else budget
                    pool, scaffold = screened_pool(matrix, order, pool_size, method_rng)
                    chosen, diagnostics = restricted_optimum(oracle, pool, d, order, scaffold)
                    if method == "hybrid":
                        chosen, extra = stochastic_swaps(oracle, chosen, d, order, method_rng)
                        diagnostics.update(extra)
                elapsed = perf_counter() - start
                validate_partition(chosen, d, order)
                retained = float(oracle.evaluate(chosen).sum())
                if method == "sgm":
                    baseline_score = retained
                    # Independent check against the existing TC implementation.
                    reference = estimate_total_correlations(tc_x, np.asarray(initial))
                    np.testing.assert_allclose(
                        oracle.evaluate(initial), reference.scores, atol=1e-10
                    )
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "dimension": d,
                    "order": order,
                    "possible_blocks": total,
                    "unique_tc": len(oracle.cache),
                    "kernel_blocks": oracle.kernel_blocks,
                    "selection_seconds": elapsed,
                    "retained_tc": retained,
                    "score_fraction_of_sgm": retained / baseline_score,
                    "blocks": chosen,
                    **diagnostics,
                    **quality(chosen, target, fit_x, test_x, seed),
                }
                rows.append(row)
                print(
                    f"{scenario:18s} seed={seed} {method:15s} TC={len(oracle.cache):4d}/{total} "
                    f"score={row['score_fraction_of_sgm']:.3f} t={elapsed:.3f}s "
                    f"KL={row['heldout_kl']:.3f} logZ={row['log_z_error']:.3f}",
                    flush=True,
                )
                payload = {
                    "config": vars(args),
                    "environment": {
                        "jax": jax.__version__,
                        "scipy": scipy.__version__,
                        "numpy": np.__version__,
                        "python": platform.python_version(),
                        "devices": str(jax.devices()),
                        "x64": bool(jax.config.jax_enable_x64),
                    },
                    "rows": rows,
                }
                output.write_text(json.dumps(payload, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tc-samples", type=int, default=256)
    parser.add_argument("--fit-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--budget-fraction", type=float, default=0.2)
    parser.add_argument(
        "--tc-budget",
        type=int,
        default=None,
        help="Fixed unique-block budget; overrides the fraction",
    )
    parser.add_argument("--sgm-seeds", type=int, default=None, help="Default: try every SGM seed")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output", default="benchmark-results/lazy-partition-search/results.json")
    args = parser.parse_args()
    if args.dimension < 6 or args.dimension % 6:
        parser.error("This synthetic suite requires dimension >= 6 and divisible by 6")
    if not 0 < args.budget_fraction <= 1 or not 3 < args.tc_samples <= args.fit_samples:
        parser.error("Need 0 < budget fraction <= 1 and 3 < TC samples <= fit samples")
    if args.tc_budget is not None and args.tc_budget < args.dimension // 2:
        parser.error("Fixed budget must cover at least one full pair partition")
    run(args)


if __name__ == "__main__":
    main()
