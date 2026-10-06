# Block distribution benchmarks

`benchmarks/distributions.py` provides normalized analytic targets inspired by
[Morph-approx-lab](https://github.com/EL-MZ/Morph-approx-lab) at commit
`7dded6163028ce6dbc1719fa783a3e3a67ea306b`. The Gaussian-copula construction uses
normal, gamma, beta, lognormal, exponential, uniform, and loguniform marginals.
Additional banana and Gaussian-mixture pairs test nonlinear and multimodal
dependence. This version uses local seeded RNGs and preserves zero density
outside support, rather than flooring likelihoods to finite values.

The default suite tests six target families at 6, 12, 20, and 30 dimensions,
with Morph orders 1, 2, and 3: 72 main fits plus seven 30D scoring ablations.
Every training, independent evaluation, and proposal batch contains 1,000
vectors, hence no more than 1,000 values per parameter in any batch.
Input columns are permuted before fitting. True-block order is used only for
display. Gaussian KDE factors retain all 1,000 fitting vectors; main total-
correlation scoring uses a seeded 256-point subset. Seven full-scoring ablations
use all 1,000 fitting vectors to expose the speed/accuracy tradeoff. SGM uses
the complete candidate collection and at most 64 seeds.

Forward KL is the mean of `log p(x) - log q(x)` on an independent draw from the
analytic target, with its sample standard deviation divided by sqrt(1000).
Raw estimates and 95% Monte Carlo intervals are saved without zero clipping.
Intervals are conditional on the fitted density; this single-training-seed
suite does not establish uncertainty across fits. A known-block KDE baseline
separates selection error from KDE smoothing error. A product-normal baseline
provides a simple density reference.

Block diagnostics include exact partition equality, exact block recall, and
precision/recall of within-block pairs. Orders 2 and 3 force full-size factors
and residual singletons, so a different true block size may be impossible to
recover exactly. Independent targets have undefined dependency recall, and
false-positive merges are still counted. Partition adjacency plots make these
differences visible.

Proposal diagnostics record importance ESS and support leakage. Gaussian KDE
has positive mass outside bounded or one-sided support, so reverse KL against
such targets is infinite, even if a finite proposal batch misses the leakage.
No rejected-draw renormalization or likelihood floors are applied.

Synchronized CPU timings cover complete fits, density evaluation, sampling,
and SGM alone. The first call may include compilation or reuse an earlier
shape's compilation; warm medians are reported separately. Warm fit timings
include host validation, scoring, selection, and final factors. They are not
a comparison against the original MorphZ or a GPU performance claim.

```bash
python -m pip install -e '.[test,benchmark]'
python -m benchmarks.block_suite --output benchmark-results/blocks-2026-10-05
```

For a short check, use `--dimensions 6 --scenarios strong_pairs banana_pairs
--skip-ablations`. The GitHub Actions workflow pins benchmark dependencies,
runs tests, executes the suite, and uploads measured artifacts for download.
Generated results stay outside Git in `benchmark-results/`.

The original merged `src/jax_morph` implementation is left unchanged; this
benchmark measures it without inserting a new estimator or selection rule.

To regenerate plots from existing recorded partitions and seeds, use
`python -m benchmarks.block_suite --replot-results --output
benchmark-results/blocks-2026-10-05`. This rebuilds factors and verifies every
main held-out KL value without rerunning TC scoring. Original measurements
and timings remain unchanged; `plot_metadata.json` records the rendering commit.

Existing local results may include `FINDINGS.md` describing tested recovery limits,
KDE smoothing errors, support leakage, and the full-scoring speed comparison.
