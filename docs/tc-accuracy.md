# TC accuracy versus sample count

Compare TC estimates from 10, 20, 30, 50, 100, 200, and 500 joint sample rows
with TC estimates from all 1,000 rows. This experiment uses the same public
KDE TC estimator as Morph's scoring stage and reports raw, unclipped scores.

```bash
python -m benchmarks.tc_accuracy --output benchmark-results/tc-accuracy \
  --dimensions 12 --orders 2 3 --repeats 100 --seed 20261006
```

The existing local WSL environment can run it with:

```powershell
wsl bash scripts/local-wsl.sh run python -m benchmarks.tc_accuracy --output benchmark-results/tc-accuracy --repeats 100
```

Defaults cover all six distributions in `benchmarks.distributions`: independent
mixed marginals, strong pairs, weak pairs, triplets, banana pairs, and multimodal
pairs. All column combinations are scored. Use `--scenarios strong_pairs` to
limit the run. Output must be a new or empty directory.

Each distribution has one fixed 1,000-row dataset. Each trial draws a uniformly
random subset without replacement; sample counts use nested prefixes of that
trial's permutation. Orders share the same subsets. Fitting and evaluating the
KDE on each subset matches the TC thinning used in production. The reference
uses all rows, float64, Silverman bandwidth, and zero regularization.

For candidate block b and trial r, the signed difference is
`TC_n[b, r] - TC_1000[b]`. The line averages these differences over all blocks
and trials. The band is plus/minus one sample standard deviation over the same
differences (`ddof=1`). The absolute-difference plot uses the absolute value
before computing those statistics. It reveals errors hidden by cancellation.

These are descriptive standard deviations, not confidence intervals. Candidates
overlap and sample counts share rows. The 1,000-row estimate is a finite-sample
reference, not exact population TC; overlapping subsets measure thinning error
on this dataset. This experiment does not run selection or impose a minimum
sample count or a significance test.

Outputs:

- `tc_difference.png` and `.pdf`: signed mean difference plus/minus SD.
- `tc_absolute_difference.png` and `.pdf`: mean absolute difference plus/minus SD.
- `summary.csv`: plot statistics, RMSE, and SD across trial-averaged errors.
- `scores_*.npz`: candidates, reference scores, subset scores, and row indices.
- `samples_*.npy`: the full reference datasets.
- `results.json`: configuration, target definitions, sample hashes, and timings.
- `README.md`: experiment details and a table of results.

The CSV and JSON are checkpointed after each completed distribution/order case.
Timing includes compilation and is recorded for context, not as a speed benchmark.
