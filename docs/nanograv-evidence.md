# NANOGrav IRN, CURN, and HD with JAX MorphZ

`nano_lnlike.py` is adapted from
`/home/apokrypha/Local NT/Nanograv/nano_lnlike.py`. It retains the original
Enterprise models from `gwb_morph.ipynb`: fixed white noise, 30-component
per-pulsar red noise, and the marginalized timing model. Model selection adds
the notebook's common process as follows:

| `--model` | Common process | Default chain directory |
| --- | --- | --- |
| `irn` | None; intrinsic red noise only | `irn_100k` |
| `curn` (default) | Uncorrelated, 14 frequencies | `curn_100k` |
| `hd` | Hellings–Downs correlated, 14 frequencies | `HD_100k` |

The GW priors remain log10 amplitude Uniform(-18, -14) and gamma Uniform(0, 7).
Per-pulsar red noise uses log10 amplitude Uniform(-20, -11) and gamma Uniform(0, 7).
IRN has no shared GW parameters. Parent and spawned workers build the same
selected model. HD uses `FourierBasisCommonGP` with `utils.hd_orf()`.

Enterprise remains a NumPy host likelihood. JAX performs Morph fitting,
proposal sampling, proposal-density evaluation, and bridge sampling. Likelihood
workers use `spawn`, build their own PTA, and do not initialize JAX. Draws
outside prior support return `-inf` and remain in the bridge calculation.

## Run in WSL

From the Max directory, use the existing `nanograv` conda environment, which
contains Enterprise, h5pulsar, and JAX. The separate Max uv environment does
not currently contain Enterprise or h5pulsar. No old `morphZ` package is needed.

```bash
conda activate nanograv
export TEMPO2="$CONDA_PREFIX/share/tempo2"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu

# Small integration check, not a scientifically useful evidence estimate:
python nano_lnlike.py --max-samples 512 --n-proposal 256 \
  --max-tc-samples 16 --n-seeds 8 --workers 1 \
  --output benchmark-results/nanograv-curn-jax-quick

# Full retained chain using the current defaults:
python nano_lnlike.py --workers 4

# IRN and HD, with separate default chains and output directories:
python nano_lnlike.py --model irn --workers 1
python nano_lnlike.py --model hd --workers 1
```

The default data and chain paths point to the original NANOGrav directory in
WSL, with the chain folder chosen by `--model`, so running from Max does not
require copying the data. Override with
`--datadir` and `--chain-dir` when using another location. Use `--workers 1`
to limit memory use; otherwise there is one PTA per worker plus the parent.
Use `--help` for all options. CPU is the tested backend; GPU requires a suitable
JAX/CUDA installation in the same environment as Enterprise.

## Statistical settings and output

The script requires a cold chain at temperature 1 and the same parameter names
in `pars.txt` and `pta.param_names`. It reorders chain columns into PTA order
when necessary and rejects missing or duplicate names. Set `--burn` to the
number of original rows to discard and `--thin` to the spacing between retained rows.
The supplied chain calls the GW parameters `gw_crn_gamma` and `gw_crn_log10_A`;
these are explicitly mapped to the original script's `gamma_gw` and `log10_A_gw`.
The HD chain's `gw_HD_gamma` and `gw_HD_log10_A` receive the corresponding
mapping only when selecting HD. An HD chain is rejected for CURN and vice versa
when identified by these model-specific names.
The retained draws must be a stationary, equal-weight posterior sequence.
The first half fits Morph and the second half supplies the bridge posterior.
The bridge estimates IAT from that ordered held-out sequence unless
`--posterior-iat` is provided.
The IRN section of `ce_logZ.ipynb` discards 40,000 original rows; use
`--model irn --burn 40000` to reproduce that burn-in. The script keeps its
existing default burn of zero unless you specify it.

Posterior targets are recomputed by default with the current PTA model and its
normalized prior. `--use-cached-posterior` uses PTMCMC column -4 only when you
have verified that it stores the same normalized log likelihood plus log prior.
The original script read that column but did not pass it to MorphZ.

Order 2 is the default. TC scoring uses at most 200 randomly selected fitting
draws, reproducibly selected using `--seed`. `--max-tc-samples` can lower this
limit (1–200); every fit draw still contributes to the fitted KDE. `--n-seeds`
overrides the default five SGM seeds. These options can change the selected blocks.
Fitting at 136 dimensions can be expensive even with the TC sample cap.

Results go to a new output directory, default
`benchmark-results/nanograv-MODEL-jax`. Existing directories are refused.
`result.json` records log evidence, relative Monte Carlo error, convergence,
IAT, overlap, importance ESS, sample counts, dimension, runtime, and forward KL.
The JSON keeps the original first 14 lines (through `dimension`) and adds
`elapsed_seconds` and `kl`. It omits blocks, parameter lists, and settings.
Bridge density vectors, proposal samples, and solver histories are not saved.

Use `--n-estimates N` to fit Morph once and run N bridge estimates with
fresh proposal draws (`--n-proposal` per estimate). The held-out posterior
and its density evaluations are reused, so the estimates share posterior
sampling and fit uncertainty. With N > 1, `result.json` contains an ordered
`estimates` array and the mean and sample standard deviation of the finite,
converged log evidence values. The standard deviation is null if fewer than
two estimates qualify. Per-estimate runtime is cumulative from the start
of the fitting phase. A replicate with zero support overlap
gets a null evidence value and does not stop later estimates. The command
returns a failing status if any estimate fails to converge.

```bash
python nano_lnlike.py --model hd --no-plots --n-estimates 5 --workers 1
```

`kl` estimates KL(posterior || Morph) in nats on the held-out posterior:
mean(log likelihood + log prior - log Morph) - estimated log evidence.
It is conditional on the fitted Morph and evidence estimate; finite-sample
estimates can be negative. It is null if the bridge does not converge.

Three corner PNGs overlay the held-out posterior (blue) and raw Morph draws
(orange), covering parameter columns 1–12, 13–24, and 25–36 in PTA order.
Each plot has normalized diagonal histograms, shared ranges, and 68%/95%
two-dimensional contours. Plots are saved before bridge sampling, including
on zero-overlap failures. Lower-dimensional inputs produce fewer/smaller chunks.
Files are named `corner_01_12d.png`, `corner_02_12d.png`, and `corner_03_12d.png`.
Plots use the first proposal batch only. Pass `--no-plots` to skip plotting
and avoid requiring `corner` and `matplotlib` (the benchmark extra).

Nonconvergence returns a failing exit status after saving the summary and plots.
Tiny smoke runs test integration only; assess posterior coverage
and overlap before interpreting evidence.
If no proposals overlap the prior support, the bridge API refuses to estimate
evidence. In single-estimate mode, no evidence JSON is produced.
Increase fit/proposal counts
and inspect the chain rather than dropping out-of-support proposals.
Both a 64-row/16-proposal check and a 512-row/256-proposal check on the supplied
136-parameter chain ran both engines but had zero proposal overlap. Neither
produced a valid evidence estimate.

Adapter checks include a NumPy-only Gaussian with known evidence, both serial
and spawned evaluation, cold-chain selection, and prior-support short-circuiting:

```bash
pytest -q tests/test_nano_lnlike.py
```

IRN (134 parameters) and HD (136 parameters) have also been checked against
their real chains: parameter alignment and a finite posterior evaluation pass
in spawned Enterprise workers. These checks do not run a full evidence estimate.
