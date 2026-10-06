# TC scaling benchmark on CPU and GPU

## Linux

With `uv` on PATH, run from the repository root using `--uv`:

```bash
# CPU, GPU, and automatic timing comparison.
bash scripts/run-tc-benchmark.sh both --uv \
  --dimensions 100 150 200 --orders 2 3 \
  --scenarios mixed --samples 300 --repeats 1 \
  --progress-every 1024 --output benchmark-results/tc-scaling

# Higher orders, on smaller dimensions.
bash scripts/run-tc-benchmark.sh both --uv \
  --dimensions 30 50 --orders 2 3 4 --scenarios mixed \
  --repeats 1 --output benchmark-results/tc-order4
```

Linux list options use spaces, such as `--orders 2 3 4`. Omit `--scenarios` to run
all five distributions. With `--uv`, the launcher uses `uv run --locked --python 3.12`
and includes the project's `cuda` extra for GPU/both runs. uv installs missing
dependencies into the project environment before running. The same CUDA extra is
retained for the CPU pass and comparison, so those calls do not remove GPU libraries.
No virtual environment activation is needed.

The launcher detects `CUDA_ROOT` using the benchmark's Python environment before
importing JAX. This handles NVIDIA namespace wheels whose `cuda_nvcc.__file__`
is `None` with the pinned JAX 0.4.35. An explicitly supplied `CUDA_ROOT` is preserved.
For direct Python commands that import JAX, first set the same compiler path:

```bash
export CUDA_ROOT="$(uv run --locked --python 3.12 --extra cuda python scripts/detect_cuda_root.py)"
```

Without `--uv`, it uses `python3` from your active environment and does not install
dependencies; use `--python /path/to/venv/bin/python` to select another interpreter.
`--uv` and `--python` cannot be combined. Both modes force CPU/CUDA explicitly.
If WSL shares a checkout containing a Windows `.venv`, set `UV_PROJECT_ENVIRONMENT`
to a separate Linux environment path before using `--uv`.
The progress counters and all timing settings below also apply on Linux.

For separate device runs and comparison:

```bash
bash scripts/run-tc-benchmark.sh cpu --uv --dimensions 100 200 --orders 2 3 --output benchmark-results/tc-scaling
bash scripts/run-tc-benchmark.sh gpu --uv --dimensions 100 200 --orders 2 3 --output benchmark-results/tc-scaling
JAX_PLATFORMS=cpu uv run --locked --python 3.12 --extra cuda python -m benchmarks.tc_compare --root benchmark-results/tc-scaling
```

All Python benchmark flags are forwarded, including `--limit-candidates 10000`
for a capped pilot and `--save-scores` to retain raw scores. CPU and GPU runs
must use matching settings when comparing. Relative output paths are resolved
against the repository root; absolute Linux paths also work.

## Workload and timings

All candidate combinations are scored by default. Candidates are generated in
streamed outer chunks, so memory does not grow with the entire candidate list.
The number of TCs grows as `choose(dimensions, order)`:

| Dimensions | Order 2 | Order 3 | Order 4 |
|---:|---:|---:|---:|
| 100 | 4,950 | 161,700 | 3,921,225 |
| 150 | 11,175 | 551,300 | 20,260,275 |
| 200 | 19,900 | 1,313,400 | 64,684,950 |

Each case has an initial pass and three warm repeats by default. The script
prints the planned counts before starting. Orders must be 2..9, no larger than
the dimension, and smaller than the sample count. Samples are restricted to
4..300 joint rows, meaning at most 300 values for each parameter, regardless of D.

Progress counts completed candidate TCs for the current pass:

```text
START gpu mixed 100D order=3 TCs=161,700/161,700 samples=300 batch=32
  warm 1/3: TC 8,192 / 161,700 (5.1%) | elapsed ...s | ETA ...s
```

The counter resets for each distribution, dimension, order, candidate batch size,
and repeat. ETA is approximate during compilation. `--progress-every` defaults to
4,096 and controls the outer chunk size; `--candidate-batch-sizes` defaults to 32
and controls JAX's inner candidate batch size. Query KDE chunks use 128 rows.

Timings report the sum of synchronized public TC API calls and a separate wall
time. API time includes validation, transfers, and marginal KDE recomputation
for each outer chunk; it excludes candidate generation, printing, and saving.
Wall time includes those overheads. Use identical progress and batch sizes for
CPU/GPU comparisons. Initial passes may compile; only the first case for a given
shape and batch configuration is cold. Both devices use float64.

## Memory measurements

Memory is enabled automatically, shown with the progress counter, and saved in
JSON/CSV and the CPU/GPU comparison. All values use MiB (1,048,576 bytes):

- **RAM**: the benchmark process's resident memory from Linux `/proc/self/status`,
  including JAX, compiled code, samples, and host buffers. GPU runs also report
  their host RAM usage.
- **GPU live**: live allocations from JAX's GPU allocator.
- **GPU pool**: GPU memory held by JAX's allocator, including cached free space.
- **GPU device**: whole-device VRAM from `nvidia-smi`, including display use and
  other applications. Baseline, sampled peak, and peak increase are separate;
  the increase cannot isolate this benchmark if another application's usage changes.

For each metric, every case records baseline, sampled peak, end, and peak increase
above baseline. The case peak spans the first pass and all warm repeats; JSON also
stores each pass separately. RAM and JAX counters are sampled every 0.1 seconds;
change this with `--memory-interval 0.05`, for example. Whole-device VRAM is sampled
at most once per second, plus pass endpoints. Short transient peaks can be missed.

Native high-water counters (`cpu_process_hwm_mib`,
`gpu_allocator_lifetime_peak_mib`, `gpu_pool_lifetime_peak_mib`) are separately
labeled **process lifetime**, since they do not reset between cases. Allocator
caches from earlier cases remain resident. Only the default JAX device is monitored.
Unavailable GPU counters are `null` in JSON, blank in CSV, and `N/A` in reports.
Polling can add timing overhead; initial/final monitoring snapshots are outside
the TC timers. Use the same interval for CPU and GPU comparisons.

JAX provides [device memory statistics](https://docs.jax.dev/en/latest/_autosummary/jax.Device.html).
Whole-device queries use [NVIDIA's documented memory fields](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/kw/driver).

Five shuffled distributions are available with `--scenarios`:

| Scenario | Dependence |
|---|---|
| `independent` | Independent standard normals |
| `gaussian_pairs` | Gaussian pairs, rho = 0.85 |
| `gaussian_triplets` | Gaussian triples, rho = 0.7 (also pair dependent) |
| `three_way` | Independent signs A,B and their product A*B, each with independent Gaussian noise of SD 0.25; pairs are independent |
| `mixed` | Repeated Gaussian pair, noisy parity triple, independent singleton |

Residual columns are independent singletons; a residual Gaussian-triplet pair
retains rho = 0.7. Samples are identical across devices for the same seed. No
selection quality metrics are calculated. The 300-row cap is a workload choice;
it does not guarantee accurate TC estimates at every block order.

Use `--limit-candidates 10000` for a capped prefix pilot. It records the full
combinatorial count and marks the result as partial. Omit it to score every TC.

```bash
bash scripts/run-tc-benchmark.sh both --uv --dimensions 100 --orders 4 \
  --scenarios mixed --limit-candidates 10000 --repeats 1 \
  --output benchmark-results/tc-pilot
```

The Python entry point also supports multiple candidate batch sizes and a seed:

```bash
JAX_PLATFORMS=cpu uv run --locked --python 3.12 --extra cuda python -m benchmarks.tc_speed --backend cpu --output benchmark-results/tc-batches/cpu --dimensions 100 --orders 2 3 --scenarios mixed --candidate-batch-sizes 8 32 128 --repeats 1 --seed 42
```

Outputs under the chosen directory:

- `cpu/results.json` and `gpu/results.json`: settings, device metadata, sample
  hashes, timings, candidate counts, and raw TC mean/min/max for sanity checks.
- `cpu/timings.csv` and `gpu/timings.csv`: timing and memory rows, saved after every case.
- Sample `.npy` files, requiring only `300 * D * 8` bytes per distribution.
- `comparison.md` and `comparison.csv`: warm median CPU/GPU times, memory, and speedups.
  A speedup above one means GPU is faster. Matching sample hashes, settings,
  and aggregate score agreement are checked.

Score arrays are omitted by default to keep large runs small on disk. Add
`--save-scores` to write all first-pass raw TCs via a memory
map. This needs eight bytes per candidate per case/device. When both runs save
scores, comparison additionally verifies every raw TC in bounded chunks.

Use a fresh output directory for each configuration. Completed-case checkpoints
are saved, but reruns start over. Comparison requires both runs to finish.
Output paths can be relative to the repository or absolute Linux paths.
Interrupt the run with Ctrl+C if a configuration is too slow; completed cases
remain saved. Full scaling benchmark runs have been left for the user.
