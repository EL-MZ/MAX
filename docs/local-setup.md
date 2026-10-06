# Linux CPU and CUDA testing with uv

Run from the repository root with `uv` on PATH. The import name is `jax_morph`.
uv uses the project's Python 3.12 environment and lockfile. Dependencies are
installed before execution; virtual environment activation is not required.

## CPU

```bash
uv sync --locked --python 3.12 --extra test --extra benchmark
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true uv run --locked --extra test --extra benchmark python scripts/check_backend.py cpu
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true uv run --locked --extra test --extra benchmark pytest -q
```

## CUDA

The project's `cuda` extra pins JAX 0.4.35, its CUDA 12 plugin, and cuDNN
9.1.1.17. Keep that extra enabled for GPU commands:

```bash
uv sync --locked --python 3.12 --extra test --extra benchmark --extra cuda
export CUDA_ROOT="$(uv run --locked --python 3.12 --extra test --extra benchmark --extra cuda python scripts/detect_cuda_root.py)"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --extra test --extra benchmark --extra cuda python scripts/check_backend.py gpu
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --extra test --extra benchmark --extra cuda pytest -q
```

Explicit `JAX_PLATFORMS` prevents GPU checks from silently falling back to CPU.
To compare CPU and GPU in the same CUDA environment, retain `--extra cuda` on
CPU calls too. Omitting extras during synchronization may remove their optional
dependencies from the environment.

## TC scaling benchmark

```bash
bash scripts/run-tc-benchmark.sh both --uv \
  --dimensions 100 150 200 --orders 2 3 \
  --scenarios mixed --samples 300 --repeats 1 --progress-every 1024
```

The launcher runs CPU then GPU and compares their timings. See
[TC benchmark options](tc-benchmark.md) for higher orders, capped pilots, and
score output. Generated results in `benchmark-results/` are ignored by Git;
existing local output is retained.

Without `--uv`, the launcher uses your active `python3`. You can select an
existing environment with `--python /path/to/venv/bin/python`.

## Other checks

```bash
JAX_PLATFORMS=cpu uv run --locked --extra test ruff check .
JAX_PLATFORMS=cpu uv run --locked --extra test ruff format --check .
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true uv run --locked python examples/gaussian_evidence.py
```

The application library leaves global JAX precision settings unchanged. Outside
the benchmark launcher, enable `jax_enable_x64` when float64 is required.

If a WSL checkout shares a Windows `.venv`, set `UV_PROJECT_ENVIRONMENT` to a
separate Linux path before running uv. The optional `scripts/local-wsl.sh`
helper keeps its environment at `~/.local/share/max-jax/.venv`; use
`bash scripts/local-wsl.sh setup` then `bash scripts/local-wsl.sh run python
scripts/check_backend.py gpu`. It also configures `CUDA_ROOT` for the existing
JAX 0.4.x/NVIDIA wheel layout. Native Linux TC runs use `run-tc-benchmark.sh`.
