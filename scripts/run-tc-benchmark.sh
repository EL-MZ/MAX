#!/usr/bin/env bash
# Use the active Python environment, or opt into uv's locked project environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/run-tc-benchmark.sh [both|cpu|gpu] [OPTIONS]

Default: both devices, 100D, orders 2 and 3, 300 samples per parameter.
Use --uv to install/run the locked project dependencies, or activate an environment.

Launcher options:
  --uv              Use uv run --locked (adds the cuda extra for gpu/both)
  --python PATH     Python executable (default: python3, or TC_PYTHON)
  --output DIR      Results root (default: benchmark-results/tc-scaling)

Other options are forwarded to benchmarks.tc_speed, for example:
  --dimensions 100 150 200 --orders 2 3 --scenarios mixed --repeats 1
  --samples 300 --progress-every 1024 --candidate-batch-sizes 32
  --memory-interval 0.1
  --limit-candidates 10000 --save-scores

Both mode runs CPU, then GPU, then creates comparison.md and comparison.csv.
HELP
}

device=both
if [[ $# -gt 0 ]]; then
    case "$1" in
        both|cpu|gpu) device="$1"; shift ;;
        cuda) device=gpu; shift ;;
    esac
fi

task_python="${TC_PYTHON:-python3}"
use_uv=false
custom_python=false
output=benchmark-results/tc-scaling
benchmark_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --uv) use_uv=true; shift ;;
        --python|--output)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --python ]]; then
                task_python="$2"
                custom_python=true
            else
                output="$2"
            fi
            shift 2
            ;;
        --python=*) task_python="${1#*=}"; custom_python=true; shift ;;
        --output=*) output="${1#*=}"; shift ;;
        --backend|--backend=*)
            printf 'Choose cpu, gpu, or both as the first argument, rather than --backend.\n' >&2
            exit 2
            ;;
        *) benchmark_args+=("$1"); shift ;;
    esac
done
if [[ -z "$task_python" || -z "$output" ]]; then
    printf 'Python executable and output directory must be nonempty.\n' >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$use_uv" == true ]]; then
    if [[ "$custom_python" == true ]]; then
        printf 'Use either --uv or --python, rather than both.\n' >&2
        exit 2
    fi
    if ! command -v uv >/dev/null 2>&1; then
        printf 'uv is not on PATH. Install uv or use --python with a configured environment.\n' >&2
        exit 1
    fi
    python_command=(uv run --locked --python 3.12)
    if [[ "$device" == both || "$device" == gpu ]]; then
        python_command+=(--extra cuda)
    fi
    python_command+=(python)
else
    if ! command -v "$task_python" >/dev/null 2>&1; then
        printf 'Python executable not found: %s\n' "$task_python" >&2
        exit 1
    fi
    python_command=("$task_python")
fi

# JAX 0.4.35 imports cuda_nvcc.__file__ even when CPU is selected. Recent
# NVIDIA wheels are namespace packages, so discover their tools before JAX
# imports and set CUDA_ROOT to bypass that incompatible lookup. Use the exact
# same Python/uv environment as the benchmark; preserve an explicit user value.
if [[ -z "${CUDA_ROOT:-}" ]]; then
    task_cuda_root="$("${python_command[@]}" scripts/detect_cuda_root.py)"
    if [[ -n "$task_cuda_root" ]]; then
        export CUDA_ROOT="$task_cuda_root"
        printf 'CUDA_ROOT: %s\n' "$CUDA_ROOT"
    fi
fi

run_backend() {
    local backend="$1"
    local jax_platform="$backend"
    if [[ "$backend" == gpu ]]; then jax_platform=cuda; fi
    # Explicit platform selection prevents a GPU run from falling back to CPU.
    JAX_PLATFORMS="$jax_platform" JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "${python_command[@]}" -m benchmarks.tc_speed --backend "$backend" \
        --output "$output/$backend" "${benchmark_args[@]}"
}

if [[ "$device" == both || "$device" == cpu ]]; then run_backend cpu; fi
if [[ "$device" == both || "$device" == gpu ]]; then run_backend gpu; fi
if [[ "$device" == both ]]; then
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
        "${python_command[@]}" -m benchmarks.tc_compare --root "$output"
fi
