#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/max-jax/.venv"
export UV_CACHE_DIR="$HOME/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# JAX 0.4.x expects cuda_nvcc.__file__; recent NVIDIA wheels use a namespace package.
export CUDA_ROOT="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvcc"
action="${1:-run}"
shift || true
if [[ "$action" == setup ]]; then
    if [[ -d .tools/wsl-wheels ]]; then
        if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
            uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
        fi
        uv export --locked --extra test --extra benchmark --extra cuda \
            --format requirements-txt --no-emit-project \
            --output-file "$UV_PROJECT_ENVIRONMENT/requirements.txt" >/dev/null
        uv pip sync --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            --no-index --find-links "$PWD/.tools/wsl-wheels" \
            "$UV_PROJECT_ENVIRONMENT/requirements.txt"
        uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            --no-index --find-links "$PWD/.tools/wsl-wheels" --no-deps -e .
    else
        uv sync --locked --python 3.12 --extra test --extra benchmark --extra cuda
    fi
else
    if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
        echo 'Run bash scripts/local-wsl.sh setup first.' >&2
        exit 1
    fi
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/check_backend.py gpu
    if [[ "$#" -gt 0 ]]; then
        executable="$1"
        shift
        "$UV_PROJECT_ENVIRONMENT/bin/$executable" "$@"
    fi
fi
