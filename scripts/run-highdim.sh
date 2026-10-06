#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$backend" in
  cpu) export JAX_PLATFORMS=cpu ;;
  gpu) export JAX_PLATFORMS=cuda ;;
  *) echo 'Usage: bash scripts/run-highdim.sh cpu|gpu [--python PATH | --uv] [benchmark options]' >&2; exit 2 ;;
esac
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python_command=(python3)
if [[ "${1:-}" == --python ]]; then
  if [[ "$#" -lt 2 ]]; then echo '--python requires an executable path' >&2; exit 2; fi
  python_command=("$2")
  shift 2
elif [[ "${1:-}" == --uv ]]; then
  python_command=(uv run --locked --extra benchmark)
  if [[ "$backend" == gpu ]]; then python_command+=(--extra cuda); fi
  python_command+=(python)
  shift
elif [[ -x "$HOME/.local/share/max-jax/.venv/bin/python" ]]; then
  python_command=("$HOME/.local/share/max-jax/.venv/bin/python")
fi
# Compatibility for JAX 0.4.x with NVIDIA namespace-package wheels (also CPU).
if [[ -z "${CUDA_ROOT:-}" ]]; then
  detected_root="$("${python_command[@]}" scripts/detect_cuda_root.py)"
  if [[ -n "$detected_root" ]]; then export CUDA_ROOT="$detected_root"; fi
fi
exec "${python_command[@]}" -m benchmarks.highdim_suite --backend "$backend" "$@"
