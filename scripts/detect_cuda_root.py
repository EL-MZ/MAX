"""Locate the NVIDIA wheel's compiler tools without importing JAX."""

import importlib.util
from pathlib import Path


def find_cuda_root():
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except ModuleNotFoundError:
        return ""
    if spec is None:
        return ""
    # Namespace packages have no __file__/origin. Their search locations still
    # identify the installed wheel, including environments at paths with spaces.
    roots = list(spec.submodule_search_locations or [])
    if spec.origin:
        roots.append(str(Path(spec.origin).parent))
    for root in roots:
        path = Path(root)
        if (path / "bin/ptxas").is_file() and (path / "nvvm/libdevice/libdevice.10.bc").is_file():
            return str(path.resolve())
    raise RuntimeError(
        "nvidia.cuda_nvcc is installed but its ptxas/libdevice files are missing. "
        "Reinstall the project's cuda extra or set CUDA_ROOT to a complete CUDA toolkit."
    )


if __name__ == "__main__":
    print(find_cuda_root())
