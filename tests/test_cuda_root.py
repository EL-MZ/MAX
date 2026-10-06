"""CUDA wheel lookup must work before JAX can be imported."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.detect_cuda_root import find_cuda_root


class CudaRootTests(unittest.TestCase):
    def make_toolkit(self, root):
        (root / "bin").mkdir(parents=True)
        (root / "nvvm/libdevice").mkdir(parents=True)
        (root / "bin/ptxas").touch()
        (root / "nvvm/libdevice/libdevice.10.bc").touch()

    def test_namespace_package_without_file_at_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "environment with spaces" / "nvidia/cuda_nvcc"
            self.make_toolkit(root)
            spec = SimpleNamespace(origin=None, submodule_search_locations=[str(root)])
            with patch("scripts.detect_cuda_root.importlib.util.find_spec", return_value=spec):
                self.assertEqual(find_cuda_root(), str(root.resolve()))

    def test_regular_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cuda_nvcc"
            self.make_toolkit(root)
            spec = SimpleNamespace(
                origin=str(root / "__init__.py"), submodule_search_locations=None
            )
            with patch("scripts.detect_cuda_root.importlib.util.find_spec", return_value=spec):
                self.assertEqual(find_cuda_root(), str(root.resolve()))

    def test_cpu_environment_without_nvidia(self):
        with patch(
            "scripts.detect_cuda_root.importlib.util.find_spec", side_effect=ModuleNotFoundError
        ):
            self.assertEqual(find_cuda_root(), "")
        with patch("scripts.detect_cuda_root.importlib.util.find_spec", return_value=None):
            self.assertEqual(find_cuda_root(), "")

    def test_incomplete_cuda_wheel_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = SimpleNamespace(origin=None, submodule_search_locations=[temporary])
            with patch("scripts.detect_cuda_root.importlib.util.find_spec", return_value=spec):
                with self.assertRaisesRegex(RuntimeError, "ptxas/libdevice files are missing"):
                    find_cuda_root()


if __name__ == "__main__":
    unittest.main()
