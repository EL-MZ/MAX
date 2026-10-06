"""Memory metrics distinguish per-case samples, lifetime peaks, and missing data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.memory import (
    MIB,
    MemoryMonitor,
    NvidiaMemory,
    combine_pass_memory,
    read_cpu_memory,
    read_jax_memory,
)


class BenchmarkMemoryTests(unittest.TestCase):
    def test_linux_rss_and_hwm_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status"
            status.write_text("Name:\tpython\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n")
            self.assertEqual(
                read_cpu_memory(status), {"cpu_rss_mib": 2.0, "cpu_process_hwm_mib": 4.0}
            )
            self.assertEqual(read_cpu_memory(status.with_name("missing")), {})

    def test_sampled_peak_is_not_lifetime_peak(self):
        values = iter((10.0, 20.0, 12.0))
        monitor = MemoryMonitor(
            cpu_reader=lambda: {
                "cpu_rss_mib": next(values),
                "cpu_process_hwm_mib": 100.0,
            }
        )
        for _ in range(3):
            monitor.sample()
        record = monitor.report()
        self.assertEqual(record["cpu_rss_baseline_mib"], 10.0)
        self.assertEqual(record["cpu_rss_peak_mib"], 20.0)
        self.assertEqual(record["cpu_rss_end_mib"], 12.0)
        self.assertEqual(record["cpu_rss_peak_increase_mib"], 10.0)
        self.assertEqual(record["cpu_process_hwm_mib"], 100.0)

    def test_gpu_zero_and_unavailable_remain_distinct(self):
        device = SimpleNamespace(
            memory_stats=lambda: {
                "bytes_in_use": 0,
                "pool_bytes": 2 * MIB,
                "peak_bytes_in_use": 4 * MIB,
            }
        )
        self.assertEqual(
            read_jax_memory(device),
            {
                "gpu_live_mib": 0.0,
                "gpu_pool_mib": 2.0,
                "gpu_allocator_lifetime_peak_mib": 4.0,
            },
        )
        self.assertEqual(read_jax_memory(None), {})
        unsupported = SimpleNamespace(memory_stats=lambda: None)
        self.assertEqual(read_jax_memory(unsupported), {})
        monitor = MemoryMonitor(cpu_reader=lambda: {}, gpu_reader=lambda: read_jax_memory(device))
        monitor.sample()
        self.assertEqual(monitor.report()["gpu_live_peak_mib"], 0.0)
        self.assertIsNone(monitor.report()["cpu_rss_peak_mib"])
        self.assertIsNone(monitor.report()["gpu_device_used_peak_mib"])

    def test_case_aggregate_includes_first_pass_and_warm_repeats(self):
        first = MemoryMonitor(cpu_reader=lambda: {"cpu_rss_mib": 10.0})
        first.sample()
        first.cpu_reader = lambda: {"cpu_rss_mib": 30.0}
        first.sample()
        warm = MemoryMonitor(cpu_reader=lambda: {"cpu_rss_mib": 15.0})
        warm.sample()
        record = combine_pass_memory([first.report(), warm.report()])
        self.assertEqual(record["cpu_rss_peak_mib"], 30.0)
        self.assertEqual(record["cpu_rss_baseline_mib"], 10.0)
        self.assertEqual(record["cpu_rss_end_mib"], 15.0)
        self.assertEqual(record["cpu_rss_peak_increase_mib"], 20.0)
        self.assertIsNone(record["gpu_live_peak_mib"])

    def test_monitor_stops_on_computation_error(self):
        monitor = MemoryMonitor(cpu_reader=lambda: {"cpu_rss_mib": 2.0}, interval=0.01)
        with self.assertRaisesRegex(RuntimeError, "computation"):
            with monitor:
                raise RuntimeError("computation")
        self.assertFalse(monitor.thread.is_alive())
        self.assertEqual(monitor.final_report["cpu_rss_peak_mib"], 2.0)

    @patch("benchmarks.memory.shutil.which", return_value="/bin/nvidia-smi")
    def test_device_query_maps_visible_gpu_and_parses_mib(self, unused_which):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3,1"}):
            reader = NvidiaMemory(SimpleNamespace(local_hardware_id=0, id=0))
        with patch(
            "benchmarks.memory.subprocess.run", return_value=SimpleNamespace(stdout="256, 8192\n")
        ) as run:
            self.assertEqual(
                reader(), {"gpu_device_used_mib": 256.0, "gpu_device_total_mib": 8192.0}
            )
            self.assertIn("--id=3", run.call_args.args[0])
        with patch(
            "benchmarks.memory.subprocess.run", return_value=SimpleNamespace(stdout="[N/A], 8192\n")
        ):
            self.assertEqual(reader(), {})
            self.assertFalse(reader.available)

    def test_comparison_exports_memory_and_marks_missing_gpu_counters(self):
        from benchmarks.tc_compare import main

        base = {
            "scenario": "mixed",
            "dimension": 100,
            "order": 2,
            "samples_per_parameter": 300,
            "candidate_batch_size": 32,
            "chunk_size": 128,
            "progress_every": 4096,
            "memory_interval": 0.1,
            "candidates": 4950,
            "all_candidates": True,
            "input_sha256": "same-input",
            "raw_tc_mean": 0.1,
            "raw_tc_min": 0.0,
            "raw_tc_max": 1.0,
            "raw_score_file": None,
            "api_warm_median_seconds": 2.0,
            "cpu_rss_peak_mib": 100.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend in ("cpu", "gpu"):
                folder = root / backend
                folder.mkdir()
                record = dict(base)
                if backend == "gpu":
                    record.update(
                        api_warm_median_seconds=1.0,
                        cpu_rss_peak_mib=150.0,
                        gpu_live_peak_mib=8.0,
                        gpu_pool_peak_mib=10.0,
                    )
                (folder / "results.json").write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "metadata": {
                                "backend": backend,
                                "jax": "0.4.35",
                                "jaxlib": "0.4.35",
                                "precision": "float64",
                            },
                            "records": [record],
                        }
                    )
                )
            with patch("sys.argv", ["tc_compare", "--root", temporary]), patch("builtins.print"):
                main()
            report = (root / "comparison.md").read_text()
            self.assertIn("CPU RAM peak", report)
            self.assertIn("100.0 | 150.0 | 8.0 | 10.0 | N/A", report)
            with (root / "comparison.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["gpu_live_peak_mib"], "8.0")
            self.assertEqual(row["gpu_device_used_peak_mib"], "")


if __name__ == "__main__":
    unittest.main()
