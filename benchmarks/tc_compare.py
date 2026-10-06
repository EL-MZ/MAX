"""Compare matching CPU/GPU TC runs, timings, inputs, and numerical scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .memory import format_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    cpu = json.loads((args.root / "cpu/results.json").read_text())
    gpu = json.loads((args.root / "gpu/results.json").read_text())
    if not cpu.get("complete") or not gpu.get("complete"):
        raise ValueError("Both benchmark runs must finish before comparison")
    for field in ("jax", "jaxlib", "precision"):
        if cpu["metadata"][field] != gpu["metadata"][field]:
            raise ValueError(f"CPU/GPU {field} differs; use matching environments")
    if cpu["metadata"]["backend"] != "cpu" or gpu["metadata"]["backend"] != "gpu":
        raise ValueError("Expected verified CPU and GPU runs")

    def key(record):
        return tuple(
            record[field]
            for field in (
                "scenario",
                "dimension",
                "order",
                "samples_per_parameter",
                "candidate_batch_size",
                "chunk_size",
                "progress_every",
                "candidates",
            )
        ) + (record.get("memory_interval"),)

    cpu_records = {key(record): record for record in cpu["records"]}
    gpu_records = {key(record): record for record in gpu["records"]}
    if cpu_records.keys() != gpu_records.keys():
        raise ValueError("CPU/GPU runs have different cases/settings or one run is incomplete")
    rows = []
    memory_lines = [
        "",
        "## Memory (MiB)",
        "",
        "Sampled peaks span the first pass and warm repeats. GPU device usage includes other "
        "applications; its baseline and peak increase are reported separately.",
        "",
        "| Distribution | D | Order | Batch | CPU RAM peak | GPU run RAM peak | "
        "GPU live peak | GPU pool peak | Device baseline | Device peak | Device increase |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# TC CPU/GPU comparison",
        "",
        "Warm median times include the same chunked public API workload on both devices.",
        "Speedup = CPU seconds / GPU seconds; greater than one means GPU is faster.",
        "",
        "| Distribution | D | Order | Batch | TCs | CPU (s) | GPU (s) | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, left in cpu_records.items():
        right = gpu_records[case]
        if left["input_sha256"] != right["input_sha256"]:
            raise ValueError(f"CPU/GPU input samples differ: {case}")
        np.testing.assert_allclose(
            [left[field] for field in ("raw_tc_mean", "raw_tc_min", "raw_tc_max")],
            [right[field] for field in ("raw_tc_mean", "raw_tc_min", "raw_tc_max")],
            rtol=1e-8,
            atol=1e-8,
        )
        difference = None
        if left["raw_score_file"] and right["raw_score_file"]:
            a = np.load(args.root / "cpu" / left["raw_score_file"], mmap_mode="r")
            b = np.load(args.root / "gpu" / right["raw_score_file"], mmap_mode="r")
            if a.shape != b.shape:
                raise ValueError(f"Score array shapes differ: {case}")
            difference = 0.0
            for start in range(0, len(a), 65536):
                part_a, part_b = a[start : start + 65536], b[start : start + 65536]
                difference = max(difference, float(np.max(np.abs(part_a - part_b))))
                np.testing.assert_allclose(part_a, part_b, rtol=1e-8, atol=1e-8)
        cpu_time, gpu_time = left["api_warm_median_seconds"], right["api_warm_median_seconds"]
        speedup = cpu_time / gpu_time
        row = {
            "scenario": left["scenario"],
            "dimension": left["dimension"],
            "order": left["order"],
            "candidate_batch_size": left["candidate_batch_size"],
            "candidates": left["candidates"],
            "all_candidates": left["all_candidates"],
            "samples_per_parameter": left["samples_per_parameter"],
            "cpu_seconds": cpu_time,
            "gpu_seconds": gpu_time,
            "gpu_speedup": speedup,
            "max_abs_raw_score_difference": difference,
        }
        for metric, record, prefix in (
            ("cpu_rss", left, "cpu_ram"),
            ("cpu_rss", right, "gpu_host_ram"),
            ("gpu_live", right, "gpu_live"),
            ("gpu_pool", right, "gpu_pool"),
            ("gpu_device_used", right, "gpu_device_used"),
        ):
            for statistic in ("baseline", "peak", "end", "peak_increase"):
                row[f"{prefix}_{statistic}_mib"] = record.get(f"{metric}_{statistic}_mib")
        row["gpu_allocator_lifetime_peak_mib"] = right.get("gpu_allocator_lifetime_peak_mib")
        row["gpu_pool_lifetime_peak_mib"] = right.get("gpu_pool_lifetime_peak_mib")
        rows.append(row)
        lines.append(
            f"| {left['scenario']} | {left['dimension']} | {left['order']} | "
            f"{left['candidate_batch_size']} | {left['candidates']:,} | "
            f"{cpu_time:.3f} | {gpu_time:.3f} | {speedup:.2f}x |"
        )
        memory_values = [
            row[field]
            for field in (
                "cpu_ram_peak_mib",
                "gpu_host_ram_peak_mib",
                "gpu_live_peak_mib",
                "gpu_pool_peak_mib",
                "gpu_device_used_baseline_mib",
                "gpu_device_used_peak_mib",
                "gpu_device_used_peak_increase_mib",
            )
        ]
        memory_lines.append(
            f"| {left['scenario']} | {left['dimension']} | {left['order']} | "
            f"{left['candidate_batch_size']} | " + " | ".join(map(format_mib, memory_values)) + " |"
        )
    lines += memory_lines
    lines += [
        "",
        "Identical sample hashes and candidate generation settings were verified. "
        "Raw TC mean/min/max agree within rtol=1e-8, atol=1e-8. Full score agreement "
        "is additionally checked when both runs saved scores.",
        "",
    ]
    (args.root / "comparison.md").write_text("\n".join(lines))
    with (args.root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
