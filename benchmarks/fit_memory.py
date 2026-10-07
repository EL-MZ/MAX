"""Measure one complete native Morph fit without evaluating or sampling it."""

import argparse
import csv
import importlib
import os
import platform
import resource
from math import comb
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import jax
import numpy as np

from .highdim_io import digest, save_json
from .regression_target import make_regression_target


def memory_snapshot():
    fields = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    rss, swap, hwm = (int(fields[key].split()[0]) / 1024 for key in ("VmRSS", "VmSwap", "VmHWM"))
    return {"rss_mib": rss, "swap_mib": swap, "rss_plus_swap_mib": rss + swap, "hwm_mib": hwm}


class FitMonitor:
    def __init__(self, output, configuration):
        self.output, self.configuration = output, configuration
        self.stop_event, self.lock = Event(), Lock()
        self.started = monotonic()
        self.stage = "initialization"
        self.phases, self.peaks = {}, {}
        self.writer = csv.writer((output / "memory.csv").open("w", buffering=1))
        self.writer.writerow(
            ["elapsed_seconds", "stage", "rss_mib", "swap_mib", "rss_plus_swap_mib"]
        )
        self.sample()
        self.thread = Thread(target=self.poll, daemon=True)
        self.thread.start()

    def sample(self):
        with self.lock:
            values = memory_snapshot()
            elapsed = monotonic() - self.started
            phase = self.phases.setdefault(
                self.stage, {"start_seconds": elapsed, "baseline": values}
            )
            phase["end_seconds"] = elapsed
            phase["end"] = values
            for key, value in values.items():
                if key == "hwm_mib":
                    continue
                phase[key.replace("_mib", "_peak_mib")] = max(
                    value, phase.get(key.replace("_mib", "_peak_mib"), 0)
                )
                self.peaks[key] = max(value, self.peaks.get(key, 0))
            self.writer.writerow(
                [
                    round(elapsed, 4),
                    self.stage,
                    values["rss_mib"],
                    values["swap_mib"],
                    values["rss_plus_swap_mib"],
                ]
            )
            return values

    def poll(self):
        last = monotonic()
        while not self.stop_event.wait(0.1):
            values = self.sample()
            if monotonic() - last >= 20:
                print(
                    f"  {self.stage}: RSS {values['rss_mib']:.1f} MiB; "
                    f"swap {values['swap_mib']:.1f} MiB",
                    flush=True,
                )
                last = monotonic()

    def set_stage(self, stage):
        self.sample()
        with self.lock:
            self.stage = stage
        values = self.sample()
        print(
            f"STAGE {stage}: RSS {values['rss_mib']:.1f} MiB; swap {values['swap_mib']:.1f} MiB",
            flush=True,
        )

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        return {"sampled_peaks_mib": self.peaks, "phases": self.phases, "final": memory_snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimensions", type=int, default=2000)
    parser.add_argument("--fit-samples", type=int, default=1000)
    parser.add_argument("--tc-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory to preserve previous measurements")
    args.output.mkdir(parents=True)
    jax.config.update("jax_enable_x64", True)
    morph = importlib.import_module("jax_morph.morph")
    configuration = {
        "dimensions": args.dimensions,
        "fit_samples": args.fit_samples,
        "tc_samples": args.tc_samples,
        "order": 2,
        "n_seeds": 10,
        "candidate_batch_size": 32,
        "seed_batch_size": 5,
        "sgm_early_stop": True,
        "query_chunk_size": 128,
        "seed": args.seed,
        "group_orders": [2],
        "variance_coordinate": "variance",
        "backend": "cpu",
        "precision": "float64",
        "pid": os.getpid(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "morph_source": morph.__file__,
        "note": "Fresh process, first-call compilation included. Measures fit only. "
        "Phase wrappers only record boundaries; native fitting algorithms are unchanged. "
        "RSS plus swap is sampled, not a native high-water counter. Other jobs can affect "
        "timing and resident-versus-swapped memory. MiB = 2**20 bytes.",
    }
    save_json(args.output / "configuration.json", configuration)
    save_json(args.output / "results.json", {"configuration": configuration, "complete": False})
    monitor = FitMonitor(args.output, configuration)
    originals = {}
    summary = {"configuration": configuration, "complete": False}
    try:
        monitor.set_stage("data_generation")
        target = make_regression_target(args.dimensions, args.seed, group_orders=(2,))
        stream = np.random.SeedSequence([args.seed, args.dimensions]).spawn(3)[0]
        training = target.sample(np.random.default_rng(stream), args.fit_samples)
        assert training.shape == (args.fit_samples, args.dimensions)
        assert np.isfinite(training).all()
        summary["input_sha256"] = digest(training)
        summary["input_mib"] = training.nbytes / 1024**2
        save_json(args.output / "target.json", target.description())
        for name, stage in (
            ("candidate_blocks", "candidate_enumeration"),
            ("estimate_total_correlations", "tc_search"),
            ("_select_validated", "sgm_selection"),
        ):
            original = getattr(morph, name)
            originals[name] = original

            def measured(*positional, _original=original, _stage=stage, **keywords):
                monitor.set_stage(_stage)
                value = _original(*positional, **keywords)
                jax.block_until_ready(value)
                monitor.sample()
                return value

            setattr(morph, name, measured)
        originals["_fit_factors"] = morph._fit_factors

        def measured_kde(*positional, **keywords):
            if monitor.stage != "final_kde_fits":
                monitor.set_stage("final_kde_fits")
            return originals["_fit_factors"](*positional, **keywords)

        morph._fit_factors = measured_kde
        monitor.set_stage("fit_validation")
        before_fit = memory_snapshot()
        started = monotonic()
        fitted = morph.fit(
            training,
            key=jax.random.split(jax.random.PRNGKey(args.seed))[0],
            order=2,
            n_seeds=10,
            max_tc_samples=args.tc_samples,
            chunk_size=128,
            candidate_batch_size=32,
            seed_batch_size=5,
            sgm_early_stop=True,
            max_candidates=comb(args.dimensions, 2),
            candidate_progress_every=max(1, comb(args.dimensions, 2) // 20),
            seed_progress_every=5,
        )
        jax.block_until_ready(fitted)
        jax.effects_barrier()
        fit_seconds = monotonic() - started
        assert fitted.diagnostics.fit_sample_count == args.fit_samples
        assert fitted.diagnostics.tc_sample_count == args.tc_samples
        assert len(fitted.diagnostics.candidates) == comb(args.dimensions, 2)
        assert sorted(i for block in fitted.state.blocks for i in block) == list(
            range(args.dimensions)
        )
        summary.update(
            complete=True,
            fit_seconds=fit_seconds,
            before_fit=before_fit,
            candidate_count=len(fitted.diagnostics.candidates),
            factor_count=len(fitted.state.factors),
            fit_sample_count=fitted.diagnostics.fit_sample_count,
            tc_sample_count=fitted.diagnostics.tc_sample_count,
            fitted_array_mib=sum(
                getattr(leaf, "nbytes", 0) for leaf in jax.tree_util.tree_leaves(fitted)
            )
            / 1024**2,
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        for name, original in originals.items():
            setattr(morph, name, original)
        summary["memory"] = monitor.stop()
        summary["process_native_max_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        save_json(args.output / "results.json", summary)
    print(
        f"COMPLETE: {summary['fit_seconds']:.2f}s; "
        f"native peak RSS {summary['process_native_max_rss_mib']:.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
