"""tqdm rendering of live completion events from the native JAX kernels."""

from contextlib import contextmanager

from tqdm import tqdm

from jax_morph._progress import progress_renderer


@contextmanager
def tqdm_progress(enabled=True):
    bars = {}
    postfixes = {}
    units = {"candidate scores": "candidate", "SGM seeds": "seed", "Bridge iterations": "iter"}

    def render(label, completed, total, details):
        if label not in bars:
            bars[label] = tqdm(
                total=total,
                desc="Bridge" if label == "Bridge iterations" else label,
                unit=units.get(label, "item"),
                dynamic_ncols=True,
                bar_format=(
                    "{desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}{postfix}]"
                    if label == "Bridge iterations"
                    else None
                ),
            )
        bar = bars[label]
        postfix = postfixes.setdefault(label, {})
        if "log_z" in details:
            postfix["logZ"] = f"{details['log_z']:.6f}"
        if "delta" in details:
            postfix["delta"] = f"{details['delta']:.3g}"
        if details.get("finished"):
            bar.total = completed
            postfix["status"] = details["status"]
        if "reported_log_z" in details:
            postfix.pop("logZ", None)
            postfix.pop("delta", None)
            postfix["mean logZ"] = f"{details['reported_log_z']:.6f}"
        bar.set_postfix(postfix, refresh=False)
        bar.update(max(0, completed - bar.n))
        if details.get("finished") or "reported_log_z" in details:
            bar.refresh()
        if total is not None and completed == total and label != "Bridge iterations":
            bar.close()

    try:
        with progress_renderer(render if enabled else None):
            yield
    finally:
        for bar in bars.values():
            bar.close()
