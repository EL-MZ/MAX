"""Optional host progress events from batched JAX computations."""

from contextlib import contextmanager
from threading import RLock

import jax
import jax.numpy as jnp
from tqdm import trange

from ._validation import positive_integer

_renderer = None
_renderer_lock = RLock()


@contextmanager
def progress_renderer(renderer):
    """Scope a process-wide renderer to a synchronously awaited benchmark stage.

    Callbacks run on JAX worker threads, so they cannot use a thread-local
    context. Serialize renderer scopes and drain callbacks before restoring it.
    The compiled kernels never capture the renderer or require recompilation.
    """
    global _renderer
    with _renderer_lock:
        previous, _renderer = _renderer, renderer
        try:
            yield
        finally:
            try:
                jax.effects_barrier()
            finally:
                _renderer = previous


def emit_progress(label, completed, total=None, **details):
    """Return whether a scoped renderer consumed this host-side event."""
    if _renderer is None:
        return False
    _renderer(label, int(completed), total, details)
    return True


@contextmanager
def candidate_progress(total, *, enabled):
    """Render candidate callbacks with trange unless a caller supplies a renderer."""
    with _renderer_lock:
        if not enabled or _renderer is not None:
            yield
            return
        with trange(total, desc="candidate scores", unit="candidate", dynamic_ncols=True) as bar:

            def render(label, completed, total, details):
                bar.update(max(0, completed - bar.n))

            with progress_renderer(render):
                yield


def report_count(label, completed, total):
    if not emit_progress(label, completed, total):
        print(f"    {label}: {int(completed):,}/{total:,}", flush=True)


def validate_progress_every(value):
    if value != 0 or isinstance(value, bool):
        positive_integer(value, "progress_every")


def progress_map(function, xs, *, batch_size, progress_every, label):
    """Use the same scan/vmap batching as lax.map, reporting between batches.

    Keeping callbacks outside vmap avoids one host callback per candidate.
    Zero progress_every preserves the original lax.map path.
    """
    if not progress_every:
        return jax.lax.map(function, xs, batch_size=batch_size)
    count = len(xs)
    full_count, remainder = divmod(count, batch_size)
    batches = xs[: full_count * batch_size].reshape((full_count, batch_size) + xs.shape[1:])

    def report(completed, dependency):
        del dependency
        report_count(label, completed, count)

    def emit(completed, values):
        # A scalar dependency keeps reporting after this batch's computation.
        dependency = jax.tree_util.tree_leaves(values)[0].reshape(-1)[0]
        jax.debug.callback(report, completed, dependency, ordered=True)

    def batch(_, item):
        index, values = item
        result = jax.vmap(function)(values)
        completed = (index + 1) * batch_size
        previous = index * batch_size
        show = (completed // progress_every != previous // progress_every) | (completed == count)
        jax.lax.cond(show, lambda _: emit(completed, result), lambda _: None, None)
        return None, result

    _, outputs = jax.lax.scan(batch, None, (jnp.arange(full_count), batches))
    outputs = jax.tree_util.tree_map(
        lambda value: value.reshape((full_count * batch_size,) + value.shape[2:]),
        outputs,
    )
    if remainder:
        tail = jax.vmap(function)(xs[full_count * batch_size :])
        emit(jnp.asarray(count), tail)
        outputs = jax.tree_util.tree_map(
            lambda prefix, suffix: jnp.concatenate((prefix, suffix)), outputs, tail
        )
    return outputs
