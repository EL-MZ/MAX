"""Optional flushed host reports from batched JAX computations."""

import jax
import jax.numpy as jnp

from ._validation import positive_integer


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
        print(f"    {label}: {int(completed):,}/{count:,}", flush=True)

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
