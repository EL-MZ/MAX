"""Verify device selection and run a small Morph fit on the requested backend."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from jax_morph import fit, log_prob, sample

jax.config.update("jax_enable_x64", True)


def main():
    expected = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    devices = jax.devices()
    if not devices or any(device.platform != expected for device in devices):
        raise RuntimeError(f"Requested {expected}, but JAX selected {devices}")
    print(f"Python: {sys.version.split()[0]}; JAX: {jax.__version__}")
    print(f"Devices: {devices}")
    print(f"Package: {__import__('jax_morph').__file__}")
    points = jax.random.normal(jax.random.key(0), (64, 3), dtype=jnp.float64)
    fitted = fit(points, order=2, max_tc_samples=32, key=jax.random.key(1))
    draws = jax.jit(lambda key: sample(key, fitted.state, 16))(jax.random.key(2))
    values = log_prob(fitted.state, draws).block_until_ready()
    assert draws.shape == (16, 3)
    assert np.isfinite(np.asarray(values)).all()
    assert all(device.platform == expected for device in values.devices())
    print(f"Morph fit/sample/log_prob passed on {expected}; blocks: {fitted.state.blocks}")


if __name__ == "__main__":
    main()
