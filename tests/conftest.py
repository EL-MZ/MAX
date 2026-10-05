import jax

# Set precision in the test application, never globally in library imports.
jax.config.update("jax_enable_x64", True)
