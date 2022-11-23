from jax.config import config; config.update("jax_enable_x64", True)    # Use 64 bit floating point numbers
import jax.numpy as npx
jax_dtype = npx.float64
