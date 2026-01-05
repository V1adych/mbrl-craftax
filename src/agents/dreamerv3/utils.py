import jax
from jax import numpy as jnp


def symlog(x: jax.Array):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def two_hot_symlog(x: jax.Array, bins: jax.Array):
    x_symlog = symlog(x)
    idx = jnp.digitize(x_symlog, bins) - 1
    idx = jnp.clip(idx, 0, len(bins) - 2)
    bin_low = bins[idx]
    bin_high = bins[idx + 1]
    p_high = (x_symlog - bin_low) / (bin_high - bin_low)
    p_high = jnp.clip(p_high, 0.0, 1.0)
    one_hot_low = jax.nn.one_hot(idx, len(bins))
    one_hot_high = jax.nn.one_hot(idx + 1, len(bins))
    return (1.0 - p_high)[..., None] * one_hot_low + p_high[..., None] * one_hot_high
