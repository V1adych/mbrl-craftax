import jax
from jax import numpy as jnp
from flax import linen
from omegaconf import DictConfig


class RetNorm(linen.Module):
    config: DictConfig

    def setup(self):
        self.q_low = self.variable("state", "q_low", lambda: jnp.asarray(self.config.low_init, dtype=jnp.float32))
        self.q_high = self.variable("state", "q_high", lambda: jnp.asarray(self.config.high_init, dtype=jnp.float32))

    def scale(self):
        return jnp.maximum(jnp.asarray(self.config.min_scale, dtype=jnp.float32), self.q_high.value - self.q_low.value)

    def update(self, returns: jax.Array):
        returns = returns.astype(jnp.float32)

        q_low_new = jnp.quantile(returns, q=float(self.config.low_quantile))
        q_high_new = jnp.quantile(returns, q=float(self.config.high_quantile))

        tau = float(self.config.tau)
        self.q_low.value = tau * self.q_low.value + (1.0 - tau) * q_low_new
        self.q_high.value = tau * self.q_high.value + (1.0 - tau) * q_high_new

    def __call__(self, x: jax.Array):
        return x / self.scale()
