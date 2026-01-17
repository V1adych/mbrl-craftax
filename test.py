from typing import Tuple
import jax
from jax import numpy as jnp

def compute_lambda_returns(rewards: jax.Array, conts: jax.Array, values: jax.Array, last_value: jax.Array, gamma: float, lam: float):
    def _lambda_return_step(next_ret: jax.Array, rew_cont_nval: Tuple[jax.Array, jax.Array, jax.Array]):
        reward, cont, next_value = rew_cont_nval
        ret = reward + cont * gamma * ((1 - lam) * next_value + lam * next_ret)
        return ret, ret

    _, returns = jax.lax.scan(_lambda_return_step, last_value, (rewards, conts, jnp.concatenate([values[1:], last_value[None]], axis=0)), reverse=True)
    return returns


rewards = jnp.array([0, 1, 0, 1, 0, 0], dtype=jnp.float32)[:, None]
conts = jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.float32)[:, None]
values = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)[:, None]
last_value = jnp.array([1], dtype=jnp.float32)

returns = compute_lambda_returns(rewards, conts, values, last_value, 0.99, 0.0)
print(returns.tolist())
