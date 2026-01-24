import jax
from jax import numpy as jnp
from typing import Any
from .base_wrapper import BaseWrapper


class ImageObsWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _to_uint8(self, obs: jax.Array) -> jax.Array:
        return (obs * 255).astype(jnp.uint8)

    def reset(self, key: jax.Array, params: Any | None = None):
        obs, env_state = self._env.reset(key, params)
        return self._to_uint8(obs), env_state

    def step(self, key: jax.Array, state: Any, action: jax.Array, params: Any | None = None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return self._to_uint8(obs), env_state, reward, done, info
