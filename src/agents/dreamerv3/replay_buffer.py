from typing import Any
import jax
from jax import numpy as jnp
from flax import struct
from omegaconf import DictConfig


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward_prev: jax.Array
    term: jax.Array
    reset: jax.Array


@struct.dataclass
class ReplayBufferState:
    data: Transition
    ptr: jax.Array
    filled: jax.Array


@struct.dataclass
class ReplayBuffer:
    config: DictConfig

    def init(self, data: Transition):
        t, b = data.obs.shape[:2]
        sz = self.config.max_length
        assert t <= sz, f"Expected rollout length to be less than or equal to max size, got t={t} and max_length={sz}"

        def _init_from_sample(arr: jax.Array, value: Any, t: int, b: int):
            data = jnp.full((t, b, *arr.shape[2:]), value, dtype=arr.dtype)
            data = data.at[: arr.shape[0]].set(arr)
            return data

        obs = _init_from_sample(data.obs, 0, sz, b)
        action = _init_from_sample(data.action, 0, sz, b)
        reward_prev = _init_from_sample(data.reward_prev, 0, sz, b)
        term = _init_from_sample(data.term, True, sz, b)
        reset = _init_from_sample(data.reset, True, sz, b)

        return ReplayBufferState(
            data=Transition(obs=obs, action=action, reward_prev=reward_prev, term=term, reset=reset),
            ptr=jnp.asarray(t, dtype=jnp.int32),
            filled=jnp.asarray(False, dtype=jnp.bool),
        )

    def add(self, state: ReplayBufferState, rollout: Transition) -> ReplayBufferState:
        sz = self.config.max_length
        t = rollout.obs.shape[0]
        ptr = state.ptr

        idx = (ptr + jnp.arange(t, dtype=jnp.int32)) % jnp.asarray(sz, dtype=jnp.int32)  # (T,)

        def _write(buf: jax.Array, x: jax.Array) -> jax.Array:
            return buf.at[idx].set(x)

        data = state.data
        new_data = Transition(
            obs=_write(data.obs, rollout.obs),
            action=_write(data.action, rollout.action),
            reward_prev=_write(data.reward_prev, rollout.reward_prev),
            term=_write(data.term, rollout.term),
            reset=_write(data.reset, rollout.reset),
        )

        wrote_past_end = (ptr + t) >= sz
        new_ptr = (ptr + t) % sz
        new_filled = jnp.logical_or(state.filled, wrote_past_end)
        return ReplayBufferState(data=new_data, ptr=new_ptr, filled=new_filled)

    def sample(
        self,
        state: ReplayBufferState,
        key: jax.Array,
        length: int,
        batch_size: int,
    ) -> Transition:
        data = state.data
        ptr = state.ptr

        cur_size = jnp.where(state.filled, self.config.max_length, ptr)
        prefix_offset = jnp.where(state.filled, ptr, 0)
        max_start = cur_size - (length - 1)

        key_t, key_env = jax.random.split(key, 2)
        start = jax.random.randint(key_t, (batch_size,), prefix_offset, max_start + prefix_offset, dtype=jnp.int32)
        env_id = jax.random.randint(key_env, (batch_size,), 0, data.obs.shape[1], dtype=jnp.int32)
        offsets = jnp.arange(length, dtype=jnp.int32)[:, None]
        time_idx = (start[None, :] + offsets) % self.config.max_length
        env_idx = jnp.broadcast_to(env_id[None, :], time_idx.shape)

        def _gather(buf: jax.Array) -> jax.Array:
            return buf[time_idx, env_idx]

        return Transition(
            obs=_gather(data.obs),
            action=_gather(data.action),
            reward_prev=_gather(data.reward_prev),
            term=_gather(data.term),
            reset=_gather(data.reset),
        )
