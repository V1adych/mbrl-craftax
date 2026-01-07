import jax
from jax import numpy as jnp
from flax import struct
from omegaconf import DictConfig


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    term: jax.Array
    is_first: jax.Array


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

        def _init_from_sample(arr: jax.Array, t: int, b: int):
            data = jnp.zeros((t, b, *arr.shape[2:]), dtype=arr.dtype)
            data = data.at[: arr.shape[0]].set(arr)
            return data

        obs = _init_from_sample(data.obs, sz, b)
        action = _init_from_sample(data.action, sz, b)
        log_prob = _init_from_sample(data.log_prob, sz, b)
        reward = _init_from_sample(data.reward, sz, b)
        term = _init_from_sample(data.term, sz, b)
        is_first = _init_from_sample(data.is_first, sz, b)

        return ReplayBufferState(
            data=Transition(obs=obs, action=action, log_prob=log_prob, reward=reward, term=term, is_first=is_first),
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
            log_prob=_write(data.log_prob, rollout.log_prob),
            reward=_write(data.reward, rollout.reward),
            term=_write(data.term, rollout.term),
            is_first=_write(data.is_first, rollout.is_first),
        )

        wrote_past_end = (ptr + jnp.asarray(t, dtype=jnp.int32)) >= jnp.asarray(sz, dtype=jnp.int32)
        new_ptr = (ptr + jnp.asarray(t, dtype=jnp.int32)) % jnp.asarray(sz, dtype=jnp.int32)
        new_filled = jnp.logical_or(state.filled, wrote_past_end)
        return ReplayBufferState(data=new_data, ptr=new_ptr, filled=new_filled)

    def sample(
        self,
        state: ReplayBufferState,
        key: jax.Array,
        length: int,
        batch_size: int,
    ) -> Transition:
        sz = self.config.max_length
        data = state.data
        ptr = state.ptr

        b_env = data.obs.shape[1]
        b_env = jnp.asarray(b_env, dtype=jnp.int32)
        sz_i32 = jnp.asarray(sz, dtype=jnp.int32)

        max_start = jnp.where(state.filled, sz_i32, ptr - jnp.asarray(length - 1, dtype=jnp.int32))
        key_t, key_env = jax.random.split(key, 2)
        start = jax.random.randint(key_t, (batch_size,), 0, max_start, dtype=jnp.int32)
        env_id = jax.random.randint(key_env, (batch_size,), 0, b_env, dtype=jnp.int32)

        offs = jnp.arange(length, dtype=jnp.int32)[:, None]
        time_idx = (start[None, :] + offs) % sz_i32
        env_idx = jnp.broadcast_to(env_id[None, :], time_idx.shape)

        def _gather(buf: jax.Array) -> jax.Array:
            return buf[time_idx, env_idx]

        out = Transition(
            obs=_gather(data.obs),
            action=_gather(data.action),
            log_prob=_gather(data.log_prob),
            reward=_gather(data.reward),
            term=_gather(data.term),
            is_first=_gather(data.is_first),
        )

        boundary_first = time_idx == ptr
        out = out.replace(is_first=jnp.logical_or(out.is_first, boundary_first))

        ptrm1 = (ptr - jnp.asarray(1, dtype=jnp.int32)) % sz_i32
        boundary_term = time_idx == ptrm1
        out = out.replace(term=jnp.logical_or(out.term, boundary_term))
        return out
