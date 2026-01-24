import numpy as np
import jax
from jax import numpy as jnp
from flax import struct
from omegaconf import DictConfig


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward_prev: jax.Array
    reward: jax.Array
    term: jax.Array
    reset: jax.Array


class ReplayBuffer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.num_worlds = config.num_worlds
        self.max_length = jnp.ceil(config.max_size / self.num_worlds).astype(jnp.int32).item()

        self.data = None
        self.ptr = None
        self.filled = None

    def init(self, transition: Transition):
        self.data = Transition(
            obs=np.zeros((self.max_length, self.num_worlds, *transition.obs.shape[2:]), dtype=transition.obs.dtype),
            action=np.zeros((self.max_length, self.num_worlds), dtype=transition.action.dtype),
            reward_prev=np.zeros((self.max_length, self.num_worlds), dtype=transition.reward_prev.dtype),
            reward=np.zeros((self.max_length, self.num_worlds), dtype=transition.reward.dtype),
            term=np.zeros((self.max_length, self.num_worlds), dtype=transition.term.dtype),
            reset=np.zeros((self.max_length, self.num_worlds), dtype=transition.reset.dtype),
        )
        self.ptr = 0
        self.filled = False

    def add(self, rollout: Transition):
        assert self.data is not None, "Replay buffer not initialized"

        t = rollout.obs.shape[0]
        ptr = self.ptr

        idx = (ptr + np.arange(t, dtype=np.int32)) % np.asarray(self.max_length, dtype=np.int32)  # (T,)

        self.data.obs[idx] = rollout.obs
        self.data.action[idx] = rollout.action
        self.data.reward_prev[idx] = rollout.reward_prev
        self.data.reward[idx] = rollout.reward
        self.data.term[idx] = rollout.term
        self.data.reset[idx] = rollout.reset

        self.filled = self.filled or (ptr + t) >= self.max_length
        self.ptr = (ptr + t) % self.max_length

    def sample(self, key: jax.Array, length: int, batch_size: int) -> Transition:
        if self.filled:
            cur_size = self.max_length
            prefix_offset = self.ptr
        else:
            cur_size = self.ptr
            prefix_offset = 0

        max_start = cur_size - (length - 1)
        assert max_start > 0, "Not enough data in replay buffer"

        key_t, key_env = jax.random.split(key, 2)
        start = jax.random.randint(key_t, (batch_size,), prefix_offset, max_start + prefix_offset, dtype=jnp.int32)
        env_id = jax.random.randint(key_env, (batch_size,), 0, self.data.obs.shape[1], dtype=jnp.int32)
        offsets = jnp.arange(length, dtype=jnp.int32)[:, None]
        time_idx = np.array((start[None, :] + offsets) % self.config.max_length)
        env_idx = np.array(env_id[None, :])

        def _gather(buf: np.ndarray) -> jax.Array:
            return jnp.asarray(buf[time_idx, env_idx])

        return Transition(
            obs=_gather(self.data.obs),
            action=_gather(self.data.action),
            reward_prev=_gather(self.data.reward_prev),
            reward=_gather(self.data.reward),
            term=_gather(self.data.term),
            reset=_gather(self.data.reset),
        )

    def can_sample(self, length: int) -> bool:
        return self.filled or self.ptr > length
