import jax
from flax import nnx
import optax
from omegaconf import DictConfig

from gymnax.environments.environment import Environment


class IRIS:
    def __init__(self, config: DictConfig):
        self.config = config

    def _init_models(self, rng: nnx.Rngs): ...

    def fit(self, key: jax.Array, env: Environment):
        rng = nnx.Rngs(key)
        self._init_models(rng)

        ...
