import jax
from jax import numpy as jnp
from flax import nnx
import optax
from omegaconf import DictConfig

from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState

from .models import Encoder, Dynamics, RewardPredictor, TermPredictor, GRUActorCritic


class IRIS:
    def __init__(self, config: DictConfig):
        self.config = config

    def _init_models(self, rng: nnx.Rngs, env: Environment, env_state: EnvState):
        self.encoder = Encoder(rng, self.config.encoder)
        self.dynamics = Dynamics(rng, self.config.dynamics, env.action_space(env_state).n)
        self.reward_predictor = RewardPredictor(rng, self.config.reward_predictor)
        self.term_predictor = TermPredictor(rng, self.config.term_predictor)
        self.actor_critic = GRUActorCritic(rng, self.config.actor_critic, env.action_space(env_state).n)

        self.tok_tx = optax.chain(optax.clip_by_global_norm(self.config.tok_max_grad_norm), optax.contrib.muon(self.config.tok_lr))
        self.wm_tx = optax.chain(optax.clip_by_global_norm(self.config.wm_max_grad_norm), optax.contrib.muon(self.config.wm_lr))
        self.ac_tx = optax.chain(optax.clip_by_global_norm(self.config.ac_max_grad_norm), optax.contrib.muon(self.config.ac_lr))

    def fit(self, key: jax.Array, env: Environment):
        rng = nnx.Rngs(key)
        env_state = jax.vmap(env.reset)(jnp.array(jax.random.split(rng.next_key(), self.config.num_worlds)))
        self._init_models(rng, env, env_state)

        print("done")
