import jax
from jax import numpy as jnp
from flax import nnx
import optax
from omegaconf import DictConfig

from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState

from .models import Encoder, Decoder, Tokenizer, Dynamics, RewardPredictor, TermPredictor, GRUActorCritic


class WorldModel(nnx.Module):
    def __init__(self, dynamics: Dynamics, reward_predictor: RewardPredictor, term_predictor: TermPredictor):
        self.dynamics = dynamics
        self.reward_predictor = reward_predictor
        self.term_predictor = term_predictor


class IRIS:
    def __init__(self, config: DictConfig):
        self.config = config

    def _init_models(self, rng: nnx.Rngs, env: Environment, env_state: EnvState):
        encoder = Encoder(rng, self.config.encoder)
        decoder = Decoder(rng, self.config.decoder)
        self.tok = Tokenizer(encoder, decoder)
        tok_tx = optax.chain(optax.clip_by_global_norm(self.config.tok_max_grad_norm), optax.contrib.muon(self.config.tok_lr))
        self.tok_opt = nnx.Optimizer(self.tok, tok_tx, wrt=nnx.Param)

        dynamics = Dynamics(rng, self.config.dynamics, env.action_space(env_state).n)
        reward_predictor = RewardPredictor(rng, self.config.reward_predictor)
        term_predictor = TermPredictor(rng, self.config.term_predictor)
        self.wm = WorldModel(dynamics, reward_predictor, term_predictor)
        wm_tx = optax.chain(optax.clip_by_global_norm(self.config.wm_max_grad_norm), optax.contrib.muon(self.config.wm_lr))
        self.wm_opt = nnx.Optimizer(self.wm, wm_tx, wrt=nnx.Param)

        self.ac = GRUActorCritic(rng, self.config.actor_critic, env.action_space(env_state).n)
        ac_tx = optax.chain(optax.clip_by_global_norm(self.config.ac_max_grad_norm), optax.contrib.muon(self.config.ac_lr))
        self.ac_opt = nnx.Optimizer(self.ac, ac_tx, wrt=nnx.Param)

    def fit(self, key: jax.Array, env: Environment):
        rng = nnx.Rngs(key)
        obs, env_state = jax.vmap(env.reset)(jnp.array(jax.random.split(rng.next_key(), self.config.num_worlds)))
        self._init_models(rng, env, env_state)
        self._log_models(obs)

        print("done")

    def _log_models(self, obs: jax.Array):
        print(nnx.tabulate(self.tok.encoder, obs))
        _, ids = self.tok.encode(obs)
        vq_toks = self.tok.encoder.codebook[ids]
        print(nnx.tabulate(self.tok.decoder, vq_toks))
