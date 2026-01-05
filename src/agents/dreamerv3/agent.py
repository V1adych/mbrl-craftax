import jax
from jax import numpy as jnp
from flax import struct
from typing import Any
from omegaconf import DictConfig
from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState

from .models import Dynamics, Encoder, Decoder, Posterior, Prior, Actor, Critic


@struct.dataclass
class Models:
    encoder: Encoder
    dynamics: Dynamics
    decoder: Decoder
    posterior: Posterior
    prior: Prior
    actor: Actor
    critic: Critic


@struct.dataclass
class Params:
    encoder: Any
    dynamics: Any
    decoder: Any
    posterior: Any
    prior: Any
    actor: Any
    critic: Any


class DreamerV3:
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: Models = None
        self.params: Params = None

    def _init_models(self, key: jax.Array, env: Environment, env_state: EnvState):
        actspace = env.action_space(env_state).n
        obs_shape = env.observation_space(env_state).shape
        batch_size = self.config.num_worlds

        self.models = Models(
            encoder=Encoder(self.config.encoder),
            dynamics=Dynamics(self.config.dynamics, actspace),
            decoder=Decoder(self.config.decoder),
            posterior=Posterior(self.config.posterior),
            prior=Prior(self.config.prior),
            actor=Actor(self.config.actor, actspace),
            critic=Critic(self.config.critic),
        )

        k_enc, k_post, k_prior, k_act, k_crit, k_dec, k_dyn, k_sample = jax.random.split(key, 8)

        dummy_obs = jnp.zeros((batch_size, *obs_shape))
        params_encoder = self.models.encoder.init(k_enc, dummy_obs)
        tokens = self.models.encoder.apply(params_encoder, dummy_obs)

        deter = self.models.dynamics.get_initial_deter(batch_size)

        params_posterior = self.models.posterior.init(k_post, deter, tokens)
        stoch_dist = self.models.posterior.postprocess(self.models.posterior.apply(params_posterior, deter, tokens))
        stoch = stoch_dist.sample(seed=k_sample)

        params_prior = self.models.prior.init(k_prior, deter)

        params_actor = self.models.actor.init(k_act, deter, stoch)
        params_critic = self.models.critic.init(k_crit, deter, stoch)
        params_decoder = self.models.decoder.init(k_dec, deter, stoch)

        act_dist = self.models.actor.postprocess(self.models.actor.apply(params_actor, deter, stoch))
        act = act_dist.sample(seed=k_sample)

        params_dynamics = self.models.dynamics.init(k_dyn, deter, stoch, act)

        self.params = Params(
            encoder=params_encoder,
            dynamics=params_dynamics,
            decoder=params_decoder,
            posterior=params_posterior,
            prior=params_prior,
            actor=params_actor,
            critic=params_critic,
        )

    def fit(self, key: jax.Array, env: Environment):
        reset_fn = jax.vmap(env.reset)
        key, *reset_keys = jax.random.split(key, self.config.num_worlds + 1)
        obs, env_state = reset_fn(jnp.array(reset_keys))

        key, init_key = jax.random.split(key)
        self._init_models(init_key, env, env_state)
        self._log_models(env, env_state)

        # ... training loop ...

    def _log_models(self, env: Environment, env_state: EnvState):
        obs_shape = env.observation_space(env_state).shape
        batch_size = self.config.num_worlds
        tab_key = jax.random.key(0)

        print("\n" + "=" * 80)
        print(f"{'MODEL ARCHITECTURES':^80}")
        print("=" * 80)

        dummy_obs = jnp.zeros((batch_size, *obs_shape))
        print("\n[Encoder]")
        print(self.models.encoder.tabulate(tab_key, dummy_obs))

        deter = self.models.dynamics.get_initial_deter(batch_size)
        tokens = jnp.zeros((batch_size, self.config.encoder.hidden_size))
        stoch = jnp.zeros((batch_size, self.config.dynamics.stoch, self.config.dynamics.classes))
        act = jnp.zeros((batch_size,), dtype=jnp.int32)

        print("\n[Posterior]")
        print(self.models.posterior.tabulate(tab_key, deter, tokens, compute_flops=True, compute_vjp_flops=True))

        print("\n[Prior]")
        print(self.models.prior.tabulate(tab_key, deter, compute_flops=True, compute_vjp_flops=True))

        print("\n[Actor]")
        print(self.models.actor.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))

        print("\n[Critic]")
        print(self.models.critic.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))

        print("\n[Decoder]")
        print(self.models.decoder.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))

        print("\n[Dynamics]")
        print(self.models.dynamics.tabulate(tab_key, deter, stoch, act, compute_flops=True, compute_vjp_flops=True))
        print("=" * 80 + "\n")
