from typing import Any, Tuple, Dict
import jax
from jax import numpy as jnp
from flax import struct
from flax.training import train_state
import optax
from einops import rearrange
from omegaconf import DictConfig
from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState

from .models import Dynamics, Encoder, Decoder, Posterior, Prior, Actor, Critic
from .replay_buffer import Transition, ReplayBuffer, ReplayBufferState


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
class Variables:
    encoder: Any
    dynamics: Any
    decoder: Any
    posterior: Any
    prior: Any
    actor: Any
    critic: Any


@struct.dataclass
class DreamerCarry:
    env_state: EnvState
    last_obs: jax.Array
    last_deter: jax.Array
    last_term: jax.Array


class DreamerState(train_state.TrainState):
    global_step: int


class DreamerV3:
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: Models = None
        self.replay: ReplayBuffer = None

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
        embed = self.models.encoder.apply(params_encoder, dummy_obs)

        deter = self.models.dynamics.get_initial_deter(batch_size)

        params_posterior = self.models.posterior.init(k_post, deter, embed)
        stoch_dist = self.models.posterior.postprocess(self.models.posterior.apply(params_posterior, deter, embed))
        stoch = stoch_dist.sample(seed=k_sample)

        params_prior = self.models.prior.init(k_prior, deter)

        params_actor = self.models.actor.init(k_act, deter, stoch)
        params_critic = self.models.critic.init(k_crit, deter, stoch)
        params_decoder = self.models.decoder.init(k_dec, deter, stoch)

        act_dist = self.models.actor.postprocess(self.models.actor.apply(params_actor, deter, stoch))
        act = act_dist.sample(seed=k_sample)

        params_dynamics = self.models.dynamics.init(k_dyn, deter, stoch, act)

        return Variables(
            encoder=params_encoder,
            dynamics=params_dynamics,
            decoder=params_decoder,
            posterior=params_posterior,
            prior=params_prior,
            actor=params_actor,
            critic=params_critic,
        ), params_critic

    def fit(self, key: jax.Array, env: Environment):
        reset_fn = jax.vmap(env.reset)
        key, *reset_keys = jax.random.split(key, self.config.num_worlds + 1)
        obs, env_state = reset_fn(jnp.array(reset_keys))

        key, init_key = jax.random.split(key)
        variables, slow_critic_params = self._init_models(init_key, env, env_state)
        self._log_models(env, env_state)

        def _is_params(path, _):
            for p in path:
                if isinstance(p, jax.tree_util.DictKey) and p.key == "params":
                    return True
            return False

        tx = optax.masked(
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.contrib.muon(learning_rate=self.config.lr, weight_decay=self.config.weight_decay),
            ),
            jax.tree.map_with_path(_is_params, variables),
        )
        init_deter = self.models.dynamics.get_initial_deter(self.config.num_worlds)
        init_term = jnp.ones((self.config.num_worlds,), dtype=jnp.bool)
        ts = DreamerState(
            step=0,
            apply_fn=None,
            params=variables,
            tx=tx,
            opt_state=tx.init(variables),
            global_step=0,
        )
        carry = DreamerCarry(env_state=env_state, last_obs=obs, last_deter=init_deter, last_term=init_term)

        obs_shape = env.observation_space(env_state).shape
        dummy = Transition(
            obs=jnp.zeros((0, self.config.num_worlds, *obs_shape), dtype=obs.dtype),
            action=jnp.zeros((0, self.config.num_worlds), dtype=jnp.int32),
            log_prob=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            reward=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            term=jnp.zeros((0, self.config.num_worlds), dtype=jnp.bool),
            is_first=jnp.zeros((0, self.config.num_worlds), dtype=jnp.bool),
        )
        self.replay = ReplayBuffer(self.config.replay_buffer)
        replay_state: ReplayBufferState = self.replay.init(dummy)

        num_samples = self.config.batch_size * self.config.num_grad_steps
        sample_length = self.config.replay_length + self.config.batch_length

        def _collect_rollout_step(state: Tuple[jax.Array, DreamerState, DreamerCarry], _):
            key, ts, carry = state
            deter = carry.last_deter
            obs = carry.last_obs
            embed = self.models.encoder.apply(ts.params.encoder, obs)
            stoch_posterior_dist = self.models.posterior.postprocess(self.models.posterior.apply(ts.params.posterior, deter, embed))
            key, sample_key, step_key = jax.random.split(key, 3)
            stoch_posterior = stoch_posterior_dist.sample(seed=sample_key)
            policy = self.models.actor.postprocess(self.models.actor.apply(ts.params.actor, deter, stoch_posterior))
            action = policy.sample(seed=sample_key)
            log_prob = policy.log_prob(action)
            obs_new, env_state, reward, done, info = jax.vmap(env.step)(jax.random.split(step_key, self.config.num_worlds), carry.env_state, action)
            deter_new = self.models.dynamics.apply(ts.params.dynamics, deter, stoch_posterior, action)
            deter_new = jnp.where(done[:, None], self.models.dynamics.get_initial_deter(self.config.num_worlds), deter_new)

            transition = Transition(obs=obs, action=action, log_prob=log_prob, reward=reward, term=done, is_first=carry.last_term)
            ts = ts.replace(global_step=ts.global_step + self.config.num_worlds)
            carry_new = DreamerCarry(env_state=env_state, last_obs=obs_new, last_deter=deter_new, last_term=done)

            def log_info(info: Dict[str, Any], global_step: int):
                return  # TODO: add tensorboard logging here

            jax.debug.callback(log_info, info, ts.global_step)

            return (key, ts, carry_new), transition

        def _update_loop(state: Tuple[jax.Array, DreamerState, DreamerCarry, ReplayBufferState], _):
            key, ts, carry, replay_state = state

            last_state, rollout = jax.lax.scan(_collect_rollout_step, (key, ts, carry), length=self.config.rollout_length)

            replay_state = self.replay.add(replay_state, rollout)
            key, ts, carry = last_state

            def _update_step(state: Tuple[jax.Array, DreamerState], minibatch: Transition):
                # TODO: will be implemented later
                return state, None

            key, sample_key = jax.random.split(key)
            batch = self.replay.sample(replay_state, sample_key, sample_length, num_samples)
            batch = jax.tree.map(lambda x: rearrange(x, "t (g b) ... -> g t b ...", g=self.config.num_grad_steps, b=self.config.batch_size), batch)

            last_state, metrics = jax.lax.scan(_update_step, (key, ts), batch)

            return (key, ts, carry, replay_state), None

        def _fit(key: jax.Array, ts: DreamerState, carry: DreamerCarry, replay_state: ReplayBufferState):
            to_prefill = sample_length - self.config.rollout_length

            def _prefill(key: jax.Array, ts: DreamerState, carry: DreamerCarry, replay_state: ReplayBufferState, to_prefill: int):
                (key, ts, carry), prefill_rollout = jax.lax.scan(_collect_rollout_step, (key, ts, carry), length=to_prefill)
                replay_state = self.replay.add(replay_state, prefill_rollout)
                return key, ts, carry, replay_state

            key, ts, carry, replay_state = jax.lax.cond(
                to_prefill > 0, lambda: _prefill(key, ts, carry, replay_state, to_prefill), lambda *_: (key, ts, carry, replay_state)
            )
            state, _ = jax.lax.scan(_update_loop, (key, ts, carry, replay_state), length=self.config.num_updates)
            _, ts, _, _ = state
            return ts

        ts = jax.jit(_fit)(key, ts, carry, replay_state)

    def _log_models(self, env: Environment, env_state: EnvState):
        obs_shape = env.observation_space(env_state).shape
        batch_size = self.config.num_worlds
        tab_key = jax.random.key(0)

        dummy_obs = jnp.zeros((batch_size, *obs_shape))
        print(self.models.encoder.tabulate(tab_key, dummy_obs, compute_flops=True, compute_vjp_flops=True))
        deter = self.models.dynamics.get_initial_deter(batch_size)
        embed = jnp.zeros((batch_size, self.config.encoder.hidden_size))
        stoch = jnp.zeros((batch_size, self.config.dynamics.stoch, self.config.dynamics.classes))
        act = jnp.zeros((batch_size,), dtype=jnp.int32)
        print(self.models.posterior.tabulate(tab_key, deter, embed, compute_flops=True, compute_vjp_flops=True))
        print(self.models.prior.tabulate(tab_key, deter, compute_flops=True, compute_vjp_flops=True))
        print(self.models.actor.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.critic.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.decoder.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.dynamics.tabulate(tab_key, deter, stoch, act, compute_flops=True, compute_vjp_flops=True))
