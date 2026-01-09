from typing import Any, Tuple, Dict, Optional, Callable
import jax
from jax import numpy as jnp
from flax import struct
from flax.training import train_state
import optax
from einops import rearrange
from omegaconf import DictConfig
from gymnax.environments.environment import Environment
from craftax.craftax_classic.envs.craftax_state import EnvState

from .models import Dynamics, Encoder, ObsDecoder, RewardPredictor, ContPredictor, Posterior, Prior, Actor, Critic
from .replay_buffer import Transition, ReplayBuffer, ReplayBufferState
from .utils import kl_divergence


@struct.dataclass
class Models:
    encoder: Encoder
    dynamics: Dynamics
    obs_decoder: ObsDecoder
    reward_predictor: RewardPredictor
    cont_predictor: ContPredictor
    posterior: Posterior
    prior: Prior
    actor: Actor
    critic: Critic


@struct.dataclass
class Variables:
    encoder: Any
    dynamics: Any
    obs_decoder: Any
    reward_predictor: Any
    cont_predictor: Any
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


@struct.dataclass
class Losses:
    dec_obs: jax.Array
    dec_reward: jax.Array
    dec_cont: jax.Array
    dyn: jax.Array
    rep: jax.Array
    actor: jax.Array
    critic: jax.Array
    entropy: jax.Array


class DreamerState(train_state.TrainState):
    global_step: int


class DreamerV3:
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: Models = None
        self.replay: ReplayBuffer = None

        self.loss_weights: Losses = Losses(
            dec_obs=jnp.float32(self.config.loss.dec_obs),
            dec_reward=jnp.float32(self.config.loss.dec_reward),
            dec_cont=jnp.float32(self.config.loss.dec_cont),
            dyn=jnp.float32(self.config.loss.dyn),
            rep=jnp.float32(self.config.loss.rep),
            actor=jnp.float32(self.config.loss.actor),
            critic=jnp.float32(self.config.loss.critic),
            entropy=jnp.float32(self.config.loss.entropy),
        )

    def _init_models(self, key: jax.Array, env: Environment, env_state: EnvState):
        actspace = env.action_space(env_state).n
        obs_shape = env.observation_space(env_state).shape
        batch_size = self.config.num_worlds

        self.models = Models(
            encoder=Encoder(self.config.encoder),
            dynamics=Dynamics(self.config.dynamics, actspace),
            obs_decoder=ObsDecoder(self.config.obs_decoder),
            reward_predictor=RewardPredictor(self.config.reward_predictor),
            cont_predictor=ContPredictor(self.config.cont_predictor),
            posterior=Posterior(self.config.posterior),
            prior=Prior(self.config.prior),
            actor=Actor(self.config.actor, actspace),
            critic=Critic(self.config.critic),
        )

        k_enc, k_post, k_prior, k_act, k_crit, k_obs_dec, k_rew, k_cont, k_dyn, k_sample = jax.random.split(key, 10)

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
        params_obs_decoder = self.models.obs_decoder.init(k_obs_dec, deter, stoch)
        params_reward_predictor = self.models.reward_predictor.init(k_rew, deter, stoch)
        params_cont_predictor = self.models.cont_predictor.init(k_cont, deter, stoch)

        act_dist = self.models.actor.postprocess(self.models.actor.apply(params_actor, deter, stoch))
        act = act_dist.sample(seed=k_sample)

        params_dynamics = self.models.dynamics.init(k_dyn, deter, stoch, act)

        return Variables(
            encoder=params_encoder,
            dynamics=params_dynamics,
            obs_decoder=params_obs_decoder,
            reward_predictor=params_reward_predictor,
            cont_predictor=params_cont_predictor,
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
        ts = DreamerState(step=0, apply_fn=None, params=variables, tx=tx, opt_state=tx.init(variables), global_step=0)
        carry = DreamerCarry(env_state=env_state, last_obs=obs, last_deter=init_deter, last_term=init_term)

        obs_shape = env.observation_space(env_state).shape
        dummy = Transition(
            obs=jnp.zeros((0, self.config.num_worlds, *obs_shape), dtype=obs.dtype),
            action=jnp.zeros((0, self.config.num_worlds), dtype=jnp.int32),
            log_prob=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            reward=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            term=jnp.zeros((0, self.config.num_worlds), dtype=jnp.bool),
            reset=jnp.zeros((0, self.config.num_worlds), dtype=jnp.bool),
        )
        self.replay = ReplayBuffer(self.config.replay_buffer)
        replay_state: ReplayBufferState = self.replay.init(dummy)

        num_samples = self.config.batch_size * self.config.num_grad_steps
        sample_length = self.config.replay_length + self.config.batch_length

        def _update_loop(state: Tuple[jax.Array, DreamerState, DreamerCarry, ReplayBufferState], _):
            key, ts, carry, replay_state = state
            last_state, rollout = self.collect_rollouts(key, ts, carry, env, self.config.rollout_length)

            replay_state = self.replay.add(replay_state, rollout)
            key, ts, carry = last_state

            def _update_step(state: Tuple[jax.Array, DreamerState], minibatch: Transition):
                key, ts = state
                key, observe_key = jax.random.split(key)
                batch_size = minibatch.obs.shape[1]
                init_deter = self.models.dynamics.get_initial_deter(batch_size)
                deter, stoch, stoch_posterior_probs = self.observe(observe_key, ts, minibatch, init_deter)
                deter, stoch, stoch_posterior_probs, minibatch = jax.tree.map(
                    lambda x: x[self.config.replay_length :], (deter, stoch, stoch_posterior_probs, minibatch)
                )

                obs_pred = self.models.obs_decoder.apply(ts.params.obs_decoder, deter, stoch)
                loss_obs = self.models.obs_decoder.apply(ts.params.obs_decoder, obs_pred, minibatch.obs, method=self.models.obs_decoder.loss)

                reward_logits = self.models.reward_predictor.apply(ts.params.reward_predictor, deter, stoch)
                loss_reward = self.models.reward_predictor.apply(
                    ts.params.reward_predictor, reward_logits, minibatch.reward, method=self.models.reward_predictor.loss
                )

                cont_logits = self.models.cont_predictor.apply(ts.params.cont_predictor, deter, stoch)
                cont_target = 1.0 - minibatch.term.astype(jnp.float32)
                loss_cont = self.models.cont_predictor.apply(ts.params.cont_predictor, cont_logits, cont_target, method=self.models.cont_predictor.loss)

                stoch_prior_probs = self.models.prior.postprocess(self.models.prior.apply(ts.params.prior, deter)).probs
                kl_dyn = kl_divergence(jax.lax.stop_gradient(stoch_posterior_probs), stoch_prior_probs)
                kl_rep = kl_divergence(stoch_posterior_probs, jax.lax.stop_gradient(stoch_prior_probs))
                kl_clipfrac = jnp.mean(jnp.float32(kl_dyn < self.config.loss.free_nats))
                loss_dyn = jnp.maximum(kl_dyn, self.config.loss.free_nats).mean()
                loss_rep = jnp.maximum(kl_rep, self.config.loss.free_nats).mean()

                imag_init = jax.tree.map(lambda x: rearrange(x[-self.config.imag_last_states :], "t b ... -> (t b) ..."), (deter, stoch))

                return (key, ts), None

            key, sample_key = jax.random.split(key)
            batch = self.replay.sample(replay_state, sample_key, sample_length, num_samples)
            batch = jax.tree.map(lambda x: rearrange(x, "t (g b) ... -> g t b ...", g=self.config.num_grad_steps, b=self.config.batch_size), batch)

            last_state, metrics = jax.lax.scan(_update_step, (key, ts), batch)

            return (key, ts, carry, replay_state), None

        def _fit(key: jax.Array, ts: DreamerState, carry: DreamerCarry, replay_state: ReplayBufferState):
            to_prefill = sample_length - self.config.rollout_length

            def _prefill(key: jax.Array, ts: DreamerState, carry: DreamerCarry, replay_state: ReplayBufferState, to_prefill: int):
                (key, ts, carry), prefill_rollout = self.collect_rollouts(key, ts, carry, env, to_prefill)
                replay_state = self.replay.add(replay_state, prefill_rollout)
                return key, ts, carry, replay_state

            state = (key, ts, carry, replay_state)
            state = jax.lax.cond(to_prefill > 0, lambda: _prefill(*state, to_prefill), lambda *_: state)
            state, _ = jax.lax.scan(_update_loop, state, length=self.config.num_updates)
            _, ts, _, _ = state
            return ts

        ts = jax.jit(_fit)(key, ts, carry, replay_state)

    def collect_rollouts(
        self,
        key: jax.Array,
        ts: DreamerState,
        carry: DreamerCarry,
        env: Environment,
        length: int,
        info_callback: Optional[Callable[[Dict[str, Any], jax.Array, int], None]] = None,
    ):
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

            transition = Transition(obs=obs, action=action, log_prob=log_prob, reward=reward, term=done, reset=carry.last_term)
            ts = ts.replace(global_step=ts.global_step + self.config.num_worlds)
            carry_new = DreamerCarry(env_state=env_state, last_obs=obs_new, last_deter=deter_new, last_term=done)

            if info_callback is not None:
                jax.debug.callback(info_callback, info, done, ts.global_step)

            return (key, ts, carry_new), transition

        return jax.lax.scan(_collect_rollout_step, (key, ts, carry), length=length)

    def observe(self, key: jax.Array, ts: DreamerState, batch: Transition, init_deter: jax.Array):
        def _observe_step(state: Tuple[jax.Array, jax.Array], transition: Transition):
            key, deter_cur = state
            embed = self.models.encoder.apply(ts.params.encoder, transition.obs)
            stoch_posterior_dist = self.models.posterior.postprocess(self.models.posterior.apply(ts.params.posterior, deter_cur, embed))
            key, sample_key = jax.random.split(key)
            stoch_posterior_probs = stoch_posterior_dist.probs
            stoch_posterior = stoch_posterior_dist.sample(seed=sample_key) + stoch_posterior_probs - jax.lax.stop_gradient(stoch_posterior_probs)
            deter_new = self.models.dynamics.apply(ts.params.dynamics, deter_cur, stoch_posterior, transition.action)
            deter_new = jnp.where(transition.term[:, None], init_deter, deter_new)
            return (key, deter_new), (deter_cur, stoch_posterior, stoch_posterior_probs)

        _, (deter, stoch, stoch_probs) = jax.lax.scan(_observe_step, (key, init_deter), batch)
        return deter, stoch, stoch_probs

    def imagine(self): ...

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
        print(self.models.obs_decoder.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.reward_predictor.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.cont_predictor.tabulate(tab_key, deter, stoch, compute_flops=True, compute_vjp_flops=True))
        print(self.models.dynamics.tabulate(tab_key, deter, stoch, act, compute_flops=True, compute_vjp_flops=True))
