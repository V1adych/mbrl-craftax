from __future__ import annotations
from typing import Any, Tuple, Dict, Optional, Callable
import operator
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
from .norm import RetNorm
from .replay_buffer import Transition, ReplayBuffer, ReplayBufferState
from .utils import kl_divergence, compute_lambda_returns


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
class Params:
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
    last_reward: jax.Array
    last_term: jax.Array


@struct.dataclass
class ImagTransition:
    deter: jax.Array
    stoch: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    cont: jax.Array


@struct.dataclass
class Losses:
    obs: jax.Array
    reward: jax.Array
    cont: jax.Array
    dyn: jax.Array
    rep: jax.Array
    actor: jax.Array
    critic_rollout: jax.Array
    critic_imag: jax.Array
    entropy: jax.Array


class DreamerState(train_state.TrainState):
    global_step: int
    slow_critic_params: Any
    ret_norm_params: Any

    @classmethod
    def create(cls, *, apply_fn: Optional[Callable] = None, params: Any, tx: optax.GradientTransformation, **kwargs) -> DreamerState:
        opt_state = tx.init(params)
        return cls(step=0, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state, **kwargs)

    def apply_gradients(self, *, grads: Any, **kwargs) -> DreamerState:
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)


class DreamerV3:
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: Models = None
        self.replay: ReplayBuffer = None
        self.ret_norm = RetNorm(self.config.ret_norm)

        self.loss_weights: Losses = Losses(
            obs=jnp.float32(self.config.loss.dec_obs),
            reward=jnp.float32(self.config.loss.dec_reward),
            cont=jnp.float32(self.config.loss.dec_cont),
            dyn=jnp.float32(self.config.loss.dyn),
            rep=jnp.float32(self.config.loss.rep),
            actor=jnp.float32(self.config.loss.actor),
            critic_rollout=jnp.float32(self.config.loss.critic_rollout),
            critic_imag=jnp.float32(self.config.loss.critic_imag),
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
        stoch_dist = self.models.posterior.apply(params_posterior, deter, embed, method=self.models.posterior.predict)
        stoch = stoch_dist.sample(seed=k_sample)

        params_prior = self.models.prior.init(k_prior, deter)

        params_actor = self.models.actor.init(k_act, deter, stoch)
        params_critic = self.models.critic.init(k_crit, deter, stoch)
        params_obs_decoder = self.models.obs_decoder.init(k_obs_dec, deter, stoch)
        params_reward_predictor = self.models.reward_predictor.init(k_rew, deter, stoch)
        params_cont_predictor = self.models.cont_predictor.init(k_cont, deter, stoch)

        act_dist = self.models.actor.apply(params_actor, deter, stoch, method=self.models.actor.predict)
        act = act_dist.sample(seed=k_sample)

        params_dynamics = self.models.dynamics.init(k_dyn, deter, stoch, act)
        params_ret_norm = self.ret_norm.init(jax.random.key(0), jnp.zeros((1,), dtype=jnp.float32))

        return (
            Params(
                encoder=params_encoder,
                dynamics=params_dynamics,
                obs_decoder=params_obs_decoder,
                reward_predictor=params_reward_predictor,
                cont_predictor=params_cont_predictor,
                posterior=params_posterior,
                prior=params_prior,
                actor=params_actor,
                critic=params_critic,
            ),
            params_critic,
            params_ret_norm,
        )

    def fit(self, key: jax.Array, env: Environment):
        reset_fn = jax.vmap(env.reset)
        key, *reset_keys = jax.random.split(key, self.config.num_worlds + 1)
        obs, env_state = reset_fn(jnp.array(reset_keys))

        key, init_key = jax.random.split(key)
        variables, slow_critic_params, ret_norm_params = self._init_models(init_key, env, env_state)
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
        init_reward = jnp.zeros((self.config.num_worlds,), dtype=jnp.float32)
        init_term = jnp.ones((self.config.num_worlds,), dtype=jnp.bool)
        ts = DreamerState.create(params=variables, tx=tx, global_step=0, slow_critic_params=slow_critic_params, ret_norm_params=ret_norm_params)
        carry = DreamerCarry(env_state=env_state, last_obs=obs, last_deter=init_deter, last_reward=init_reward, last_term=init_term)

        obs_shape = env.observation_space(env_state).shape
        dummy = Transition(
            obs=jnp.zeros((0, self.config.num_worlds, *obs_shape), dtype=obs.dtype),
            action=jnp.zeros((0, self.config.num_worlds), dtype=jnp.int32),
            reward_prev=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            reward=jnp.zeros((0, self.config.num_worlds), dtype=jnp.float32),
            term=jnp.ones((0, self.config.num_worlds), dtype=jnp.bool),
            reset=jnp.zeros((0, self.config.num_worlds), dtype=jnp.bool),
        )
        self.replay = ReplayBuffer(self.config.replay_buffer)
        replay_state: ReplayBufferState = self.replay.init(dummy)

        num_samples = self.config.batch_size * self.config.num_grad_steps
        sample_length = self.config.replay_length + self.config.batch_length

        def _update_loop(state: Tuple[jax.Array, DreamerState, DreamerCarry, ReplayBufferState], _):
            key, ts, carry, replay_state = state
            (key, ts, carry), rollout = self.collect_rollouts(key, ts, carry, env, self.config.rollout_length)

            replay_state = self.replay.add(replay_state, rollout)

            key, sample_key = jax.random.split(key)
            batch = self.replay.sample(replay_state, sample_key, sample_length, num_samples)
            batch = jax.tree.map(lambda x: rearrange(x, "t (g b) ... -> g t b ...", g=self.config.num_grad_steps, b=self.config.batch_size), batch)

            def _update(carry: Tuple[jax.Array, DreamerState], minibatch: Transition):
                key, ts = carry
                key, update_key = jax.random.split(key)
                ts = self.update(update_key, ts, minibatch)
                return (key, ts), None

            (key, ts), metrics = jax.lax.scan(_update, (key, ts), batch)
            ts = ts.replace(slow_critic_params=self._update_slow_critic(ts.slow_critic_params, ts.params.critic))

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

        ts = _fit(key, ts, carry, replay_state)

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
            stoch_posterior_dist = self.models.posterior.apply(ts.params.posterior, deter, embed, method=self.models.posterior.predict)
            key, stoch_key, act_key, step_key = jax.random.split(key, 4)
            stoch_posterior = stoch_posterior_dist.sample(seed=stoch_key)
            policy = self.models.actor.apply(ts.params.actor, deter, stoch_posterior, method=self.models.actor.predict)
            action = policy.sample(seed=act_key)
            obs_new, env_state, reward, done, info = jax.vmap(env.step)(jax.random.split(step_key, self.config.num_worlds), carry.env_state, action)
            deter_new = self.models.dynamics.apply(ts.params.dynamics, deter, stoch_posterior, action)
            deter_new = jnp.where(done[:, None], self.models.dynamics.get_initial_deter(self.config.num_worlds), deter_new)

            transition = Transition(obs=obs, action=action, reward_prev=carry.last_reward, reward=reward, term=done, reset=carry.last_term)
            ts = ts.replace(global_step=ts.global_step + self.config.num_worlds)
            carry_new = DreamerCarry(env_state=env_state, last_obs=obs_new, last_deter=deter_new, last_reward=reward, last_term=done)

            if info_callback is not None:
                jax.debug.callback(info_callback, info, done, ts.global_step)

            return (key, ts, carry_new), transition

        return jax.lax.scan(_collect_rollout_step, (key, ts, carry), length=length)

    def observe(self, key: jax.Array, params: Params, batch: Transition, init_deter: jax.Array):
        reset_deter = self.models.dynamics.get_initial_deter(batch.obs.shape[1])

        def _observe_step(state: Tuple[jax.Array, jax.Array], transition: Transition):
            key, deter_cur = state
            deter_cur_masked = jnp.where(transition.reset[:, None], reset_deter, deter_cur)
            embed = self.models.encoder.apply(params.encoder, transition.obs)
            stoch_posterior_dist = self.models.posterior.apply(params.posterior, deter_cur_masked, embed, method=self.models.posterior.predict)
            key, sample_key = jax.random.split(key)
            stoch_posterior_probs = stoch_posterior_dist.probs
            stoch_posterior = stoch_posterior_dist.sample(seed=sample_key) + stoch_posterior_probs - jax.lax.stop_gradient(stoch_posterior_probs)
            deter_new = self.models.dynamics.apply(params.dynamics, deter_cur_masked, stoch_posterior, transition.action)
            return (key, deter_new), (deter_cur, stoch_posterior, stoch_posterior_probs)

        _, (deter, stoch, stoch_probs) = jax.lax.scan(_observe_step, (key, init_deter), batch)
        return deter, stoch, stoch_probs

    def imagine(self, key: jax.Array, params: Params, init: Tuple[jax.Array, jax.Array], length: int):
        deter, stoch = init

        def _imagine_step(state: Tuple[jax.Array, jax.Array, jax.Array], _):
            key, deter_cur, stoch_cur = state
            policy = self.models.actor.apply(params.actor, deter_cur, stoch_cur, method=self.models.actor.predict)
            key, policy_key, stoch_key = jax.random.split(key, 3)
            action = policy.sample(seed=policy_key)
            log_prob = policy.log_prob(action)
            deter_new = self.models.dynamics.apply(params.dynamics, deter_cur, stoch_cur, action)
            stoch_new = self.models.prior.apply(params.prior, deter_new, method=self.models.prior.predict).sample(seed=stoch_key)
            reward = self.models.reward_predictor.apply(params.reward_predictor, deter_new, stoch_new, method=self.models.reward_predictor.predict)
            cont = self.models.cont_predictor.apply(params.cont_predictor, deter_new, stoch_new, method=self.models.cont_predictor.predict)
            transition = ImagTransition(deter=deter_cur, stoch=stoch_cur, action=action, log_prob=log_prob, reward=reward, cont=cont)
            return (key, deter_new, stoch_new), transition

        (_, deter_last, stoch_last), rollout = jax.lax.scan(_imagine_step, (key, deter, stoch), length=length)
        return (deter_last, stoch_last), rollout

    def update(self, key: jax.Array, ts: DreamerState, minibatch: Transition):
        key, observe_key = jax.random.split(key)

        def _loss_fn(params: Params, minibatch: Transition, slow_critic_params: Any, ret_norm_params: Any):
            batch_size = minibatch.obs.shape[1]
            gamma = self.config.gamma
            lam = self.config.lam

            init_deter = self.models.dynamics.get_initial_deter(batch_size)
            deter, stoch, stoch_posterior_probs = self.observe(observe_key, params, minibatch, init_deter)
            deter, stoch, stoch_posterior_probs, minibatch = jax.tree.map(
                lambda x: x[self.config.replay_length :], (deter, stoch, stoch_posterior_probs, minibatch)
            )

            obs_pred = self.models.obs_decoder.apply(params.obs_decoder, deter, stoch)
            loss_obs = self.models.obs_decoder.apply(params.obs_decoder, obs_pred, minibatch.obs, method=self.models.obs_decoder.loss)

            reward_symlog = self.models.reward_predictor.apply(params.reward_predictor, deter, stoch)
            loss_reward = self.models.reward_predictor.apply(
                params.reward_predictor, reward_symlog, minibatch.reward_prev, method=self.models.reward_predictor.loss
            )

            cont_logits = self.models.cont_predictor.apply(params.cont_predictor, deter, stoch)
            cont_target = 1.0 - minibatch.reset.astype(jnp.float32)
            loss_cont = self.models.cont_predictor.apply(params.cont_predictor, cont_logits, cont_target, method=self.models.cont_predictor.loss)

            stoch_prior_probs = self.models.prior.apply(params.prior, deter, method=self.models.prior.predict).probs
            kl_dyn = kl_divergence(jax.lax.stop_gradient(stoch_posterior_probs), stoch_prior_probs).sum(axis=-1)
            kl_rep = kl_divergence(stoch_posterior_probs, jax.lax.stop_gradient(stoch_prior_probs)).sum(axis=-1)
            kl_clipfrac = jnp.mean(jnp.float32(kl_dyn < self.config.loss.free_nats))
            loss_dyn = jnp.maximum(kl_dyn, self.config.loss.free_nats).mean()
            loss_rep = jnp.maximum(kl_rep, self.config.loss.free_nats).mean()

            imag_init = jax.tree.map(lambda x: rearrange(x[-self.config.imag_last_states :], "t b ... -> (t b) ..."), (deter, stoch))

            (deter_last, stoch_last), imag_rollout = jax.lax.stop_gradient(self.imagine(key, params, imag_init, self.config.imag_horizon))

            values_rollout = self.models.critic.apply(slow_critic_params, deter, stoch, method=self.models.critic.predict)
            values_imag = self.models.critic.apply(slow_critic_params, imag_rollout.deter, imag_rollout.stoch, method=self.models.critic.predict)
            values_rollout_last = rearrange(values_imag[1], "(t b) ... -> t b ...", t=self.config.imag_last_states, b=self.config.batch_size)[-1]
            values_imag_last = self.models.critic.apply(slow_critic_params, deter_last, stoch_last, method=self.models.critic.predict)

            cont = ac_rollout_weight = jnp.float32(~minibatch.term)
            returns_rollout = jax.lax.stop_gradient(compute_lambda_returns(minibatch.reward, cont, values_rollout, values_rollout_last, gamma, lam))
            returns_imag = jax.lax.stop_gradient(
                compute_lambda_returns(imag_rollout.reward, imag_rollout.cont, values_imag, values_imag_last, self.config.gamma, self.config.lam)
            )
            ac_imag_weight = jnp.cumprod(imag_rollout.cont * gamma, axis=0) / gamma

            value_pred_rollout_symlog = self.models.critic.apply(params.critic, deter, stoch)
            loss_critic_rollout = self.models.critic.apply(
                params.critic, value_pred_rollout_symlog, returns_rollout, ac_rollout_weight, method=self.models.critic.loss
            )
            value_pred_imag_symlog = self.models.critic.apply(params.critic, imag_rollout.deter, imag_rollout.stoch)
            loss_critic_imag = self.models.critic.apply(params.critic, value_pred_imag_symlog, returns_imag, ac_imag_weight, method=self.models.critic.loss)

            ret_norm_params = self.ret_norm.apply(ret_norm_params, returns_imag, method=self.ret_norm.update, mutable=["state"])[1]
            policy = self.models.actor.apply(params.actor, imag_rollout.deter, imag_rollout.stoch, method=self.models.actor.predict)
            log_prob = policy.log_prob(imag_rollout.action)
            adv = jax.lax.stop_gradient(self.ret_norm.apply(ret_norm_params, returns_imag - values_imag))
            loss_actor = -jnp.mean(adv * log_prob * ac_imag_weight)
            loss_entropy = -jnp.mean(policy.entropy() * ac_imag_weight)

            losses = Losses(
                obs=loss_obs,
                reward=loss_reward,
                cont=loss_cont,
                dyn=loss_dyn,
                rep=loss_rep,
                actor=loss_actor,
                critic_imag=loss_critic_imag,
                critic_rollout=loss_critic_rollout,
                entropy=loss_entropy,
            )

            return jax.tree.reduce(operator.add, jax.tree.map(operator.mul, losses, self.loss_weights)), ret_norm_params

        (loss, ret_norm_params), grads = jax.value_and_grad(_loss_fn, has_aux=True)(ts.params, minibatch, ts.slow_critic_params, ts.ret_norm_params)
        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(ret_norm_params=ret_norm_params)

        return ts

    def _update_slow_critic(self, slow_critic_params: Any, critic_params: Any):
        critic_tau = self.config.critic_tau
        slow_critic_params = jax.tree.map(lambda x, y: critic_tau * x + (1 - critic_tau) * y, slow_critic_params, critic_params)
        return slow_critic_params

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
