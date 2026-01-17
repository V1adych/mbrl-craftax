### Modeling
1) [DIFF] original DreamerV3 decoder uses sigmoid (ours currently returns raw pixels/logits)
NOTE: reference still uses an MSE-style reconstruction loss after sigmoid (pixel targets are scaled to [0,1]).
2) [DIFF] encoder/decoder architecture: reference uses conv + (often) max-pool style downsampling and upsampling-by-repeat+conv; ours uses strided `VALID` convs and `ConvTranspose`. Not “wrong”, but it changes inductive bias and loss scale/details.
3) [FIXED] reward/value two-hot symlog bins: current bins use `linspace(symexp(min_val), symexp(max_val), ...)`; with `min_val=-1000,max_val=1000` this overflows to ±inf and `linspace` produces NaNs
this was a bug: configs were designed to contain min/max values in original space, so taking symexp is wrong: we need to take symlog. fix (already in models.py):
```python
    def setup(self):
        self.bins = self.variable("state", "bins", lambda: jnp.linspace(symlog(self.config.min_val), symlog(self.config.max_val), self.config.num_bins))
```

4) [FIXED] latent KL scale: sum KL over stochastic variables (stoch_size); fixed by `.sum(axis=-1)` after `kl_divergence(...)`
5) [FIXED] free_nats semantics: code now applies `free_nats` after summing KL over stochastic variables (matches reference behavior)
6) [FIXED] unimix/uniform mixing naming: reference uses RSSM `unimix` inside the OneHot distribution; ours uses `uniform_frac` in Posterior/Prior (behavior likely equivalent if values match)
7) [TOCHECK] straight-through + unimix details: posterior uses ST in `observe()`; imagined prior samples are plain samples. With `stop_gradient(imagine(...))` this mostly matters only if/when you enable AC gradients through imagination; reference uses `outs.OneHot` (not vendored here) which may differ in ST details.
8) [IMPORTANT_NOTE] two reward/termination conventions coexist (may look like “misalignment” but can be consistent):
   - `reward_prev/reset` are “arrival” targets: at time t, given latent for state \(s_t\), predict reward/cont from transition \(s_{t-1}\to s_t\).
   - `reward/term` are “leaving” targets: at time t, reward/term from transition \(s_t\to s_{t+1}\) (used for rollout λ-returns).
   Imagination uses reward/cont predicted from the *next* imagined state, so it still yields the “leaving” reward for the imagined transition.

### Actor-Critic / returns
1) [FIXED] imagined losses weighting: reference multiplies policy/value losses by `weight = cumprod(disc * con)`; ours currently uses unweighted means
FIXED: actor/critic losses are weighted by cumprod(disc * con) for imaginary rollouts (matches reference high-level intent).
2) [DIFF] continuation/discounting parameterization: reference (default `contdisc: True`) trains `con_target = (~is_terminal) * (1 - 1/horizon)` and then uses `weight = cumprod(con)` (with `disc=1`). Our code trains `cont_target = 1 - reset` (shifted arrival convention) and uses `cumprod(gamma * cont) / gamma`. This is equivalent only if `gamma == 1 - 1/horizon` and `cont` predicts `~terminal`; current `src/configs/run.yaml` uses `gamma: 0.99`, so it is a real config/behavior difference from the reference defaults.
3) [FIXED] λ-returns: `jax.lax.scan(reverse=True)` iterates in reverse but returns outputs in the original time order (your `test.py` confirms). Current `compute_lambda_returns()` bootstraps from `last_value` and produces a length-`T` return sequence aligned with `rewards/conts`.
Current implementation in `src/agents/dreamerv3/utils.py`:
```python
def compute_lambda_returns(rewards: jax.Array, conts: jax.Array, values: jax.Array, last_value: jax.Array, gamma: float, lam: float):
    def _lambda_return_step(next_ret: jax.Array, rew_cont_nval: Tuple[jax.Array, jax.Array, jax.Array]):
        reward, cont, next_value = rew_cont_nval
        ret = reward + cont * gamma * ((1 - lam) * next_value + lam * next_ret)
        return ret, ret

    _, returns = jax.lax.scan(
        _lambda_return_step,
        last_value,
        (rewards, conts, jnp.concatenate([values[1:], last_value[None]], axis=0)),
        reverse=True,
    )
    return returns
```
4) [MINOR] normalization options: reference has `valnorm`/`advnorm` but both are disabled by default (`impl: none`); we don't implement them (we do have our own `RetNorm`)
5) [DIFF] slow target update cadence: reference updates `slowvalue` every train step with `rate`; ours updates `slow_critic_params` once per outer update loop
6) [MINOR] imagination gradients option: reference supports `ac_grads` but default is `False`; ours currently `stop_gradient(imagine(...))` unconditionally
7) [FIXED] policy unimix: reference policy has `unimix` (default 0.01) for discrete actions; ours uses plain `distrax.Categorical(logits=...)` without mixing
FIXED: added unimix to policy
8) [DIFF] slow critic usage for targets/baseline: reference default (`imag_loss.slowtar: False`, `repl_loss.slowtar: False`) uses the *current* value head for return targets/baseline, and uses slow value mostly as a regularizer (via `slowreg`). Our code uses `slow_critic_params` for rollout+imag return targets and for the baseline in advantages, and does not include an explicit `slowreg * value.loss(slowvalue.pred())` regularizer term.
   - [DIFF] missing slow critic regularizer term: reference value loss includes `slowreg * value.loss(slowvalue.pred())`; we currently don't have an analog.
   - [DIFF] baseline/targets source differs: reference defaults use current value head for targets/baseline (`slowtar: False`), while we use slow critic for both.
   - [DIFF] rollout bootstrap differs: for replay λ-returns we bootstrap `last_value` from an *imagined* 1-step successor of the last replay state (policy action), not from the real next state under the recorded last action (and not from `V(last replay state)` either).
9) [FIXED] rollout critic weighting: `loss_critic_rollout` now uses a simple mask weight `ac_rollout_weight = cont = float32(~minibatch.term)` (instead of a cumulative product).
10) [CRITDIFF] value learning structure differs from reference:
   - reference trains value on imagination (`imag_loss`) and optionally a separate replay value loss (`repl_loss`, `repval_loss: True` by default) that bootstraps from imagination returns.
   - our implementation trains value both on imagination (`loss_critic_imag`) and also directly on real rollout λ-returns (`loss_critic_rollout`) computed from replayed rewards/cont + slow critic bootstrap.
   This changes the training signal mixture and is more than a pure “scale-only” diff.
11) [MINOR] shape brittleness in rollout bootstrap: `values_rollout_last = rearrange(..., t=self.config.imag_last_states, b=self.config.batch_size)` assumes `minibatch.obs.shape[1] == self.config.batch_size`.

### Data / replay / env
1) [FIXED] auto-reset semantics: ensure `reset` flag matches what the env returns after `done` when `auto_reset=true`
of course it does, it is crafrtax, after all
2) [FIXED] transition alignment vs reference: we store `obs_t` (pre-step) together with `reward_t`/`term_t` from the step `s_t --a_t--> s_{t+1}`; reference aligns `reward/terminal` with the *arrival* state (`obs_{t+1}`), which affects reward/cont training and indexing
FIXED: Code refactored to store arrival rewards in transitions as reward_prev
3) [FIXED] replay sampling contiguity across wrap: sampling is now done in a *logical time axis* relative to `ptr`.
   - when not filled: sample `start ∈ [0, ptr - (length - 1))` so indices stay within written prefix.
   - when filled: sample `start ∈ [ptr, ptr + (max_length - (length - 1)))`, then modulo by `max_length`, which is equivalent to sampling contiguous sequences in chronological order without crossing the `ptr` boundary in time.
4) [DIFF] `reset`/`term` semantics naming vs reference: reference uses explicit `is_first` and `is_terminal` fields; our `Transition.reset` is “previous step done” (used to reset RSSM state) and `Transition.term` is “current step done” (used for rollout returns/weights). Naming mismatch is fine as long as all shifts are consistent.

### Update schedule / train ratio
1) [CONFIG] current config effective train ratio proxy:
   - env steps per outer iter: `num_worlds * rollout_length = 8 * 128 = 1024`
   - gradient samples per outer iter: `num_grad_steps * batch_size * batch_length = 4 * 16 * 64 = 4096`
   - proxy ratio ≈ 4 (reference commonly uses 32+)
2) [CONFIG] imagination starts count: reference default `imag_last: 0` => uses all `T=batch_length` states as start states; our config uses `imag_last_states: 16`, which reduces the number of imagined trajectories per update.
3) [CONFIG] optimizer/LR differ from reference defaults: reference uses RMS+momentum style optimizer with `lr~4e-5` (plus warmup), while our config currently uses `optax.contrib.muon` with `lr: 2e-2` (debug-scale).
4) [DIFF] train-ratio enforcement differs: reference enforces `run.train_ratio` in an asynchronous actor/replay/trainer setup; our loop is lockstep “collect rollout → do exactly `num_grad_steps` updates”. Even if the scalar ratio matches, the distribution of data ages and update timing differs.

### Optimization
1) [DIFF] optimizer + clipping differ: reference uses AGC + RMS + momentum + LR schedule/warmup; we use global-norm clipping + `optax.contrib.muon` (+ weight decay mask). This is likely not “fundamental algorithm logic”, but it can materially change stability/speed.

### RNG
1) [FIXED] rollout sampling: ensure posterior sample key and action sample key are different