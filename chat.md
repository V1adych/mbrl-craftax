# DreamerV3 re-implementation analysis

---

**User**

Hello! I just finished up dreamer implementation. here:
@src/agents/dreamerv3/agent.py:1-430 , you can also see entrypoint @src/main.py:1-21 and helper files for models @src/agents/dreamerv3/models.py:1-238 , replay buffer @src/agents/dreamerv3/replay_buffer.py:1-115 and other utilities @src/agents/dreamerv3/utils.py:1-27 @src/agents/dreamerv3/norm.py:1-29 more importantly, config: @src/configs/run.yaml:1-101 (it is now in debug mode by default, for real training, I am gonna change a lot of values, but still check how much adequate are they)
now I thing I only need to plug in logging, pick proper hyperparams and we are free to go, but I still need sanity-check

I want you to take as much time as required to deeply analyze my code and compare it against original dreamer implementation: @dreamerv3/dreamerv3/agent.py:1-491 @dreamerv3/dreamerv3/main.py:1-277 @dreamerv3/dreamerv3/rssm.py:1-360 @dreamerv3/dreamerv3/configs.yaml:1-221 
and check if my re-implementation is consistent with original code, and does not have any obvious mistakes. do not try to analyze algorighm correctness: it runs perfectly. I am interested more in real differences from original code and whether they carry some fundamental importance or not. for example: original paper uses weird definitions for reward and value, r_t in ther notation means reward of transition s_{t-1} \to s_t, and v_t means discounted sum of rewards starting from r_{t+1} - next-step reward, which introduces difference in return estimation and advantage estimation. I use classic text-book notations, my value estimates returns starting from current step. I tried to pick necessary shifts and other tricks to match proper return calculation anyway. not only that, but also how they syncronize updates and rollout collections: they spawn multiple workers for performance, and use train-to-update (or something like that) ratio to determine period and number of updates. I have jax-compatible environment, so I collect everything into a single xla graph. hope no major discrepancies. train-to-update ratio is very important, I want you to look at my configs and tell me, are my numbers adequate to keep proper rollout collection, WM update and AC update fractions. I think, you understand the stuff, and can start analysis. or ask some questions, if you want to know more. you can ask both before starting diving in, if you want to clarify something about the task, or after, to clarify some aspects of my/original codebases. for now, let's go like this: you scan through the code, identify all potential spots to analyze (e.g rollout collection, loss functions, update logic, etc) as  a list, with possible subcomponents (like do I have a matching logic for calculating hidden states for rollouts, then how properly do I calculate WM loss, gradient flow examination, etc.) (if any). then, we will both think about the analysis plan, and you will dive into the codebases, identifying points of interest

---

**Cursor**

## Spots to analyze (and what to compare against)

### 1) **Time/indexing conventions (the “what is \(t\)” problem)**
- **Your `Transition` semantics**: `obs[t]` (pre-action), `action[t]`, `reward[t]`, `term[t]` (done *after* `action[t]`), `reset[t]` (first obs of episode) in [`src/agents/dreamerv3/replay_buffer.py`](src/agents/dreamerv3/replay_buffer.py) and how you use them in `observe()`/returns in [`src/agents/dreamerv3/agent.py`](src/agents/dreamerv3/agent.py).
- **Return target indexing**: how your `_compute_lambda_returns()` aligns `reward`, `cont`, `next_values`, `last_value` vs original `lambda_return()` in [`dreamerv3/dreamerv3/agent.py`](dreamerv3/dreamerv3/agent.py) (original uses the “shifted” convention heavily).
- **Shape checkpoints** (so we can catch silent off-by-ones):
  - replay sample: `minibatch.obs` \((L,B,...) = (replay_length+batch_length,\; batch_size,\; ...)\)
  - after burn-in: \((batch_length,B,...)\)
  - `deter`: \((L,B,D)\), `stoch`: \((L,B,S,C)\)

### 2) **Rollout collection + env reset semantics**
- **Auto-reset interaction** (`auto_reset: true`): whether `obs_new` returned on terminal steps is already a reset observation, and whether your `reset` flag matches that (you store `reset=carry.last_term`).
- **State reset policy**: you reset `deter_new` with `done`, but you don’t explicitly reset anything else; check against original `reset = obs['is_first']` masking in RSSM.
- **RNG usage**: in `collect_rollouts()` you sample posterior and action using the same `sample_key` (correlation difference vs original).

### 3) **Replay buffer correctness for recurrent training**
- **Episode boundary handling**: you mark boundaries via `reset` from env + ring-buffer boundary patches (`boundary_first`, `boundary_term`). Verify no sampled sequence crosses an episode boundary without a reset/term being set.
- **Sampling legality early in training**: `max_start = ptr - (length - 1)` when not filled; confirm it never goes \(\le 0\) given your prefill logic and chosen `rollout_length`, `sample_length`.
- **Shape plumbing**: `sample()` returns \((L,N,...)\) then you reshape to \((G,T,B,...)\) with `rearrange` in `fit()`; verify `N == num_grad_steps * batch_size`.

### 4) **Encoder/decoder preprocessing + architecture parity**
- **Pixel scaling**: you do `obs - 0.5` assuming obs already in \([0,1]\); original codebase typically uses `uint8 / 255 - 0.5`. Confirm Craftax pixels dtype/range.
- **Conv stride/padding geometry**: you use multiple `Conv(stride=2, padding="VALID")` / transpose conv with `VALID`; check output sizes match input exactly (or whether you’re relying on implicit cropping).
- **Loss reduction**: you do pixelwise MSE summed over \((H,W,C)\) then mean; compare to original aggregation.

### 5) **RSSM/dynamics + latent distribution details**
- **Core update equivalence**: your `Dynamics` block-GRU vs original `RSSM._core()` (gating, block layout, normalization, activation).
- **Latent distribution**: you implement categorical one-hot with uniform mixing:
  - original: `outs.OneHot(..., unimix)` + `outs.Agg(..., sum)` (also determines KL reduction across stochastic dims)
  - yours: `distrax.OneHotCategorical(probs=...)`
- **Reparameterization / straight-through**:
  - you do straight-through *only* for posterior samples in `observe()`
  - check whether original effectively uses straight-through for *both* posterior/prior samples (important for gradients through imagined trajectories).
- **KL computation + reduction**:
  - you compute KL per-stochastic-dimension and then `.mean()` (so you’re effectively averaging over `stoch_size`)
  - original typically aggregates (often sums) across stochastic dims via `outs.Agg`.

### 6) **Reward/value “two-hot symlog” implementation**
- **Bin definition** (critical parity point): your heads set
  - `bins = linspace(symexp(min_val), symexp(max_val), num_bins)`
  - but your `two_hot_symlog()` digitizes **`symlog(x)` against `bins`**, which suggests `bins` should live in **symlog-space** (like original `symexp_twohot` heads).
- **Postprocess**: you do `symexp(sum(softmax(logits) * bins))`; this also assumes `bins` are in symlog-space.
- **Compare against original** `MLPHead(..., output=symexp_twohot, bins=255)` behavior.

### 7) **Continuation (discount) modeling**
- **Target definition**: you use `cont_target = 1 - term`; original optionally uses `contdisc` and multiplies by \((1 - 1/horizon)\).
- **Where cont is used**:
  - in return computation (lambda return)
  - in weighting imagined losses by cumulative product of discounts (original `weight = cumprod(disc * con, ...)`).

### 8) **Imagination rollout alignment**
- **What your `ImagTransition.reward/cont` correspond to**: you predict them from `(deter_new, stoch_new)` but store them alongside `(deter_cur, stoch_cur, action)`. That’s a legitimate “reward-on-transition” convention, but it must match your return code shifts.
- **Horizon shapes**: you roll `length = imag_horizon + 1`; confirm which indices are used for:
  - actor loss: \((H,B)\) terms
  - critic targets: bootstrap at last state

### 9) **Actor/critic losses vs original imag_loss / repl_loss**
- **Weighting**: original weights losses by `weight[t] = Π_{i≤t} disc*con`; your actor/critic losses currently look *unweighted* (pure mean), so this is a big behavioral difference to check explicitly.
- **Advantage normalization**: original has `retnorm`, `valnorm`, `advnorm`; you only have `RetNorm` on advantages (and only scale, no centering).
- **Slow critic usage**: original has `SlowModel` with `rate` and `every`; you do EMA with `critic_tau`, but update it once per outer loop (after `num_grad_steps`).

### 10) **Update scheduling + train-to-update ratio mapping**
This is where we’ll directly answer “are my numbers adequate?” by computing an *equivalent* train ratio.

- **Per outer iteration in your code**:
  - env steps collected: \(N_{env} = num\_worlds \times rollout\_length\)
  - gradient steps: \(G = num\_grad\_steps\)
  - effective time steps contributing to loss per grad step: \(N_{loss} = batch\_size \times batch\_length\) (after dropping `replay_length`)
- **A good first-order proxy**:
  \[
  train\_ratio\_{\text{proxy}} \approx \frac{G \cdot batch\_size \cdot batch\_length}{num\_worlds \cdot rollout\_length}
  \]
- **Compare** that proxy to original `run.train_ratio` (often 32+), and note that original’s ratio is enforced asynchronously while yours is “lockstep” inside one XLA program (distributional difference even if the scalar matches).

### 11) **Optimizer + parameter update mechanics**
- **Masked weight decay correctness**: your `_is_params` mask checks for `"params"` DictKey; verify it matches your actual param tree layout.
- **Optimizer choice mismatch**: original uses a very specific RMS+momentum style optimizer with warmup schedules; you’re using `optax.contrib.muon` with a very large LR in debug config—worth sanity-checking relative scaling expectations.

### 12) **Numerical/stability invariants**
- **KL uses `log(probs)`**: ensure `uniform_frac` truly prevents zeros everywhere (posterior and prior).
- **Quantile in `RetNorm.update()`** inside JIT: confirm it’s stable and not dominating compile/runtime (even if “it runs”).

---

## High-impact “likely important” diffs to prioritize when we dive in
- **Two-hot symlog bins**: bin construction vs `two_hot_symlog()` expectations (parity with original `symexp_twohot`).
- **Discount/weighting in actor+critic losses**: original uses cumulative discount weights; your losses look unweighted.
- **Straight-through in imagination**: posterior uses ST, prior sample in imagine does not (potential gradient-path mismatch vs original).
- **KL reduction scaling**: mean vs sum across stochastic dimensions (affects effective loss scale and free-nats behavior).
- **Slow critic update cadence**: once-per-outer-loop vs every gradient step (rate mismatch).
- **RNG reuse**: same key for posterior and action in rollouts.

If this spot list looks good, next step is I’ll go through these in order, and for each one: (1) write down the exact tensor shapes at each line, (2) map it to the corresponding original function (`Agent.loss`, `imag_loss`, `lambda_return`, `RSSM.loss/_core`), and (3) classify each difference as “cosmetic”, “scale-only”, or “fundamental behavioral change”.

---

**User**

yes, that's what I was talking about! you already found a bug in rng, which I think I fixed: @src/agents/dreamerv3/agent.py:340-347, and raised a lot of important points, like details for loss functions  (are they just CE for categorical reparametrizations), when and how exactly unimix and straight-through gradients should be used, and many more! regarding symlog: I tried to properly follow dreamer original implementation and common practices to interpolate and two-hot in symlog space (biased, but still commonly adopted practice), but have it checked anyway. I have nothing to add from my side. you can calmly proceed with your analysis, if you don't have any questions

---

**Cursor**

Got it — I’ll now do the deep diff pass against the reference DreamerV3 code, focusing on **implementation mismatches that matter** (indexing/returns, KL/unimix/straight-through, loss weighting, update schedule/train_ratio, replay boundary handling, etc.). I’ll start by pulling the exact file contents with real line numbers so I can cite precisely.

I’ve loaded both implementations with line-accurate references. Next I’ll go subsystem-by-subsystem (obs preprocessing → latent dist/KL → return indexing → actor/critic loss weighting → update scheduling/train_ratio), and I’ll flag each mismatch as “scale-only” vs “behavioral”.

I’m starting with the “input/output contract” checks (obs dtype/range and decoder geometry), because mismatches here can silently change loss scales and stability even if everything compiles and runs.

Quick snag: this shell image doesn’t have `python` on PATH; I’ll retry with `python3` to directly inspect Craftax’s observation dtype/range so we can validate your `obs - 0.5` preprocessing.

### Quick findings so far (high-signal diffs)

- **Obs preprocessing is correct for Craftax**: `Craftax-Classic-Pixels-v1` returns **`float32` in `[0,1]` with shape `(63,63,3)`**, so your encoder’s `obs - 0.5` is consistent. Also, your `VALID` stride-2 conv stack exactly maps `63→31→15→7→3`, so the decoder’s `3×3` starting grid is geometrically consistent.
- **Minor but real model diff**: your `ObsDecoder` returns raw pixels (no sigmoid), while reference DreamerV3’s decoder outputs `sigmoid`ed pixels; this mostly affects **output range + loss scaling**, not “fundamental logic”.
- **Potentially important latent KL scaling mismatch**: reference RSSM uses `outs.Agg(..., jnp.sum)` to **sum KL over the `stoch` dimension** before applying `free_nats`; your `kl_divergence()` returns KL per-stochastic-variable and then you do `.mean()` after `max(free_nats)`. That changes both **KL magnitude** and **what `free_nats=1.0` means**, so your effective `dyn/rep` regularization strength can be very different even with the same weights.

Next I’ll go into the big one: **reward/value time-indexing + your rollout/imagination λ-return alignment**, and then the **actor/critic loss weighting vs DreamerV3’s `weight=cumprod(discount*cont)`**.

---

**User**

great progress so far! you identified one more bug! here is a fix: @src/agents/dreamerv3/agent.py:274-275 addes .sum(axis=-1) (over number of stoch classes) to get correct loss scale. I also beleive that previous bug had no issue with free nats logic, only with scale, because free_nats were applied to raw per-class kl values. regarding sigmoid in decoder: do they still use MSE loss with it? anyways, let's write all issues down here @issues.md:1-2 , feel free to extend this file

great, already a lot of findings! let's extend our convention for file @issues.md : for every element let's assign a tag. let's define [BUG] all issues, like what you found with inf overflow for symexp(1000), [DIFF] for non-fundamental discrepances from original implementation, like sigmoid for outputs of decoder, [CRITDIFF] - critical fundamental differences, such as weighting AC targets @issues.md:17 , and [MINOR] things that exist in their repo as optional configurable behaviours, but disabled by default and not mentioned in paper, like this: @issues.md:18-19 (pls check if they are indeed disabled by default), [FIXED] for bugs/diffs, that were already fixed: @issues.md:3-8 @src/agents/dreamerv3/models.py:104-106 , [TOCHECK] something like this: @issues.md:26 (do not bother yourself to debug this complicated stuff, just bring it to my attention), [CONFIG] something like this: @issues.md:28-32 

now you can go on

---
