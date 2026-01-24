import jax
from jax import numpy as jnp
from flax import linen
import distrax
from omegaconf import DictConfig
from .utils import symexp, symlog, two_hot_symlog


class BlockDense(linen.Module):
    features: int
    bias: bool = True
    kernel_init: linen.initializers.Initializer = linen.initializers.lecun_normal(batch_axis=0)
    bias_init: linen.initializers.Initializer = linen.initializers.zeros_init()

    @linen.compact
    def __call__(self, x: jax.Array):
        kernel = self.param("kernel", self.kernel_init, (*x.shape[-2:], self.features))
        x = jnp.einsum("... g i, g i o -> ... g o", x, kernel)
        if self.bias:
            bias = self.param("bias", self.bias_init, (x.shape[-2], self.features))
            x += bias.reshape((1,) * (x.ndim - 2) + bias.shape)
        return x


class Dynamics(linen.Module):
    config: DictConfig
    act_space: int

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array, act: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        hidden_size = self.config.hidden_size
        deter_size = deter.shape[-1]
        blocks = self.config.blocks
        assert deter_size % blocks == 0, f"Expected deter_size to be divisible by blocks, got deter_size={deter_size} and blocks={blocks}"
        block_size = deter_size // blocks

        deter_emb = jax.nn.silu(linen.RMSNorm(name="deter_norm")(linen.Dense(features=hidden_size, name="deter_emb")(deter)))
        stoch_emb = jax.nn.silu(linen.RMSNorm(name="stoch_norm")(linen.Dense(features=hidden_size, name="stoch_emb")(stoch)))
        act_emb = jax.nn.silu(linen.RMSNorm(name="act_norm")(linen.Embed(num_embeddings=self.act_space, features=hidden_size, name="act_emb")(act)))
        x = jnp.concat([deter_emb, stoch_emb, act_emb], axis=-1)

        x = x[..., None, :].repeat(blocks, axis=-2)
        deter_blocks = deter.reshape((*deter.shape[:-1], blocks, -1))
        x = jnp.concat([deter_blocks, x], axis=-1)

        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(BlockDense(features=block_size, name=f"block_dense_{i}")(x)))

        x = BlockDense(features=3 * block_size, name="gru_dense")(x)
        reset, cand, update = jnp.split(x, 3, axis=-1)

        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1.0)
        deter_new = jnp.reshape(update * cand + (1 - update) * deter_blocks, deter.shape)
        return deter_new

    def get_initial_deter(self, batch_size: int):
        return jnp.zeros((batch_size, self.config.deter_size), dtype=jnp.float32)


class Encoder(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, obs: jax.Array):
        x = obs.astype(jnp.float32) / 255.0 - 0.5
        for i, mult in enumerate(self.config.mults):
            x = linen.Conv(features=self.config.depth * mult, kernel_size=(3, 3), strides=(2, 2), padding="VALID", name=f"conv_{i}")(x)
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = jax.nn.silu(linen.RMSNorm(name="dense_norm")(linen.Dense(features=self.config.hidden_size, name="dense")(x)))
        return x


class ObsDecoder(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        features = jnp.concat([deter, stoch], axis=-1)

        x = jax.nn.silu(linen.RMSNorm(name="obs_norm")(linen.Dense(features=self.config.hidden_size, name="obs_dense")(features)))
        x = linen.Dense(features=self.config.depth * self.config.mults[-1] * 3 * 3, name="spatial_dense")(x)
        x = x.reshape((*x.shape[:-1], 3, 3, self.config.depth * self.config.mults[-1]))
        for i, mult in reversed(list(enumerate(self.config.mults[:-1]))):
            x = linen.ConvTranspose(features=self.config.depth * mult, kernel_size=(3, 3), strides=(2, 2), padding="VALID", name=f"obs_conv_{i}")(x)
            x = jax.nn.silu(linen.RMSNorm(name=f"obs_norm_{i}")(x))
        obs = linen.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2), padding="VALID", name="obs_final")(x)
        return obs

    def loss(self, pred: jax.Array, target: jax.Array):
        return 0.5 * jnp.square(pred - target).sum(axis=(-3, -2, -1)).mean()

    def predict(self, deter: jax.Array, stoch: jax.Array):
        return self(deter, stoch)


class RewardPredictor(linen.Module):
    config: DictConfig

    def setup(self):
        self.bins = self.variable("state", "bins", lambda: jnp.linspace(symlog(self.config.min_val), symlog(self.config.max_val), self.config.num_bins))

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        x = jnp.concat([deter, stoch], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.num_bins, name="logits")(x)
        return logits

    def postprocess(self, logits: jax.Array):
        return symexp(jnp.sum(jax.nn.softmax(logits) * self.bins.value, axis=-1))

    def predict(self, deter: jax.Array, stoch: jax.Array):
        return self.postprocess(self(deter, stoch))

    def loss(self, pred: jax.Array, target: jax.Array):
        target_two_hot = two_hot_symlog(target, self.bins.value)
        return -jnp.sum(target_two_hot * jax.nn.log_softmax(pred), axis=-1).mean()


class ContPredictor(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        x = jnp.concat([deter, stoch], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=1, name="logits")(x).squeeze(-1)
        return logits

    def postprocess(self, logits: jax.Array):
        return jax.nn.sigmoid(logits)

    def predict(self, deter: jax.Array, stoch: jax.Array):
        return self.postprocess(self(deter, stoch))

    def loss(self, pred: jax.Array, target: jax.Array):
        return -jnp.mean(target * jax.nn.log_sigmoid(pred) + (1.0 - target) * jax.nn.log_sigmoid(-pred))


class Posterior(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array, embed: jax.Array):
        x = jnp.concat([deter, embed], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.stoch_size * self.config.classes, name="logits")(x)
        logits = logits.reshape((*logits.shape[:-1], self.config.stoch_size, self.config.classes))
        return logits

    def postprocess(self, logits: jax.Array):
        probs = jax.nn.softmax(logits, axis=-1) * (1 - self.config.uniform_frac) + self.config.uniform_frac / logits.shape[-1]
        return distrax.OneHotCategorical(probs=probs, dtype=jnp.float32)

    def predict(self, deter: jax.Array, embed: jax.Array):
        return self.postprocess(self(deter, embed))


class Prior(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array):
        x = deter
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.stoch_size * self.config.classes, name="logits")(x)
        logits = logits.reshape((*logits.shape[:-1], self.config.stoch_size, self.config.classes))
        return logits

    def postprocess(self, logits: jax.Array):
        probs = jax.nn.softmax(logits, axis=-1) * (1 - self.config.uniform_frac) + self.config.uniform_frac / logits.shape[-1]
        return distrax.OneHotCategorical(probs=probs, dtype=jnp.float32)

    def predict(self, deter: jax.Array):
        return self.postprocess(self(deter))


class Actor(linen.Module):
    config: DictConfig
    act_space: int

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        x = jnp.concat([deter, stoch], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.act_space, name="logits")(x)
        return logits

    def postprocess(self, logits: jax.Array):
        probs = jax.nn.softmax(logits, axis=-1) * (1 - self.config.uniform_frac) + self.config.uniform_frac / logits.shape[-1]
        return distrax.Categorical(probs=probs)

    def predict(self, deter: jax.Array, stoch: jax.Array):
        return self.postprocess(self(deter, stoch))


class Critic(linen.Module):
    config: DictConfig

    def setup(self):
        self.bins = self.variable("state", "bins", lambda: jnp.linspace(symlog(self.config.min_val), symlog(self.config.max_val), self.config.num_bins))

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        x = jnp.concat([deter, stoch], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.num_bins, name="logits")(x)
        return logits

    def postprocess(self, raw: jax.Array):
        return symexp(jnp.sum(jax.nn.softmax(raw) * self.bins.value, axis=-1))

    def predict(self, deter: jax.Array, stoch: jax.Array):
        return self.postprocess(self(deter, stoch))

    def loss(self, pred: jax.Array, target: jax.Array, weight: jax.Array):
        target_two_hot = two_hot_symlog(target, self.bins.value)
        return -jnp.mean(weight * jnp.sum(target_two_hot * jax.nn.log_softmax(pred), axis=-1))
