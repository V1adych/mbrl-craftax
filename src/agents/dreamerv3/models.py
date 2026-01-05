import jax
from jax import numpy as jnp
from flax import linen, struct
import distrax
from omegaconf import DictConfig
from .utils import symexp, two_hot_symlog


@struct.dataclass
class WorldModelOutputs:
    obs: jax.Array
    reward: jax.Array
    cont: jax.Array


class BlockDense(linen.Module):
    features: int
    bias: bool = True
    kernel_init: linen.initializers.Initializer = linen.initializers.lecun_normal(batch_axis=0)
    bias_init: linen.initializers.Initializer = linen.initializers.zeros_init()

    @linen.compact
    def __call__(self, x: jax.Array):
        kernel = self.param("kernel", self.kernel_init, (*x.shape[-2:], self.features))
        x = jnp.einsum("...gi,gio->...go", x, kernel)
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


class Encoder(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, obs: jax.Array):
        x = obs - 0.5
        for i, mult in enumerate(self.config.mults):
            x = linen.Conv(features=self.config.depth * mult, kernel_size=(5, 5), strides=(2, 2), name=f"conv_{i}")(x)
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = jax.nn.silu(linen.RMSNorm(name="dense_norm")(linen.Dense(features=self.config.hidden_size, name="dense")(x)))
        return x


class Decoder(linen.Module):
    config: DictConfig

    def setup(self):
        self.bins = self.variable(
            "state", "bins", lambda: jnp.linspace(symexp(self.config.min_val), symexp(self.config.max_val), self.config.num_bins)
        )

    @linen.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array):
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        features = jnp.concat([deter, stoch], axis=-1)

        x = jax.nn.silu(linen.RMSNorm(name="obs_norm")(linen.Dense(features=self.config.hidden_size, name="obs_dense")(features)))
        x = linen.Dense(features=self.config.depth * self.config.mults[-1] * 4 * 4, name="spatial_dense")(x)
        x = x.reshape((*x.shape[:-1], 4, 4, self.config.depth * self.config.mults[-1]))

        for i, mult in reversed(list(enumerate(self.config.mults[:-1]))):
            x = linen.ConvTranspose(features=self.config.depth * mult, kernel_size=(5, 5), strides=(2, 2), name=f"obs_conv_{i}")(x)
            x = jax.nn.silu(linen.RMSNorm(name=f"obs_norm_{i}")(x))

        obs = linen.ConvTranspose(features=3, kernel_size=(5, 5), strides=(2, 2), name="obs_final")(x)

        r = features
        for i in range(self.config.reward_layers):
            r = jax.nn.silu(linen.RMSNorm(name=f"reward_norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"reward_dense_{i}")(r)))
        reward = linen.Dense(features=self.config.num_bins, name="reward_final")(r)

        c = features
        for i in range(self.config.cont_layers):
            c = jax.nn.silu(linen.RMSNorm(name=f"cont_norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"cont_dense_{i}")(c)))
        cont = linen.Dense(features=1, name="cont_final")(c)
        return WorldModelOutputs(obs=obs, reward=reward, cont=cont)

    def postprocess(self, raw: WorldModelOutputs):
        obs = jax.nn.sigmoid(raw.obs)
        reward = symexp(jnp.sum(jax.nn.softmax(raw.reward) * self.bins.value, axis=-1))
        cont = jax.nn.sigmoid(raw.cont)
        return WorldModelOutputs(obs=obs, reward=reward, cont=cont)

    def loss(self, pred: WorldModelOutputs, target: WorldModelOutputs):
        obs_loss = 0.5 * jnp.square(pred.obs - target.obs).sum(axis=(-3, -2, -1)).mean()
        reward_target = two_hot_symlog(target.reward, self.bins.value)
        reward_loss = -jnp.sum(reward_target * jax.nn.log_softmax(pred.reward), axis=-1).mean()
        cont_loss = -jnp.sum(target.cont * jax.nn.log_sigmoid(pred.cont) + (1.0 - target.cont) * jax.nn.log_sigmoid(-pred.cont), axis=-1).mean()
        return obs_loss + reward_loss + cont_loss


class Posterior(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array, tokens: jax.Array):
        x = jnp.concat([deter, tokens], axis=-1)
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.stoch * self.config.classes, name="logits")(x)
        return distrax.OneHotCategorical(logits=logits.reshape((*logits.shape[:-1], self.config.stoch, self.config.classes)))


class Prior(linen.Module):
    config: DictConfig

    @linen.compact
    def __call__(self, deter: jax.Array):
        x = deter
        for i in range(self.config.num_layers):
            x = jax.nn.silu(linen.RMSNorm(name=f"norm_{i}")(linen.Dense(features=self.config.hidden_size, name=f"dense_{i}")(x)))
        logits = linen.Dense(features=self.config.stoch * self.config.classes, name="logits")(x)
        return distrax.OneHotCategorical(logits=logits.reshape((*logits.shape[:-1], self.config.stoch, self.config.classes)))


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
        return distrax.Categorical(logits=logits)


class Critic(linen.Module):
    config: DictConfig

    def setup(self):
        self.bins = self.variable(
            "state", "bins", lambda: jnp.linspace(symexp(self.config.min_val), symexp(self.config.max_val), self.config.num_bins)
        )

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

    def loss(self, pred: jax.Array, target: jax.Array):
        target_two_hot = two_hot_symlog(target, self.bins.value)
        return -jnp.sum(target_two_hot * jax.nn.log_softmax(pred), axis=-1).mean()
