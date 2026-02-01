import jax
from jax import numpy as jnp
from flax import nnx
from omegaconf import DictConfig


class RoPE(nnx.Module):
    def __init__(self, theta_base: float = 10000.0):
        self.theta_base = theta_base

    def __call__(self, x: jax.Array, pos_idx: jax.Array | None = None) -> jax.Array:
        """
        Args:
            x: (B, H, L, D)
            pos_idx: (B, L)
        """
        assert x.shape[-1] % 2 == 0, f"x must have an even number of features, got {x.shape[-1]}"
        if pos_idx is None:
            pos_idx = jnp.arange(x.shape[-3])[None].repeat(x.shape[0], axis=0)
        inv_freq = 1 / self.theta_base ** (jnp.arange(0, x.shape[-1], 2) / x.shape[-1])
        angles = pos_idx[:, :, None, None] * inv_freq[None, None, None, :]
        sin, cos = jnp.sin(angles), jnp.cos(angles)
        x1, x2 = jnp.split(x, 2, axis=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return jnp.concatenate([o1, o2], axis=-1)


def make_rope_attn_fn(rope: RoPE):
    def _rope_attn_fn(query: jax.Array, key: jax.Array, value: jax.Array, **kwargs):
        query_rot = rope(query)
        key_rot = rope(key)
        return nnx.dot_product_attention(query_rot, key_rot, value, **kwargs)

    return _rope_attn_fn


class AttentionBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, hidden_size: int, num_heads: int, dropout: float = 0.0, ffn_scale: float = 2.0):
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            in_features=hidden_size,
            qkv_features=hidden_size,
            num_heads=num_heads,
            decode=False,
            dropout_rate=dropout,
            attention_fn=make_rope_attn_fn(RoPE()),
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        ffn_hidden_size = int(ffn_scale * hidden_size)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, ffn_hidden_size, rngs=rngs),
            nnx.gelu,
            nnx.Linear(ffn_hidden_size, hidden_size, rngs=rngs),
        )

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: DictConfig):
        self.config = config
        self.conv_proj = nnx.Conv(3, config.hidden_size, kernel_size=(config.patch_size, config.patch_size), stride=(config.patch_size, config.patch_size), padding="VALID")
        self.blocks = [AttentionBlock(rngs, config.hidden_size, config.num_heads, config.dropout, config.ffn_scale) for _ in range(config.num_layers)]
        max_val = 1 / config.num_codes
        self.codebook = nnx.Param(jax.random.uniform(rngs.next_key(), (config.num_codes, config.hidden_size), minval=-max_val, maxval=max_val))

    def __call__(self, x: jax.Array):
        x = self.conv_proj(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        for block in self.blocks:
            x = block(x)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        dist = jnp.sum(x**2, axis=-1, keepdims=True) + jnp.sum(self.codebook**2, axis=-1)[None, :] - 2 * jnp.einsum("b d, n d -> b n", x_flat, self.codebook)
        tok_ids = dist.argmin(axis=-1).reshape(B, T)
        vq_toks = self.codebook[tok_ids]

        return x, vq_toks, tok_ids


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: DictConfig):
        self.config = config
        self.blocks = nnx.Sequential(*[AttentionBlock(rngs, config.hidden_size, config.num_heads, config.dropout, config.ffn_scale) for _ in range(config.num_layers)])
        self.conv_upsample = nnx.ConvTranspose(
            config.hidden_size, 3, kernel_size=(config.patch_size, config.patch_size), strides=(config.patch_size, config.patch_size), padding="VALID", rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.blocks(x)
        x = x.reshape(x.shape[0], self.config.latent_h, self.config.latent_w, x.shape[-1])
        x = self.conv_upsample(x)
        return x


class Dynamics(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: DictConfig):
        self.config = config
        self.act_embed = nnx.Embedding(config.action_space_size, config.hidden_size, rngs=rngs)
        self.obs_embed = nnx.Embedding(config.num_codes, config.hidden_size, rngs=rngs)
        self.blocks = [AttentionBlock(rngs, config.hidden_size, config.num_heads, config.dropout, config.ffn_scale) for _ in range(config.num_layers)]

    def _get_block_causal_mask(self, B: int, L: int, K: int) -> jax.Array:
        return jnp.broadcast_to(jnp.kron(jnp.tril(jnp.ones((L, L)), k=0), jnp.ones((K, K)))[None, :, :], (B, L * K, L * K))

    def __call__(self, obs_codes: jax.Array, action: jax.Array, causal: bool = False) -> jax.Array:
        """
        Args:
            obs_codes: (B, L, K)
            action: (B, L)
        """
        B, Lm1, K = obs_codes.shape
        obs_embedded = self.obs_embed(obs_codes)
        act_embedded = self.act_embed(action)
        x = jnp.concat([obs_embedded, act_embedded[:, :, None]], axis=-1)
        mask = self._get_block_causal_mask(B, Lm1 + 1, K) if causal else None
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class RewardPredictor(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: DictConfig):
        self.config = config
        mlp = []
        for _ in range(config.num_layers):
            mlp.append(nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs))
            mlp.append(nnx.gelu)
        mlp.append(nnx.Linear(config.hidden_size, 3, rngs=rngs))
        self.mlp = nnx.Sequential(*mlp)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.mlp(x)


class TermPredictor(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: DictConfig):
        self.config = config
        self.mlp = []
        for _ in range(config.num_layers):
            self.mlp.append(nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs))
            self.mlp.append(nnx.gelu)
        self.mlp.append(nnx.Linear(config.hidden_size, 2, rngs=rngs))
        self.mlp = nnx.Sequential(*self.mlp)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.mlp(x)
