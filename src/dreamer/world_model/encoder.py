import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable

from dreamer.utils.distributions import straight_through_sample
from dreamer.core.types import LatentOut

"""Encoder module for processing image and vector inputs.
This module defines two types of encoders: one for images using a convolutional neural network,
and another for vectors using a fully connected neural network.

Single Input:
Image Input  : (H, W, C)
Vector Input : (D,)
Batched Input:  
Image Input  : (B, H, W, C)
Vector Input : (B, D)

Output:
Logits : (L, K) for single input or (B, L, K) for batched
Sample : (L,) for single input or (B, L) for batched  # int32 indices
"""


class ImageEncoder(nn.Module):
    latent_spec: tuple[int, int]  # (L, K)
    num_filters: int = 32
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> LatentOut:
        """
        Forward pass through the image encoder for a single input.

        Input: (H, W, C)
        """
        L, K = self.latent_spec

        for i, mult in enumerate([1, 2, 4, 8]):
            x = nn.Conv(
                features=self.num_filters * mult,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                name=f"conv_{i}",
            )(x)
            x = nn.RMSNorm(name=f"norm_{i}")(x)
            x = self.activation(x)

        x = x.reshape(-1)  # (feat,)
        logits = nn.Dense(L * K, name="logits")(x)  # (L*K,)
        logits = logits.reshape(L, K)

        probs = 0.99 * jax.nn.softmax(logits, axis=-1) + 0.01 / K
        key = self.make_rng("sample")
        z_st, idx = straight_through_sample(probs, key)

        return LatentOut(z_st=z_st, logits=logits, probs=probs, idx=idx)


class VectorEncoder(nn.Module):
    """
    Vector encoder module using a fully connected neural network for a single input.
    3-layer MLP

    Parameters
    ----------
    latent_spec : tuple[int, int]
        Specification of the latent space as a tuple (L, K) where L is the number of categorical variables
        and K is the number of bins for each variable.
    hidden_sizes : list of int
        Sizes of the hidden layers.
    activation : Callable
        Activation function to apply after each dense layer.
    """

    latent_spec: tuple[int, int]  # (L, K)
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> LatentOut:
        """
        Forward pass through the encoder for a single input.

        Input: (D,)
        Output is L categorical variables (eg. L=32-96), each over K bins (eg. K=41).
        """

        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(size, name=f"dense_{i}")(x)
            x = nn.RMSNorm(name=f"norm_{i}")(x)
            x = self.activation(x)

        L, K = self.latent_spec
        logits = nn.Dense(L * K, name="logits")(x)  # (L*K,)
        logits = logits.reshape(L, K)

        probs = 0.99 * jax.nn.softmax(logits, axis=-1) + 0.01 / K
        key = self.make_rng("sample")
        z_st, idx = straight_through_sample(probs, key)

        return LatentOut(z_st=z_st, logits=logits, probs=probs, idx=idx)


BatchImageEncoder = nn.vmap(
    ImageEncoder,
    variable_axes={"params": None, "dropout": None, "sample": 0},
    split_rngs={"params": False, "dropout": True, "sample": True},
    in_axes=0,
    out_axes=0,
)

BatchVectorEncoder = nn.vmap(
    VectorEncoder,
    variable_axes={"params": None, "dropout": None, "sample": 0},
    split_rngs={"params": False, "dropout": True, "sample": True},
    in_axes=0,
    out_axes=0,
)
