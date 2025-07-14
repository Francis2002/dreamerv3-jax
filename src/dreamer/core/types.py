# core/types.py
from typing import NamedTuple
import jax.numpy as jnp


class LatentOut(NamedTuple):
    """Return type of all encoders."""

    z_st: jnp.ndarray  # (B, L, K) float32  – straight-through one-hot with gradients
    logits: jnp.ndarray  # (B, L, K) float32  – pre-softmax values
    probs: jnp.ndarray  # (B, L, K) float32  – 0.99*softmax + 0.01/K
    idx: jnp.ndarray  # (B, L)     int32   – argmax of z_st, useful for logging
