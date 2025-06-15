# src/dreamerv3/utils/transforms.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

### ---------------------------------  symlog / symexp  ---------------------------------------- ###

_SYMLOG_EPS = 1e-6  # avoids log(0) on jitted edge-cases


@jax.jit
def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Bi-symmetric log transform used by DreamerV3.
    Compresses magnitude while preserving sign."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x) + _SYMLOG_EPS)


@jax.jit
def symexp(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(y) * (jnp.expm1(jnp.abs(y)))


### ------------------------------  two-hot discretization  ---------------------------------- ###

# Exponentially spaced bin centers (range: ~e±20 ≈ ±4.85e8)
_BINS = jnp.asarray(symexp(jnp.arange(-20.0, 21.0, 1.0)))  # 41 bins


# @partial(jax.jit, static_argnums=(1,))
def two_hot(x: jnp.ndarray, bins: jnp.ndarray = _BINS) -> jnp.ndarray:
    """Encode a scalar or array of scalars into two-hot targets.

    Returns a probability vector whose mass is split between the two nearest
    bins — linearly weighted by distance."""
    x = x[..., None]  # shape [..., 1]
    # Find index of the rightmost bin <= x
    idx_hi = jnp.clip(jnp.sum(bins <= x, axis=-1) - 1, 0, bins.size - 2)
    idx_lo = idx_hi + 1

    bin_lo = jnp.take(bins, idx_hi)
    bin_hi = jnp.take(bins, idx_lo)

    # Linear weights
    w_hi = (x - bin_lo[..., None]) / (bin_hi[..., None] - bin_lo[..., None] + 1e-8)
    w_lo = 1.0 - w_hi

    one_hot_lo = jax.nn.one_hot(idx_hi, bins.size)
    one_hot_hi = jax.nn.one_hot(idx_lo, bins.size)

    return w_lo * one_hot_lo + w_hi * one_hot_hi  # shape [..., bins]


@jax.jit
def two_hot_decode(probs: jnp.ndarray, bins: jnp.ndarray = _BINS) -> jnp.ndarray:
    """Expected scalar under a softmax distribution parameterised by `bins`."""
    # Sum positives & negatives separately for numerical stability (large range)
    pos_mask = bins > 0
    neg_mask = ~pos_mask
    pos = jnp.sum(probs * bins * pos_mask, axis=-1)
    neg = jnp.sum(probs * bins * neg_mask, axis=-1)
    return pos + neg


### -------------  quick test  -------------------------------------- ###


def _self_test():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1e3, 1e3, size=(128,))
    y = symexp(symlog(x))
    assert np.allclose(x, y, atol=1e-5), "symexp symlog mismatch"

    probs = two_hot(jnp.asarray(x))
    assert probs.shape == (128, 41), "two_hot shape mismatch"
    x_hat = two_hot_decode(probs)
    assert np.mean(np.abs(x - np.asarray(x_hat))) < 1e-1, "two_hot ≈ id failed"


if __name__ == "__main__":
    _self_test()
    print("transforms self-test passed")
