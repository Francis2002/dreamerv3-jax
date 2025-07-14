import jax.numpy as jnp
import jax.nn


def categorical_kl(
    p_logits: jnp.ndarray,  # (L,K)
    q_logits: jnp.ndarray,  # (L,K)
) -> jnp.ndarray:
    """
    Compute the KL divergence between two categorical distributions given their logits.
    Returns KL in nats using logsumexp

    Parameters
    ----------
    p_logits : jnp.ndarray
        Logits of the first categorical distribution (shape: [L, K]).
    q_logits : jnp.ndarray
        Logits of the second categorical distribution (shape: [L, K]).

    Returns
    -------
    jnp.ndarray
        The KL divergence between the two distributions (scalar).
    """

    # Convert logits to log probabilities
    p_log_probs = jax.nn.log_softmax(p_logits, axis=-1)
    q_log_probs = jax.nn.log_softmax(q_logits, axis=-1)

    # Compute the KL divergence
    kl = jnp.sum(
        jax.nn.softmax(p_logits, axis=-1) * (p_log_probs - q_log_probs), axis=-1
    )

    return jnp.sum(kl, axis=-1)  # Sum over L


def straight_through_sample(
    probs: jnp.ndarray,  # (L, K), sums to 1 on axis=-1
    key: jnp.ndarray,  # JAX random key for sampling
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns
    -------
    z_st : float32, (L,K)  - straight-through one-hot
    idx  : int32,  (L,)     - discrete indices
    """
    # Ensure input is 2D (batch_size, length, features)
    if probs.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, got shape {probs.shape}")

    # Sample latent variables using the provided rng key
    z = jax.random.categorical(key, probs, axis=-1)

    # Convert to one-hot encoding
    z_one_hot = jax.nn.one_hot(z, probs.shape[-1])  # shape (L, K)

    # Correct straight-through: use stop_gradient to forward the one-hot values
    z_st = z_one_hot + jax.lax.stop_gradient(probs - z_one_hot)
    return z_st, z
