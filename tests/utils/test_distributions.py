import jax
import jax.numpy as jnp
import numpy as np

from dreamer.utils.distributions import categorical_kl, straight_through_sample


def test_categorical_kl_zero():
    # When both distributions are identical, KL should be 0.
    p_logits = jnp.array([[1.0, 2.0, 3.0]])  # shape (L=1, K=3)
    q_logits = jnp.array([[1.0, 2.0, 3.0]])
    kl = categorical_kl(p_logits, q_logits)
    np.testing.assert_allclose(kl, 0.0, atol=1e-6)


def test_categorical_kl_positive():

    # Test with known difference.
    p_logits = jnp.array([[2.0, 1.0, 0.1]])  # shape (L=1, K=3)
    q_logits = jnp.array([[1.0, 2.0, 3.0]])

    # Compute KL manually
    p_log_probs = jax.nn.log_softmax(p_logits, axis=-1)
    q_log_probs = jax.nn.log_softmax(q_logits, axis=-1)
    p_probs = jax.nn.softmax(p_logits, axis=-1)
    expected_kl = jnp.sum(p_probs * (p_log_probs - q_log_probs), axis=-1)
    expected_kl_total = jnp.sum(expected_kl, axis=-1)

    kl = categorical_kl(p_logits, q_logits)

    np.testing.assert_allclose(kl, expected_kl_total, atol=1e-6)


def test_straight_through_sample_shape_and_properties():
    # Test that the straight-through sampling returns correct shape and valid probabilities.
    rng = jax.random.key(42)
    probs = jax.random.uniform(rng, shape=(5, 7))  # (L=5, K=7)
    probs = jax.nn.softmax(
        probs, axis=-1
    )  # Ensure probabilities sum to 1 along last dimension
    z_st, idx = straight_through_sample(probs, rng)

    assert z_st.shape == probs.shape, "Output shape should match input shape."
    assert idx.shape == probs.shape[:-1], "Index shape should match (L)."
    assert jnp.allclose(
        jnp.sum(z_st, axis=-1), 1.0
    ), "One-hot encoding should sum to 1 along last dimension."
