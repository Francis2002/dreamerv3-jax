import jax
import jax.numpy as jnp

from dreamer.world_model.encoder import (
    ImageEncoder,
    VectorEncoder,
    BatchImageEncoder,
    BatchVectorEncoder,
)


def test_image_encoder_output_shape():
    H, W, C = 64, 64, 3
    L, K = 10, 32  # Example latent spec
    dummy_image = jnp.ones((H, W, C))
    model = ImageEncoder(latent_spec=(L, K))  # Example latent spec
    rngs = {"sample": jax.random.key(0), "params": jax.random.key(0)}
    variables = model.init(rngs, dummy_image)
    out = model.apply(variables, dummy_image, rngs=rngs, mutable=False)

    z_st = out[0]
    logits = out[1]
    probs = out[2]
    idx = out[3]
    assert z_st.shape == (L, K)
    assert logits.shape == (L, K)
    assert probs.shape == (L, K)
    assert idx.shape == (L,)


def test_vector_encoder_output_shape():
    D = 10
    L, K = 8, 16  # Example latent spec
    dummy_vectors = jnp.ones((D,))
    model = VectorEncoder(latent_spec=(L, K))  # Example latent spec
    rngs = {"sample": jax.random.key(0), "params": jax.random.key(0)}
    variables = model.init(rngs, dummy_vectors)
    out = model.apply(variables, dummy_vectors, rngs=rngs, mutable=False)

    z_st = out[0]
    logits = out[1]
    probs = out[2]
    idx = out[3]
    assert z_st.shape == (L, K)
    assert logits.shape == (L, K)
    assert probs.shape == (L, K)
    assert idx.shape == (L,)


def test_batch_image_encoder():
    B, H, W, C = 4, 64, 64, 3
    L, K = 10, 32  # Example latent spec
    dummy_images = jnp.ones((B, H, W, C))
    model = BatchImageEncoder(latent_spec=(L, K))  # Example latent spec
    rngs = {"sample": jax.random.key(0), "params": jax.random.key(0)}
    variables = model.init(rngs, dummy_images)
    out = model.apply(variables, dummy_images, rngs=rngs, mutable=False)

    z_st = out[0]
    logits = out[1]
    probs = out[2]
    idx = out[3]
    assert z_st.shape == (B, L, K)
    assert logits.shape == (B, L, K)
    assert probs.shape == (B, L, K)
    assert idx.shape == (
        B,
        L,
    )


def test_batch_vector_encoder():
    B, D = 4, 10
    L, K = 8, 16  # Example latent spec
    dummy_vectors = jnp.ones((B, D))
    model = BatchVectorEncoder(latent_spec=(L, K))  # Example latent spec
    rngs = {"sample": jax.random.key(0), "params": jax.random.key(0)}
    variables = model.init(rngs, dummy_vectors)
    out = model.apply(variables, dummy_vectors, rngs=rngs, mutable=False)

    z_st = out[0]
    logits = out[1]
    probs = out[2]
    idx = out[3]
    assert z_st.shape == (B, L, K)
    assert logits.shape == (B, L, K)
    assert probs.shape == (B, L, K)
    assert idx.shape == (
        B,
        L,
    )


if __name__ == "__main__":
    test_image_encoder_output_shape()
    test_vector_encoder_output_shape()
    test_batch_image_encoder()
    test_batch_vector_encoder()
    print("All tests passed!")
