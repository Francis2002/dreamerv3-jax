# ============================================================
# test_replay_buffer.py  (pytest)
# ============================================================
"""Unit‑tests for `ReplayBuffer`.
Run with:  `pytest replay_buffer.py`  (thanks to pytest‑collector running all test_*)
"""

import pytest
from dreamer.utils.replay_buffer import ReplayBuffer
import jax
import jax.numpy as jnp
import numpy as np


def _fill_buffer(cap: int = 32):
    rb = ReplayBuffer(capacity=cap, obs_shape=(4,), action_shape=())
    # Fill half with a single episode, half with another.
    for i in range(cap // 2):
        rb.add(
            np.full((4,), i, np.uint8),
            np.array(np.int32(i % 4)),
            reward=float(i),
            discount=1.0,
            done=False,
        )
    rb.add(
        np.zeros((4,), np.uint8), np.array(0), 0.0, 0.0, done=True
    )  # mark episode end
    for i in range(cap // 2):
        rb.add(
            np.full((4,), i, np.uint8),
            np.array(np.int32(i % 4)),
            reward=float(i),
            discount=1.0,
            done=False,
        )
    return rb


def test_add_and_size():
    rb = ReplayBuffer(capacity=10, obs_shape=(3,), action_shape=())
    for i in range(5):
        rb.add(np.zeros(3, np.uint8), np.array(np.int32(0)), 0.0, 1.0, False)
    assert rb.size == 5
    for i in range(10):
        rb.add(np.zeros(3, np.uint8), np.array(np.int32(0)), 0.0, 1.0, False)
    assert rb.size == 10  # capacity capped


def test_can_sample_false_when_insufficient():
    rb = ReplayBuffer(capacity=16, obs_shape=(2,), action_shape=())
    assert rb.can_sample(batch_size=1) is False
    rb.add(np.zeros(2, np.uint8), np.array(np.int32(0)), 0.0, 1.0, False)
    assert rb.can_sample(batch_size=1) is True


def test_sample_shapes_and_dtypes():
    rb = _fill_buffer(64)
    rng = jax.random.PRNGKey(0)
    batch = rb.sample(rng, batch_size=8, seq_len=4)
    assert batch["obs"].shape == (8, 4, 4)
    assert batch["actions"].shape == (8, 4)
    assert batch["rewards"].shape == (8, 4)
    assert batch["discounts"].shape == (8, 4)
    assert batch["dones"].shape == (8, 4)
    assert batch["obs"].dtype == jnp.float32


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__]))
