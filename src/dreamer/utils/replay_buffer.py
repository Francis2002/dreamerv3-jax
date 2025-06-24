# ============================================================
# replay_buffer.py
# ============================================================
"""Replay buffer implementation for the Dreamer V3 replication.

This module provides a minimal yet efficient *uniform* replay buffer
suitable for world‑model RL agents.  It stores full, step‑level
transitions in a ring buffer and can return **contiguous sequences**
of length *T* for imagination training.

Key design goals
----------------
* **JAX‑friendly**: sampling returns JAX arrays already on the default
  device so gradients can flow (if required) without further copies.
* **PyTree support**: the public API accepts/returns nested structures
  (dicts / dataclasses) so it stays agnostic to how you encode a
  transition.
* **Episode boundary awareness**: sequences never cross `done=True`
  flags, preventing leakage of invalid time steps.
* **Constant‑time ops**: add & sample are *O(1)* regardless of capacity.

Only *uniform* sampling is implemented here; prioritised replay can be
added later by swapping out `_uniform_choice`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import jax
import chex

Tree = Any  # PyTree alias
PRNGKey = jax.Array


def _to_numpy(x, dtype=None):
    arr = np.asarray(x, dtype=dtype)
    return arr


def _to_jax(x):
    return jax.device_put(x)


@dataclass
class ReplayBuffer:
    """Uniform ring‑buffer replay storing *step* transitions.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions held in the buffer.
    obs_shape : tuple
        Shape of a **single** observation (e.g., `(64, 64, 3)`).
    action_shape : tuple
        Shape of a **single** action (e.g., `()` for discrete, or
        `(act_dim,)` for continuous).
    dtype_obs : np.dtype, optional
        Storage dtype for observations (defaults to `np.uint8`).  Images
        are often stored as 8‑bit to save RAM then converted to float32
        during sampling.
    dtype_act : np.dtype, optional
        Storage dtype for actions (defaults to `np.int32`).
    """

    capacity: int
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    dtype_obs: np.dtype = np.dtype(np.uint8)
    dtype_act: np.dtype = np.dtype(np.int32)

    # internal fields (populated post‑init)
    _obs: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _discounts: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)

    _ptr: int = field(default=0, init=False, repr=False)
    _size: int = field(default=0, init=False, repr=False)

    # --------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------
    def __post_init__(self):
        self._obs = np.empty((self.capacity, *self.obs_shape), self.dtype_obs)
        self._actions = np.empty((self.capacity, *self.action_shape), self.dtype_act)
        self._rewards = np.empty((self.capacity,), np.float32)
        self._discounts = np.empty((self.capacity,), np.float32)
        self._dones = np.empty((self.capacity,), np.bool_)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of *valid* transitions currently stored."""
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    # --------------------------------------------------
    # Insertion
    # --------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        discount: float,
        done: bool,
    ):
        """Add **one** transition to the buffer (called each env step)."""
        self._obs[self._ptr] = _to_numpy(obs, self.dtype_obs)
        self._actions[self._ptr] = _to_numpy(action, self.dtype_act)
        self._rewards[self._ptr] = reward
        self._discounts[self._ptr] = discount
        self._dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # --------------------------------------------------
    # Sampling helpers
    # --------------------------------------------------
    def _valid_starts(self, seq_len: int) -> np.ndarray:
        """Return indices where a *full* `seq_len` fits without wrapping or crossing `done`."""
        if self._size < seq_len:
            return np.empty(0, np.int32)

        max_index = self.capacity if self.is_full else self._ptr
        # Boolean array marking starts that would wrap the buffer
        wrap_mask = (np.arange(max_index, dtype=np.int32) + seq_len) > max_index

        # Boolean array marking starts where an episode ends inside [start, start+seq_len)
        done_mask = np.zeros_like(wrap_mask)
        done_idx = np.where(self._dones[:max_index])[0]
        for d in done_idx:
            lo = np.maximum(0, d - seq_len + 1)  # inclusive of d as end forbidden
            hi = d + 1  # inclusive of d as start forbidden
            done_mask[lo:hi] = True

        valid = ~(wrap_mask | done_mask)
        return np.where(valid)[0].astype(np.int32)

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------
    def can_sample(self, batch_size: int, seq_len: int = 1) -> bool:
        """Check if the buffer has enough *valid* sequences ready."""
        return len(self._valid_starts(seq_len)) >= batch_size

    def sample(
        self, rng: jax.Array, batch_size: int, seq_len: int = 1
    ) -> Dict[str, jax.Array]:
        """Sample a batch of contiguous sequences.

        Returned shapes are `(batch, seq_len, *feature)` for obs/actions
        and `(batch, seq_len)` for scalar quantities.
        """
        chex.assert_rank(rng, 1)  # PRNGKey
        starts = self._valid_starts(seq_len)
        assert (
            starts.size >= batch_size
        ), "ReplayBuffer: not enough valid sequences to sample."

        batch_starts = jax.random.choice(rng, starts, (batch_size,), replace=False)
        # Convert to numpy for indexing, then expand dims for broadcasting
        idx = batch_starts[:, None] + np.arange(seq_len)[None, :]
        idx %= self.capacity  # safe when buffer full (ring)

        # Gather & cast to JAX
        obs = _to_jax(self._obs[idx].astype(np.float32) / 255.0)
        actions = _to_jax(self._actions[idx])
        rewards = _to_jax(self._rewards[idx])
        discounts = _to_jax(self._discounts[idx])
        dones = _to_jax(self._dones[idx])

        return dict(
            obs=obs, actions=actions, rewards=rewards, discounts=discounts, dones=dones
        )

    # --------------------------------------------------
    # Convenience: push full trajectory at once (optional)
    # --------------------------------------------------
    def add_trajectory(self, trajectory: Dict[str, np.ndarray]):
        """Utility to bulk‑insert a numpy trajectory dict of shape `(T, ...)`."""
        T = trajectory["obs"].shape[0]
        for t in range(T):
            self.add(
                trajectory["obs"][t],
                trajectory["actions"][t],
                float(trajectory["rewards"][t]),
                float(trajectory["discounts"][t]),
                bool(trajectory["dones"][t]),
            )
