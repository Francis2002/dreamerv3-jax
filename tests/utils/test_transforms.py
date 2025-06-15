import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st

from dreamer.utils.transforms import symlog, symexp, two_hot, two_hot_decode


@given(st.floats(-1e6, 1e6))
def test_symlog_round_trip(x):
    # round-trip must be (almost) identity
    y = symexp(symlog(jnp.asarray(x)))
    assert np.allclose(x, y, atol=1e-5)


@pytest.mark.parametrize("val", [-3.2, 0.0, 5.7])
def test_two_hot_shape_and_decode(val):
    probs = two_hot(jnp.asarray(val))
    assert probs.shape == (41,)  # 41 = len(_BINS)
    decoded = two_hot_decode(probs)
    assert np.isclose(val, decoded, atol=1e-1)
