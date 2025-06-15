# Utils

## transforms.py

Dreamer V3 wants every scalar prediction task (reward, value/return, sometimes episode length) to:

- **Behave like a classification loss**
    - Stable gradients, scale-invariant cross-entropy, no exploding MSE when numbers are huge
- **Behave like a regression loss**
    - Small changes in the target should cause small changes in the loss, not an all-or-nothing class flip

The classic trick in distributional RL (e.g. C-51) is to turn a scalar into a probability distribution over a fixed support of values and use cross-entropy.
The two-hot encoding is the minimal such distribution: we put probability mass on exactly the two neighboring support points and linearly split it so the expectation equals the original scalar.

### Example

Assume the symlogged target is s = 3.7.

1. Locate the two nearest bins → 3 and 4.
2. Compute the fractional distance: w = (s - 3)/(4 - 3) = 0.7
3. Emit a 41-D vector with:
     - prob = 0.3 at bin 3
     - prob = 0.7 at bin 4
     - zeros elsewhere

Because probability mass is shared, the gradient w.r.t the logits at both neighbor bins is non-zero, giving smooth learning.
