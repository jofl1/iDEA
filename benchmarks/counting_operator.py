"""Counting LinearOperator for matvec-count instrumentation.

Per the friend's red flag #7 from
``notes/friend_optimization_advice.md``: "Do not benchmark only wall
time. Add a counting LinearOperator and record setup time, matvec
count, matvec time...". This wraps an existing matvec callable,
counts calls, and forwards the result. Used by
``benchmarks/benchmark_interacting.py`` behind a ``--count-matvecs``
flag.
"""

from scipy.sparse.linalg import LinearOperator


class CountingLinearOperator(LinearOperator):
    """LinearOperator that counts matvec applications."""

    def __init__(self, matvec, shape, dtype=float):
        self._inner_matvec = matvec
        self.count = 0
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x):
        self.count += 1
        return self._inner_matvec(x)

    def reset(self):
        self.count = 0
