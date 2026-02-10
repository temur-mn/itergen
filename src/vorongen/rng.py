"""
Randomness utilities with deterministic seed derivation.
"""

import hashlib

import numpy as np


# Seed namespace convention:
# - Use one base seed for the entire generation run.
# - Derive per-feature streams with RNG.derive_seed(seed, "<namespace>", column_id, ...).
# - Reserved namespaces: init, marginal, conditional.
class RNG:
    def __init__(self, seed=42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def derive_seed(base_seed, *parts):
        h = hashlib.sha256()
        h.update(str(base_seed).encode())
        for part in parts:
            h.update(b":")
            h.update(str(part).encode())
        return int(h.hexdigest(), 16) % (2**32)

    def choice(self, a, size=None, replace=True, p=None):
        return self.rng.choice(a, size=size, replace=replace, p=p)

    def random(self, size=None):
        return self.rng.random(size)
