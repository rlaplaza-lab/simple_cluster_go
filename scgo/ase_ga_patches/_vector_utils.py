from __future__ import annotations

import numpy as np


def random_unit_vector(rng, fallback=None):
    vector = rng.normal(0.0, 1.0, 3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def append_unique_unit_vector(candidates, vector, tol=0.995):
    unit = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(unit)
    if norm <= 1e-12:
        return
    unit /= norm
    for existing in candidates:
        if float(np.dot(unit, existing)) > tol:
            return
    candidates.append(unit)
