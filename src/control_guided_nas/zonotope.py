from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Zonotope:
    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(-1)
        generators = np.asarray(self.generators, dtype=float)
        if generators.ndim != 2:
            raise ValueError("generators must be a 2D array")
        if generators.shape[0] != center.shape[0]:
            raise ValueError("center and generators dimension mismatch")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    @property
    def order(self) -> float:
        if self.dim == 0:
            return 0.0
        return self.generators.shape[1] / self.dim


def linear_map(matrix: np.ndarray, zonotope: Zonotope) -> Zonotope:
    matrix = np.asarray(matrix, dtype=float)
    return Zonotope(matrix @ zonotope.center, matrix @ zonotope.generators)


def minkowski_sum(left: Zonotope, right: Zonotope) -> Zonotope:
    if left.dim != right.dim:
        raise ValueError("zonotopes must have the same dimension")
    center = left.center + right.center
    generators = np.concatenate([left.generators, right.generators], axis=1)
    return Zonotope(center, generators)


def remove_redundant_generators(zonotope: Zonotope, tol: float = 1e-12) -> Zonotope:
    if zonotope.generators.size == 0:
        return zonotope
    norms = np.linalg.norm(zonotope.generators, axis=0)
    keep = norms > tol
    if np.all(keep):
        return zonotope
    generators = zonotope.generators[:, keep]
    return Zonotope(zonotope.center, generators)


def reduce_order(zonotope: Zonotope, reduced_order: float) -> Zonotope:
    if reduced_order <= 0:
        raise ValueError("reduced_order must be positive")
    dim = zonotope.dim
    if dim == 0:
        return zonotope
    max_generators = int(np.floor(reduced_order * dim))
    if zonotope.generators.shape[1] <= max_generators:
        return zonotope

    norms = np.linalg.norm(zonotope.generators, axis=0)
    order = np.argsort(norms)[::-1]
    keep = order[:max_generators]
    drop = order[max_generators:]

    kept = zonotope.generators[:, keep]
    if drop.size == 0:
        return Zonotope(zonotope.center, kept)

    box_diag = np.sum(np.abs(zonotope.generators[:, drop]), axis=1)
    box_generators = np.diag(box_diag)
    generators = np.concatenate([kept, box_generators], axis=1)
    return Zonotope(zonotope.center, generators)


def diameter(zonotope: Zonotope, nx: int) -> float:
    if nx <= 0:
        return 0.0
    if zonotope.generators.size == 0:
        return 0.0
    generators = zonotope.generators[:nx, :]
    return float(2.0 * np.max(np.sum(np.abs(generators), axis=1)))
