"""Deterministic weighted base-count allocation shared by public generators."""
from __future__ import annotations


def allocate_counts(count: int, weights: dict[str, float], *, minimum: int = 0) -> dict[str, int]:
    if count < 1 or not weights or any(value <= 0 for value in weights.values()):
        raise ValueError('count and family weights must be positive')
    if minimum * len(weights) > count:
        raise ValueError('count is too small to cover the requested families')
    remaining = count - minimum * len(weights)
    total = sum(weights.values())
    exact = {key: remaining * value / total for key, value in weights.items()}
    result = {key: minimum + int(value) for key, value in exact.items()}
    for key in sorted(weights, key=lambda key: (-(exact[key] % 1), key))[:count-sum(result.values())]:
        result[key] += 1
    return result
