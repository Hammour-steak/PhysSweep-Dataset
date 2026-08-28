"""Axis-aligned visual-to-proxy fitting shared by PhysAssets curation tools."""

from __future__ import annotations

import math


def rotation_matrix(euler: tuple[int, int, int]) -> list[list[float]]:
    x, y, z = (math.radians(value) for value in euler)
    rx = [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    ry = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    rz = [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]

    def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return multiply(rz, multiply(ry, rx))


def best_axis_alignment(
    source: list[float], target: list[float]
) -> tuple[list[float], list[float], float]:
    candidates = []
    for x in (0, 90, 180, 270):
        for y in (0, 90, 180, 270):
            for z in (0, 90, 180, 270):
                euler = (x, y, z)
                matrix = rotation_matrix(euler)
                rotated = [sum(abs(matrix[i][j]) * source[j] for j in range(3)) for i in range(3)]
                ratios = [target[i] / max(rotated[i], 1.0e-12) for i in range(3)]
                scale = sorted(ratios)[1]
                predicted = [value * scale for value in rotated]
                error = max(abs(predicted[i] - target[i]) / target[i] for i in range(3))
                candidates.append((error, sum(min(value, 360 - value) for value in euler), euler, predicted))
    error, _, euler, predicted = min(candidates)
    return [float(value) for value in euler], predicted, float(error)
