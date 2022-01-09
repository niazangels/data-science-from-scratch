from typing import List

Vector = List[float]

height_weight_age = [70, 170, 40]  # inches  pounds  years
grades = [95, 80, 75, 62]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be of same length"
    return [vi + wi for vi, wi in zip(v, w)]


assert add([1, 2, 3], [1, 2, 3]) == [2, 4, 6]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be of same length"
    return [vi - wi for vi, wi in zip(v, w)]


assert subtract([1, 2, 3], [1, 2, 3]) == [0, 0, 0]


def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "no vectors provided"

    length = len(vectors[0])
    assert all(len(v) == length for v in vectors)

    return [sum(v[i] for v in vectors) for i in range(length)]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * vi for vi in v]


assert scalar_multiply(3, [1, 2, 3]) == [3, 6, 9]


def vector_mean(vectors: List[Vector]) -> Vector:
    summed_vector = vector_sum(vectors)
    return [v / len(vectors) for v in summed_vector]


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def dot(v: Vector, w: Vector) -> float:
    return sum((vi * wi for vi, wi in zip(v, w)))


assert dot([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    # return sum(vi**2 for vi in v)
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14

import math


def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


assert magnitude([3, 4]) == 5


def squared_distance(v: Vector, w: Vector):
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector):
    # return math.sqrt(squared_distance(v,w))
    return magnitude(subtract(v, w))


Matrix = List[List[float]]

A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 2], [3, 4], [5, 6]]

from typing import Tuple


def shape(A: Matrix) -> Tuple[int, int]:
    n_rows = len(A)
    n_cols = len(A[0]) if A else 0  # shape([])
    return n_rows, n_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
assert shape([]) == (0, 0)


def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


def get_col(A: Matrix, i: int) -> Vector:
    return [row[i] for row in A]


from typing import Callable


def make_matrix(n_rows: int, n_cols: int, entry_fn: Callable[[int, int], float]):
    return [[entry_fn(i, j) for j in range(n_cols)] for i in range(n_rows)]


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, entry_fn=lambda x, y: 1 if x == y else 0)


assert identity_matrix(5) == [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]


