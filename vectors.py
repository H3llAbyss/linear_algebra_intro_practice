from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    non_zero_count = np.random.randint(0, dim)
    non_zero_positions = np.random.choice(dim, non_zero_count, replace=False)

    dense = np.zeros(dim)
    dense[non_zero_positions] = np.random.rand(non_zero_count)

    dense = dense.reshape(-1, 1)

    return sparse.coo_matrix(dense)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return x * a


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    return sum(c * v for c, v in zip(coeffs, vectors))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    return float(np.dot(x.T, y))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, ord= order)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return np.linalg.norm(x - y)


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        np.ndarray: angle in deg.
    """
    dot_product = np.dot(x.T, y)[0, 0] 
    norm_product = np.linalg.norm(x) * np.linalg.norm(y)
    cosine_similarity = dot_product / norm_product

    return float(np.degrees(np.arccos(cosine_similarity)))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    return cos_between_vectors(x, y) == 90


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    return np.linalg.solve(a,b)
