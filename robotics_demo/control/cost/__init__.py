from typing import Tuple

import numpy as np

from robotics_demo.control.cost.quadratic import (
    QuadraticStateCost,
    QuadraticStateControlCost,
)


class ConstRefQuadraticStateCost(QuadraticStateCost):
    def __init__(self, weight_matrix: np.ndarray, x_ref: np.ndarray):
        super().__init__(weight_matrix)
        self.x_ref = x_ref

    def get_difference(self, x: np.ndarray, t: float) -> np.ndarray:
        return x - self.x_ref

    def get_d_difference_dx(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.identity(x.size)


class ConstRefQuadraticStateControlCost(QuadraticStateControlCost):
    def __init__(
        self,
        weight_x: np.ndarray,
        weight_u: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
    ):
        super().__init__(weight_x, weight_u)
        self.x_ref = x_ref
        self.u_ref = u_ref

    def get_difference(
        self, x: np.ndarray, u: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x - self.x_ref, u - self.u_ref

    def get_d_difference_dx(
        self, x: np.ndarray, u: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.identity(x.size), np.identity(u.size)
