import abc
from typing import Optional, Tuple

import numpy as np

from robotics_demo.control.cost.base import ReferenceBasedCostFunction, StateCost, StateControlCost


class QuadraticForm(ReferenceBasedCostFunction):
    def __init__(self, weight_x: np.ndarray, weight_u: Optional[np.ndarray]):
        self.weight_x = weight_x
        self.weight_u = weight_u
        self.weight_x_sym = self.weight_x + self.weight_x.T
        self.weight_u_sym = (
            self.weight_u + self.weight_u.T if self.weight_u is not None else None
        )

    def _get_value(self, diff_x: np.ndarray, diff_u: Optional[np.ndarray]) -> float:
        value = 0.0
        value += 0.5 * np.dot(diff_x, self.weight_x @ diff_x)
        if self.weight_u is not None:
            value += 0.5 * np.dot(diff_u, self.weight_u @ diff_u)
        else:
            assert diff_u is None
        return value

    def _get_derivative(
        self, diff_x: np.ndarray, diff_u: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ddx = 0.5 * self.weight_x_sym @ diff_x
        ddu = 0.5 * self.weight_u_sym @ diff_u if self.weight_u_sym is not None else None
        return ddx, ddu


class QuadraticStateCost(StateCost, QuadraticForm):
    def __init__(self, weight_matrix: np.ndarray):
        super(QuadraticStateCost, self).__init__(weight_x=weight_matrix, weight_u=None)

    @abc.abstractmethod
    def get_difference(self, x: np.ndarray, t: float) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_d_difference_dx(self, x: np.ndarray, t: float) -> np.ndarray:
        """A matrix with (diff_i)_{x_j}: i is row index, j is column index"""
        pass

    def get_value(self, x: np.ndarray, t: float) -> float:
        diff = self.get_difference(x, t)
        return self._get_value(diff, None)

    def get_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        diff = self.get_difference(x, t)
        d_diff_dx = self.get_d_difference_dx(x, t)
        df_d_diff = self._get_derivative(diff, None)[0]
        return df_d_diff @ d_diff_dx


class QuadraticStateControlCost(StateControlCost, QuadraticForm):
    """ Cross-term is not implemented for now. """

    def __init__(self, weight_x: np.ndarray, weight_u: np.ndarray):
        super().__init__(weight_x, weight_u)

    @abc.abstractmethod
    def get_difference(
        self, x: np.ndarray, u: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def get_d_difference_dx(self, x: np.ndarray, u: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_value(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        diff_x, diff_u = self.get_difference(x, u, t)
        return self._get_value(diff_x, diff_u)

    def get_derivative(
        self, x: np.ndarray, u: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        diffx, diffu = self.get_difference(x, u, t)
        d_diffx_dx, d_diffu_du = self.get_d_difference_dx(x, u, t)
        df_d_diffx, df_d_diffu = self._get_derivative(diffx, diffu)
        return df_d_diffx @ d_diffx_dx, df_d_diffu @ d_diffu_du
