import abc
import dataclasses
from typing import Tuple

import numpy as np


@dataclasses.dataclass
class ConstantQuadraticControlMatrices:
    """ xf'Qf xf + integrate (x'Qx + u'Ru + 2x'Nu) """
    Q: np.ndarray
    R: np.ndarray
    N: np.ndarray
    Qf: np.ndarray

    def convert_to_infinite_horizon(self, terminal_convert_ratio: float):
        """ :returns (Q, R, N) """
        return self.Q + terminal_convert_ratio * self.Qf, self.R, self.N


class StateControlCost:
    """This is a function f(x,u,t) with its derivatives (f_x, f_u)"""

    @abc.abstractmethod
    def get_value(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        pass

    @abc.abstractmethod
    def get_derivative(
        self, x: np.ndarray, u: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """This function return the derivative of the cost with respect to state and control as two row-vectors."""
        pass


class StateCost:
    """ This is a function f(x,t) with its derivatives f_x """

    @abc.abstractmethod
    def get_value(self, x: np.ndarray, t: float) -> float:
        pass

    @abc.abstractmethod
    def get_derivative(self, x: np.ndarray, t: float) -> np.ndarray:
        """This function return the derivative of the cost with respect to x as row-vector """
        pass


class ReferenceBasedCostFunction:
    """
        This is the model class for a function of form: f(diff_x, diff_u).
        It computes derivatives f_x, f_u at (diff_x, diff_u).
        Note that the derivatives are always represented as row-vectors.
    """

    @abc.abstractmethod
    def _get_value(self, diff_x: np.ndarray, diff_u: np.ndarray):
        pass

    @abc.abstractmethod
    def _get_derivative(
        self, diff_x: np.ndarray, diff_u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
