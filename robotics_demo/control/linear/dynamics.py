from typing import Tuple

import numpy as np


class LinearDynamicsAtFixedPoint:
    """Convert and approximate original system into a linear one
    The affine part is kept in this class and restored calling suitable function of this class.
    Given the dynamics f, x0, u0: this class linearize the dynamics at x0, u0.
    New var xx, uu is used: xx = x - x0, uu = u - u0 with ff(xx,uu) = f(xx+x0,uu+u0)
        xx'=ff(xx,uu)~=ff(0,0)+ff_x(0,0)*xx+ff_u(0,0)*uu
        A=ff_x(0,0)=f_x(x0,u0), B=ff_u(0,0)=f_u(x0,u0)
    """

    def __init__(self, dynamics, x0: np.ndarray, u0: np.ndarray):
        self._dynamics = dynamics
        self.x0 = x0
        self.u0 = u0
        f0 = self._dynamics.get_value(x0, u0)
        assert np.linalg.norm(f0) <= 1e-10, "Should Only Linearize At Equilibrium."
        self.A, self.B = self._dynamics.get_linear_dynamics(x0, u0)

    def step_forward(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        nq = self._dynamics.nu
        df_dx = self.get_value(x, u)
        v = x[nq:] + dt * df_dx[nq:]
        q = x[:nq] + dt * v
        return np.r_[q, v]

    def get_linear_dynamics(
        self, x0: np.ndarray, u0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_dynamics_matrix()

    def get_dynamics_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.A, self.B

    def get_value(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xx = x - self.x0
        uu = u - self.u0
        return np.dot(self.A, xx) + np.dot(self.B, uu)

    @property
    def nu(self) -> int:
        return self._dynamics.nu

    @property
    def nx(self) -> int:
        return self._dynamics.nx
