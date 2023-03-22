""" This module should not be used by other modules in package manipulator """
import logging
import time

import numpy as np


from pydrake.systems.controllers import (
    FiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    FiniteHorizonLinearQuadraticRegulatorResult,
)
from pydrake.systems.primitives import LinearSystem
from pydrake.systems.framework import System
from pydrake.trajectories import PiecewisePolynomial


class LinearQuadraticController:
    """This controller assumes:
    - the nominal trajectories are all zero: x0, u0 = 0 (This required linear dynamics (not affine))
    - the desired trajectories are all zero: xd, ud = 0
    Please use LinearDynamicsAtFixedPoint to do conversion.
    """

    def __init__(
        self,
        dynamics,
        quadratic_cost_matrices,
        t0: float,
        tf: float,
    ):
        super(LinearQuadraticController, self).__init__()
        self.dynamics = dynamics
        self.t0 = t0
        self.tf = tf
        self.cost_matrices = quadratic_cost_matrices
        self.result = self.solve_lqr()
        u0 = [self.result.u0.value(t) for t in np.linspace(self.t0, self.tf, 1000)]
        x0 = [self.result.x0.value(t) for t in np.linspace(self.t0, self.tf, 1000)]
        assert np.all(
            [np.linalg.norm(u) <= 1e-15 for u in u0]
        ), "LQR supports only at (0,0)."
        assert np.all(
            [np.linalg.norm(x) <= 1e-15 for x in x0]
        ), "LQR supports only at (0,0)."

    def get_lqr_option(self) -> FiniteHorizonLinearQuadraticRegulatorOptions:
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        x0 = PiecewisePolynomial(np.zeros(shape=[self.dynamics.nx, 1]))
        u0 = PiecewisePolynomial(np.zeros(shape=[self.dynamics.nu, 1]))
        options.x0 = x0
        options.u0 = u0
        # xd, ud will default to x0, u0; Don't set them to the x0, u0 defined above. That will cause Segment fault.
        # options.xd = x0
        # options.ud = u0
        options.N = self.cost_matrices.N
        options.Qf = self.cost_matrices.Qf
        # options.use_square_root_method = True
        return options

    def get_drake_system(self) -> System:
        A, B = self.dynamics.get_dynamics_matrix()
        C = np.identity(A.shape[1])
        D = np.zeros(shape=[C.shape[0], B.shape[1]])
        system = LinearSystem(A, B, C, D)
        return system

    def solve_lqr(self) -> FiniteHorizonLinearQuadraticRegulatorResult:
        system = self.get_drake_system()
        context = system.CreateDefaultContext()
        options = self.get_lqr_option()

        start = time.time()
        result = FiniteHorizonLinearQuadraticRegulator(
            system=system,
            context=context,
            t0=self.t0,
            tf=self.tf,
            Q=self.cost_matrices.Q,
            R=self.cost_matrices.R,
            options=options,
        )
        end = time.time()
        logging.info(f"Solving LQR in {end - start} seconds.")
        self.report_result(result)
        return result

    def report_result(self, result: FiniteHorizonLinearQuadraticRegulatorResult):
        piecewise_poly = result.K
        seg_times = np.array(piecewise_poly.get_segment_times())
        dts = seg_times[1:] - seg_times[:-1]
        near = 0.05
        rate = np.sum(seg_times >= self.tf - near * (self.tf - self.t0)) / len(
            seg_times
        )
        logging.info(
            f"In [{self.t0}, {self.tf}], "
            f"there are totally {piecewise_poly.get_number_of_segments()} segments.\n"
            f"Of which, {rate*100}% are concentrated in {near*100}% near terminal time.\n"
            f"The minimal segment is of length {dts.min()} (max={dts.max()}).\n"
            f"Each segment is a polynomial of order {piecewise_poly.getSegmentPolynomialDegree(0)}.\n"
        )

    def get_control(self, t: float, x: np.ndarray) -> np.ndarray:
        x = x.reshape((-1, 1))
        K = self.result.K.value(t)
        k0 = self.result.k0.value(t)
        return (-np.dot(K, x) - k0).reshape(-1)
