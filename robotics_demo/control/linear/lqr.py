""" This module should not be used by other modules in package manipulator """
import io
import logging
import time
from typing import Union
import pickle

import numpy as np
import torch

from robotics_demo.control.cost.base import ConstantQuadraticControlMatrices
from robotics_demo.control.linear.dynamics import LinearDynamicsAtFixedPoint

from pydrake.systems.controllers import (
    FiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    FiniteHorizonLinearQuadraticRegulatorResult,
)
from pydrake.systems.primitives import LinearSystem
from pydrake.systems.framework import System
from pydrake.trajectories import PiecewisePolynomial

from robotics_demo.network import AutoTorchDeviceUnpickler, SaveLoad


class LinearQuadraticController(SaveLoad):
    """This controller assumes:
    - the nominal trajectories are all zero: x0, u0 = 0 (This required linear dynamics (not affine))
    - the desired trajectories are all zero: xd, ud = 0
    Please use LinearDynamicsAtFixedPoint to do conversion.
    """

    def __init__(
        self,
        dynamics: LinearDynamicsAtFixedPoint,
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


class LinearQuadraticControllerTorch(SaveLoad):
    @classmethod
    def construct_lqr(
        cls,
        dynamics: LinearDynamicsAtFixedPoint,
        quadratic_cost_matrices: ConstantQuadraticControlMatrices,
        t0: float,
        tf: float,
        dt: float,
        drop_last: float = 0.0,
    ) -> "LinearQuadraticControllerTorch":
        lqr = LinearQuadraticController(dynamics, quadratic_cost_matrices, t0, tf)
        n_steps = round((tf - t0) / dt)
        assert (
            abs(tf - t0 - n_steps * dt) <= dt * 1e-8
        ), "Time interval length must be multiples of dt."
        if drop_last >= 1e-7:
            segment_times = np.array(lqr.result.K.get_segment_times())
            drop_after = tf - (tf - t0) * drop_last
            last_index = np.where(segment_times >= drop_after)[0][0]
            n_drops = len(segment_times) - last_index
            logging.info(f"Drop last {n_drops} segments of LQR controller!")
            for _ in range(n_drops):
                lqr.result.K.RemoveFinalSegment()
                lqr.result.k0.RemoveFinalSegment()

        Kt: torch.Tensor = torch.stack(
            [
                torch.tensor(lqr.result.K.value(lqr.t0 + n * dt))
                for n in range(n_steps + 1)
            ]
        )
        k0t: torch.Tensor = torch.stack(
            [
                torch.tensor(lqr.result.k0.value(lqr.t0 + n * dt))
                for n in range(n_steps + 1)
            ]
        )
        return cls(t0, tf, dt, Kt, k0t)

    def get_data_dict(self):
        return {
            "t0": self.t0,
            "tf": self.tf,
            "dt": self.dt,
            "Kt": self.Kt,
            "k0t": self.k0t,
        }

    def save_impl(self, buffer, *args, **kwargs):
        pickle.dump(self.get_data_dict(), buffer)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs) -> "LinearQuadraticControllerTorch":
        # return cls(**pickle.load(buff))
        return cls(**AutoTorchDeviceUnpickler(buff).load())

    def __init__(
        self,
        t0: float,
        tf: float,
        dt: float,
        Kt: torch.Tensor,
        k0t: torch.Tensor,
    ):
        self.tf = torch.scalar_tensor(tf)
        self.t0 = torch.scalar_tensor(t0)
        self.dt = torch.scalar_tensor(dt)
        self.Kt = Kt
        self.k0t = k0t

    def to(self, *args, **kwargs):
        self.Kt = self.Kt.to(*args, **kwargs)
        self.k0t = self.k0t.to(*args, **kwargs)
        self.tf = self.tf.to(*args, **kwargs)
        self.t0 = self.t0.to(*args, **kwargs)
        self.dt = self.dt.to(*args, **kwargs)
        return self

    def from_time_to_n(self, t: Union[float, torch.Tensor]) -> int:
        t = torch.scalar_tensor(t)
        return torch.round((t - self.t0) / self.dt).long()

    def from_time_to_n_batch(self, t: torch.Tensor) -> torch.LongTensor:
        return torch.round((t - self.t0) / self.dt).long().flatten()

    def get_control(
        self, t: Union[float, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """x: (n_batch, nx)
        K: (nu, nx), k0: (nu, 1)
        (K*x.T): (nu, n_batch)
        """
        n = self.from_time_to_n(t)
        K, k0 = self.Kt[n], self.k0t[n]
        return (-torch.matmul(K, x.T) - k0).T

    def get_control_batch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The differentiability through t is broken.
        It can only be resolved by recovering the formulae of K(t), k0(t),
         which can be done through taking each polynomial of K, k0. (there are thousands segments)
         t: (n_batch,) x: (n_batch, nx)
        """
        indices = self.from_time_to_n_batch(t)
        Ks, k0s = (
            self.Kt[indices],
            self.k0t[indices],
        )  # shape: (n_batch, nu, nx), (n_batch, nu, 1)
        return (-torch.bmm(Ks, x.unsqueeze(dim=-1)) - k0s).squeeze()

    def convert_time_to_go_to_time(self, time_to_go: torch.Tensor) -> torch.Tensor:
        """Assuming time_to_go=time to go to tf,
        t0 ... t <time to go> tf: time_to_go = tf - t
        """
        return self.tf - time_to_go

    def __eq__(self, other: "LinearQuadraticControllerTorch"):
        if self.t0 == other.t0 and self.tf == other.tf and self.dt == other.dt:
            if torch.norm(self.k0t - other.k0t) <= 1e-15:
                return torch.norm(self.Kt - other.Kt) <= 1e-15
        return False


class ShiftLinearQuadraticControllerAtFixedPointTorch(SaveLoad):
    def __init__(
        self,
        x0: Union[np.ndarray, torch.Tensor],
        u0: Union[np.ndarray, torch.Tensor],
        controller,
    ):
        self.controller = controller
        self.x0 = (
            torch.tensor(x0)
            if not isinstance(x0, torch.Tensor)
            else x0.detach().clone()
        )
        self.u0 = (
            torch.tensor(u0)
            if not isinstance(u0, torch.Tensor)
            else u0.detach().clone()
        )

    def save_impl(self, buffer, *args, **kwargs):
        controller_buffer = io.BytesIO()
        controller_buffer.seek(0)
        self.controller.save(controller_buffer)
        data_dict = {
            "x0": self.x0,
            "u0": self.u0,
            "controller_cls": self.controller.__class__,
            "controller_buff": controller_buffer,
        }
        pickle.dump(data_dict, buffer)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs):
        # data_dict = pickle.load(buff)
        data_dict = AutoTorchDeviceUnpickler(buff).load()
        controller_cls = data_dict.pop("controller_cls")
        controller_buff = data_dict.pop("controller_buff")
        controller_buff.seek(0)
        controller = controller_cls.load(controller_buff)
        return cls(
            **data_dict,
            controller=controller,
        )

    def to(self, *args, **kwargs):
        self.controller.to(*args, **kwargs)
        self.x0 = self.x0.to(*args, **kwargs)
        self.u0 = self.u0.to(*args, **kwargs)
        return self

    def get_control(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """x: (n_batch, nx)
        self.x0: (nx,)
        u_out: (n_batch, nu)
        self.u0: (nu,)
        """
        return self.controller.get_control(t, x - self.x0) + self.u0

    def get_control_batch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.controller.get_control_batch(t, x - self.x0) + self.u0

    def convert_time_to_go_to_time(self, time_to_go: torch.Tensor) -> torch.Tensor:
        """t0 ... tf: time_to_go = tf - t"""
        return self.controller.convert_time_to_go_to_time(time_to_go)


class AtFixedPointLinearQuadraticControllerTorch(
    ShiftLinearQuadraticControllerAtFixedPointTorch
):
    """This is the LQR controller for using.
     Previous controller assumes expansion at origin while this class also do the shifting.
    If the linear dynamics is also needed (e.g. step_forward, get_value, etc.),
    use LinearDynamicsAtFixedPointWithControllerTorch instead
    """

    @classmethod
    def construct_lqr(
        cls,
        dynamics,
        x0: np.ndarray,
        u0: np.ndarray,
        t0: float,
        tf: float,
        dt: float,
        quadratic_cost_matrices: ConstantQuadraticControlMatrices,
        drop_last: float = 0.0,
    ):
        logging.info("Construct finite horizon LQR at fixed point.")
        linear_dynamics = LinearDynamicsAtFixedPoint(dynamics, x0, u0)
        controller = LinearQuadraticControllerTorch.construct_lqr(
            dynamics=linear_dynamics,
            quadratic_cost_matrices=quadratic_cost_matrices,
            t0=t0,
            tf=tf,
            dt=dt,
            drop_last=drop_last,
        )
        return cls(x0, u0, controller)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs):
        # data_dict = pickle.load(buff)
        data_dict = AutoTorchDeviceUnpickler(buff).load()
        # Support for models saved by old api
        if "controller" in data_dict:
            controller = LinearQuadraticControllerTorch(**data_dict.pop("controller"))
            return cls(
                **data_dict,
                controller=controller,
            )
        buff.seek(0)
        return ShiftLinearQuadraticControllerAtFixedPointTorch.load(buff)
